from model_embedding.model import ModelEmbedding
from src.llm import LLMModel
import os
import tempfile
import time
from src.loader import PDFLoader
from src.semantic_splitter import SemanticChunker
from vectordb.chroma import ChromaVectorStore
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
# Session state initialization
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = ""

@st.cache_resource
def load_embeddings():
    """Load HuggingFace embeddings for vector store"""
    try:
        model = ModelEmbedding("bkai-foundation-models/vietnamese-bi-encoder")
        return model.return_model()
    except Exception as e:
        logger.warning(f"Failed to load Vietnamese model, using backup: {e}")
        model = ModelEmbedding("all-MiniLM-L6-v2")
        return model.return_model()

@st.cache_resource
def load_embedding_model():
    """Load the actual embedding model object for SemanticChunker"""
    try:
        model = ModelEmbedding("bkai-foundation-models/vietnamese-bi-encoder")
        return model.model
    except Exception as e:
        logger.warning(f"Failed to load Vietnamese model, using backup: {e}")
        model = ModelEmbedding("all-MiniLM-L6-v2")
        return model.model

@st.cache_resource
def load_llm():
    """Load language model"""
    model_name = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
    try:
        llm_model = LLMModel(
            model_name=model_name,
            quantization=True
        )
        return llm_model.llm
    except Exception as e:
        logger.error(f"Failed to load LLM {model_name}: {e}")
        # Fallback to smaller model
        logger.info("Trying fallback model...")
        llm_model = LLMModel(
            model_name="microsoft/DialoGPT-small",
            quantization=True
        )
        return llm_model.llm


def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PDFLoader(tmp_file_path)
    documents = loader.load()

    semantic_splitter = SemanticChunker(model_embedding = st.session_state.embedding_model)
    docs = semantic_splitter.split_documents(documents)

    vector_db = ChromaVectorStore(embedding_function = st.session_state.embeddings)
    vector_db.add_documents(docs)
    retriever = vector_db.as_retriever()

    # Create prompt template with fallback
    try:
        from langchain import hub
        prompt = hub.pull("rlm/rag-prompt")
    except Exception as e:
        logger.warning(f"Failed to pull prompt from hub, using fallback: {e}")
        template = """Bạn là trợ lý AI thông minh. Sử dụng thông tin từ tài liệu để trả lời câu hỏi một cách chính xác.

Thông tin từ tài liệu:
{context}

Câu hỏi: {question}

Hướng dẫn:
- Chỉ sử dụng thông tin từ tài liệu được cung cấp
- Nếu không tìm thấy thông tin liên quan, hãy nói "Tôi không tìm thấy thông tin liên quan trong tài liệu"
- Trả lời bằng tiếng Việt
- Trả lời ngắn gọn và chính xác

Trả lời:"""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    os.unlink(tmp_file_path)
    return rag_chain, len(docs)

def add_message(role, content):
    """Thêm tin nhắn vào lịch sử chat"""
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })

def clear_chat():
    """Xóa lịch sử chat"""
    st.session_state.chat_history = []

def display_chat():
    """Hiển thị lịch sử chat"""
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write("Xin chào! Tôi là AI assistant. Hãy upload file PDF và bắt đầu đặt câu hỏi về nội dung tài liệu nhé! 😊")

# UI
def main():
    st.set_page_config(
        page_title="PDF RAG Chatbot",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header with emoji logo
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>🤖 PDF RAG Assistant</h1>
        <p style="color: #666; margin-top: -10px;">Trò chuyện thông minh với tài liệu PDF của bạn</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Cài đặt")

        # Load models
        if not st.session_state.models_loaded:
            st.warning("⏳ Đang tải models...")
            with st.spinner("Đang tải AI models..."):
                try:
                    st.session_state.embeddings = load_embeddings()
                    st.session_state.embedding_model = load_embedding_model()
                    st.session_state.llm = load_llm()
                    st.session_state.models_loaded = True
                    logger.info("All models loaded successfully!")
                except Exception as e:
                    st.error(f"Lỗi tải models: {str(e)}")
                    logger.error(f"Model loading failed: {e}")
                    st.stop()
            st.success("✅ Models đã sẵn sàng!")
            st.rerun()
        else:
            st.success("✅ Models đã sẵn sàng!")

        st.markdown("---")

        # Upload PDF
        st.subheader("📄 Upload tài liệu")
        uploaded_file = st.file_uploader("Chọn file PDF", type="pdf")

        if uploaded_file:
            if st.button("🔄 Xử lý PDF", use_container_width=True):
                with st.spinner("Đang xử lý PDF..."):
                    st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_file.name
                    # Reset chat history khi upload PDF mới
                    clear_chat()
                    add_message("assistant", f"✅ Đã xử lý thành công file **{uploaded_file.name}**!\n\n📊 Tài liệu được chia thành {num_chunks} phần. Bạn có thể bắt đầu đặt câu hỏi về nội dung tài liệu.")
                st.rerun()

        # PDF status
        if st.session_state.pdf_processed:
            st.success(f"📄 Đã tải: {st.session_state.pdf_name}")
        else:
            st.info("📄 Chưa có tài liệu")

        st.markdown("---")

        # Chat controls
        st.subheader("💬 Điều khiển Chat")
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            clear_chat()
            st.rerun()

        st.markdown("---")

        # Instructions
        st.subheader("📋 Hướng dẫn")
        st.markdown("""
        **Cách sử dụng:**
        1. **Upload PDF** - Chọn file và nhấn "Xử lý PDF"
        2. **Đặt câu hỏi** - Nhập câu hỏi trong ô chat
        3. **Nhận trả lời** - AI sẽ trả lời dựa trên nội dung PDF
        """)

    # Main content
    st.markdown("*Trò chuyện với Chatbot để trao đổi về nội dung tài liệu PDF của bạn*")

    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display chat history
        display_chat()

    # Chat input
    if st.session_state.models_loaded:
        if st.session_state.pdf_processed:
            # User input
            user_input = st.chat_input("Nhập câu hỏi của bạn...")

            if user_input:
                # Add user message
                add_message("user", user_input)

                # Display user message immediately
                with st.chat_message("user"):
                    st.write(user_input)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Đang suy nghĩ..."):
                        try:
                            output = st.session_state.rag_chain.invoke(user_input)
                            # Clean up the response
                            if 'Answer:' in output:
                                answer = output.split('Answer:')[1].strip()
                            else:
                                answer = output.strip()

                            # Display response
                            st.write(answer)

                            # Add assistant message to history
                            add_message("assistant", answer)

                        except Exception as e:
                            error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
                            st.error(error_msg)
                            add_message("assistant", error_msg)
        else:
            st.info("🔄 Vui lòng upload và xử lý file PDF trước khi bắt đầu chat!")
            st.chat_input("Nhập câu hỏi của bạn...", disabled=True)
    else:
        st.info("⏳ Đang tải AI models, vui lòng đợi...")
        st.chat_input("Nhập câu hỏi của bạn...", disabled=True)

if __name__ == "__main__":
    main()



