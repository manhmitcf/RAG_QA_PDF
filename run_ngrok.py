#!/usr/bin/env python3
"""
Complete run.py for Google Colab with ngrok authtoken from .env
Usage: !python run_ngrok.py
"""

import os
import sys
import subprocess
import threading
import time
import socket
from contextlib import closing

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        "streamlit",
        "pyngrok",
        "transformers==4.52.4",
        "bitsandbytes==0.46.0", 
        "accelerate==1.7.0",
        "langchain==0.3.25",
        "langchainhub==0.1.21",
        "langchain-chroma==0.2.4",
        "langchain-community==0.3.24",
        "langchain_huggingface==0.2.0",
        "python-dotenv==1.1.0",
        "pypdf",
        "sentence-transformers",
        "chromadb"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        except Exception as e:
            print(f"⚠️ Failed to install {package}: {e}")
    
    print("✅ Dependencies installed!")

def setup_ngrok():
    """Setup ngrok with authtoken from .env"""
    print("🌐 Setting up ngrok...")
    
    # Download ngrok if not exists
    if not os.path.exists("ngrok"):
        try:
            subprocess.run(["wget", "-q", "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip"], check=True)
            subprocess.run(["unzip", "-qq", "ngrok-stable-linux-amd64.zip"], check=True)
            print("✅ Ngrok downloaded!")
        except Exception as e:
            print(f"❌ Failed to download ngrok: {e}")
            return False
    
    # Load authtoken from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        authtoken = os.getenv("NGROK_AUTHTOKEN")
        if not authtoken or authtoken == "your_ngrok_authtoken_here":
            print("❌ NGROK_AUTHTOKEN not found in .env file!")
            print("💡 Please:")
            print("   1. Sign up at: https://dashboard.ngrok.com/signup")
            print("   2. Get your authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken")
            print("   3. Add it to .env file: NGROK_AUTHTOKEN=your_actual_token")
            return False
        
        # Set authtoken
        from pyngrok import ngrok
        ngrok.set_auth_token(authtoken)
        print("✅ Ngrok authtoken configured!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up ngrok: {e}")
        return False

def create_project_files():
    """Create all necessary project files"""
    print("📁 Creating project files...")
    
    # Create directories
    os.makedirs('model_embedding', exist_ok=True)
    os.makedirs('src', exist_ok=True)
    os.makedirs('vectordb', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Create model_embedding/__init__.py
    with open('model_embedding/__init__.py', 'w') as f:
        f.write('')
    
    # Create src/__init__.py
    with open('src/__init__.py', 'w') as f:
        f.write('')
    
    # Create vectordb/__init__.py
    with open('vectordb/__init__.py', 'w') as f:
        f.write('')
    
    # Create model_embedding/model.py
    model_embedding_code = '''
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class ModelEmbedding:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Embedding model loaded successfully!")
        except Exception as e:
            logger.warning(f"Error loading {self.model_name}, using backup model: {e}")
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model_name = 'all-MiniLM-L6-v2'
                logger.info("✅ Backup embedding model loaded!")
            except Exception as e2:
                logger.error(f"Failed to load backup model: {e2}")
                raise
    
    def return_model(self):
        return HuggingFaceEmbeddings(model_name=self.model_name)
'''
    
    with open('model_embedding/model.py', 'w') as f:
        f.write(model_embedding_code)
    
    # Create src/llm.py
    llm_code = '''
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from langchain_huggingface.llms import HuggingFacePipeline
import logging

logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self, model_name: str, quantization=True):
        self.model_name = model_name
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self.llm = None
        self.load_model()
    
    def load_model(self):
        try:
            logger.info(f"Loading LLM: {self.model_name}")
            
            if self.quantization:
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                quantization_config = nf4_config
            else:
                quantization_config = None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                pad_token_id=self.tokenizer.eos_token_id,
                device_map="auto",
                do_sample=True,
                temperature=0.7,
                return_full_text=False
            )
            
            self.llm = HuggingFacePipeline(pipeline=self.model_pipeline)
            logger.info("✅ LLM loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading LLM: {str(e)}")
            raise
    
    def generate(self, prompt: str):
        return self.llm(prompt)
'''
    
    with open('src/llm.py', 'w') as f:
        f.write(llm_code)
    
    # Create src/loader.py
    loader_code = '''
from langchain_community.document_loaders import PyPDFLoader
import logging

logger = logging.getLogger(__name__)

class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = PyPDFLoader(file_path)
    
    def load(self):
        try:
            documents = self.loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise
'''
    
    with open('src/loader.py', 'w') as f:
        f.write(loader_code)
    
    # Create src/semantic_splitter.py
    splitter_code = '''
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

class SemanticChunker:
    def __init__(self, model_embedding=None, chunk_size=800, chunk_overlap=100):
        self.model_embedding = model_embedding
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def split_documents(self, documents):
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
'''
    
    with open('src/semantic_splitter.py', 'w') as f:
        f.write(splitter_code)
    
    # Create vectordb/chroma.py
    chroma_code = '''
from langchain_community.vectorstores import Chroma
import tempfile
import logging

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.vectorstore = None
        self.persist_directory = tempfile.mkdtemp()
    
    def add_documents(self, documents):
        try:
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_function,
                    persist_directory=self.persist_directory
                )
                logger.info("Created new Chroma vectorstore")
            else:
                self.vectorstore.add_documents(documents)
                logger.info("Added documents to existing vectorstore")
        except Exception as e:
            logger.error(f"Error adding documents to vectorstore: {e}")
            raise
    
    def as_retriever(self, search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
'''
    
    with open('vectordb/chroma.py', 'w') as f:
        f.write(chroma_code)
    
    print("✅ Project files created!")

def create_streamlit_app():
    """Create the main Streamlit application"""
    print("🎨 Creating Streamlit app...")
    
    main_content = '''
from model_embedding.model import ModelEmbedding
from src.llm import LLMModel
import os
import tempfile
import time
import logging
from src.loader import PDFLoader
from src.semantic_splitter import SemanticChunker
from vectordb.chroma import ChromaVectorStore
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

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
    """Process uploaded PDF file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load documents
        loader = PDFLoader(tmp_file_path)
        documents = loader.load()

        # Split documents using the actual embedding model
        semantic_splitter = SemanticChunker(model_embedding=st.session_state.embedding_model)
        docs = semantic_splitter.split_documents(documents)

        # Create vector store
        vector_db = ChromaVectorStore(embedding_function=st.session_state.embeddings)
        vector_db.add_documents(docs)
        retriever = vector_db.as_retriever()

        # Create prompt template
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
            return "\\n\\n".join([doc.page_content for doc in docs])
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | st.session_state.llm
            | StrOutputParser()
        )

        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        logger.info(f"Successfully processed PDF with {len(docs)} chunks")
        return rag_chain, len(docs)
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise

def add_message(role, content):
    """Add message to chat history"""
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": time.time()
    })

def clear_chat():
    """Clear chat history"""
    st.session_state.chat_history = []

def display_chat():
    """Display chat history"""
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

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="PDF RAG Chatbot - Google Colab",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🤖"
    )
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
        <h1>🤖 PDF RAG Assistant</h1>
        <p style="margin-top: -10px;">Chạy trên Google Colab với Ngrok</p>
        <p style="font-size: 0.8em; opacity: 0.8;">🌐 Powered by Ngrok Tunnel</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Cài đặt")
        st.success("🚀 Đang chạy trên Google Colab")
        st.info("🌐 Sử dụng Ngrok tunnel")

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
                    try:
                        st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        clear_chat()
                        add_message("assistant", f"✅ Đã xử lý thành công file **{uploaded_file.name}**!\\n\\n📊 Tài liệu được chia thành {num_chunks} phần. Bạn có thể bắt đầu đặt câu hỏi về nội dung tài liệu.")
                        logger.info(f"PDF processed successfully: {uploaded_file.name}")
                    except Exception as e:
                        error_msg = f"Lỗi xử lý PDF: {str(e)}"
                        st.error(error_msg)
                        logger.error(error_msg)
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
        
        # Model info
        st.subheader("🔧 Thông tin Model")
        model_name = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
        st.info(f"**LLM**: {model_name}")
        st.info(f"**Embedding**: Vietnamese-bi-encoder")
        
        st.markdown("---")
        
        # Instructions
        st.subheader("📋 Hướng dẫn")
        st.markdown("""
        **Cách sử dụng:**
        1. **Upload PDF** - Chọn file và nhấn "Xử lý PDF"
        2. **Đặt câu hỏi** - Nhập câu hỏi trong ô chat
        3. **Nhận trả lời** - AI sẽ trả lời dựa trên nội dung PDF
        
        **💡 Tips:**
        - Đặt câu hỏi cụ thể
        - Sử dụng tiếng Việt
        - Kiên nhẫn chờ AI xử lý
        
        **🔧 Ngrok Setup:**
        - Cần NGROK_AUTHTOKEN trong .env
        - Đăng ký t���i: ngrok.com
        """)

    # Main content
    st.markdown("*Trò chuyện với AI về nội dung tài liệu PDF của bạn*")

    # Chat container
    chat_container = st.container()
    with chat_container:
        display_chat()

    # Chat input
    if st.session_state.models_loaded:
        if st.session_state.pdf_processed:
            user_input = st.chat_input("Nhập câu hỏi của bạn...")

            if user_input:
                add_message("user", user_input)

                with st.chat_message("user"):
                    st.write(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Đang suy nghĩ..."):
                        try:
                            output = st.session_state.rag_chain.invoke(user_input)
                            
                            # Clean up the response
                            if isinstance(output, str):
                                if 'Answer:' in output:
                                    answer = output.split('Answer:')[1].strip()
                                elif 'Trả lời:' in output:
                                    answer = output.split('Trả lời:')[1].strip()
                                else:
                                    answer = output.strip()
                            else:
                                answer = str(output).strip()

                            st.write(answer)
                            add_message("assistant", answer)
                            logger.info(f"Question answered: {user_input[:50]}...")

                        except Exception as e:
                            error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
                            st.error(error_msg)
                            add_message("assistant", error_msg)
                            logger.error(f"Error answering question: {e}")
        else:
            st.info("🔄 Vui lòng upload và xử lý file PDF trước khi bắt đầu chat!")
            st.chat_input("Nhập câu hỏi của bạn...", disabled=True)
    else:
        st.info("⏳ Đang tải AI models, vui lòng đợi...")
        st.chat_input("Nhập câu hỏi của bạn...", disabled=True)

if __name__ == "__main__":
    main()
'''
    
    with open('main.py', 'w') as f:
        f.write(main_content)
    
    print("✅ Streamlit app created!")

def run_streamlit_with_ngrok():
    """Run Streamlit with ngrok tunnel"""
    print("🚀 Starting Streamlit with ngrok...")
    
    # Import ngrok
    try:
        from pyngrok import ngrok
    except ImportError:
        print("❌ pyngrok not installed!")
        return False
    
    def run_streamlit():
        """Run Streamlit server"""
        subprocess.run([
            "streamlit", "run", "main.py", 
            "--server.port=8501", 
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false"
        ])
    
    # Start Streamlit in background thread
    print("🔄 Starting Streamlit server...")
    streamlit_thread = threading.Thread(target=run_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()
    
    # Wait for Streamlit to start
    print("⏳ Waiting for Streamlit to start...")
    time.sleep(10)
    
    # Create ngrok tunnel
    try:
        print("🌐 Creating ngrok tunnel...")
        public_url = ngrok.connect(8501)
        
        print(f"\n🎉 SUCCESS! Streamlit is ready!")
        print(f"🔗 Public URL: {public_url}")
        print(f"\n📱 Instructions:")
        print(f"1. Click the URL above")
        print(f"2. Open in new tab")
        print(f"3. Use Streamlit normally")
        print(f"4. Share the URL with others if needed")
        print(f"\n⚠️ Keep this running to maintain the connection")
        
        # Show tunnel info
        tunnels = ngrok.get_tunnels()
        if tunnels:
            print(f"\n🔧 Tunnel info:")
            for tunnel in tunnels:
                print(f"   - Local: {tunnel.config['addr']}")
                print(f"   - Public: {tunnel.public_url}")
                print(f"   - Proto: {tunnel.proto}")
        
        # Keep running
        print(f"\n🔄 Running... (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(60)
                print(f"⏰ Still running... {time.strftime('%H:%M:%S')}")
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping...")
            ngrok.disconnect(public_url)
            ngrok.kill()
            return True
            
    except Exception as e:
        print(f"❌ Error creating ngrok tunnel: {str(e)}")
        print(f"💡 Please check:")
        print(f"   1. NGROK_AUTHTOKEN is correct in .env")
        print(f"   2. Internet connection is stable")
        print(f"   3. Try running again")
        return False

def main():
    """Main function"""
    print("🚀 RAG QA PDF - Complete Google Colab Runner with Ngrok")
    print("=" * 70)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("⚠️ Not in Google Colab - some features may not work optimally")
    
    # Install dependencies
    install_dependencies()
    
    # Setup ngrok
    if not setup_ngrok():
        print("❌ Ngrok setup failed. Please check your authtoken.")
        return
    
    # Create project files
    create_project_files()
    
    # Create Streamlit app
    create_streamlit_app()
    
    # Instructions
    print("\n" + "="*70)
    print("📋 READY TO START!")
    print("🔗 Ngrok will create a public URL for your Streamlit app")
    print("🌐 You can share this URL with others")
    print("📱 The app will have full RAG functionality")
    print("="*70 + "\n")
    
    # Run Streamlit with ngrok
    success = run_streamlit_with_ngrok()
    
    if success:
        print("✅ Session completed successfully!")
    else:
        print("❌ Session ended with errors. Please check the logs above.")

if __name__ == "__main__":
    main()