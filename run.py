#!/usr/bin/env python3
"""
Simple run.py for Google Colab
Usage: !python run.py
"""

import os
import sys
import subprocess
import threading
import time

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
        except:
            print(f"⚠️ Failed to install {package}")
    
    print("✅ Dependencies installed!")

def setup_ngrok():
    """Setup ngrok"""
    print("🌐 Setting up ngrok...")
    
    if not os.path.exists("ngrok"):
        subprocess.run(["wget", "-q", "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip"])
        subprocess.run(["unzip", "-qq", "ngrok-stable-linux-amd64.zip"])
    
    print("✅ Ngrok ready!")

def check_main_py():
    """Check if main.py exists, if not create a simple version"""
    if not os.path.exists("main.py"):
        print("📝 Creating main.py...")
        
        # Create a simple main.py based on existing structure
        main_content = '''
import streamlit as st
import os
import tempfile
import time
from pathlib import Path

# Simple fallback if other modules don't exist
try:
    from model_embedding.model import ModelEmbedding
    from src.llm import LLMModel
    from src.loader import PDFLoader
    from src.semantic_splitter import SemanticChunker
    from vectordb.chroma import ChromaVectorStore
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    st.error("❌ Required modules not found. Please ensure all project files are present.")

if MODULES_AVAILABLE:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.prompts import PromptTemplate
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Session state initialization
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
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
        try:
            model = ModelEmbedding("bkai-foundation-models/vietnamese-bi-encoder")
            return model.return_model()
        except:
            model = ModelEmbedding("all-MiniLM-L6-v2")
            return model.return_model()

    @st.cache_resource
    def load_llm():
        model_name = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
        llm_model = LLMModel(
            model_name=model_name,
            quantization=True
        )
        return llm_model.llm

    def process_pdf(uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PDFLoader(tmp_file_path)
        documents = loader.load()

        semantic_splitter = SemanticChunker(model_embedding=st.session_state.embeddings)
        docs = semantic_splitter.split_documents(documents)

        vector_db = ChromaVectorStore(embedding_function=st.session_state.embeddings)
        vector_db.add_documents(docs)
        retriever = vector_db.as_retriever()

        template = """Bạn là trợ lý AI thông minh. Sử dụng thông tin từ tài liệu để trả lời câu hỏi.

Thông tin từ tài liệu:
{context}

Câu hỏi: {question}

Trả lời ngắn gọn bằng tiếng Việt:"""

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

        os.unlink(tmp_file_path)
        return rag_chain, len(docs)

    def add_message(role, content):
        st.session_state.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

    def clear_chat():
        st.session_state.chat_history = []

    def display_chat():
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
    st.set_page_config(
        page_title="PDF RAG Chatbot - Google Colab",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
        <h1>🤖 PDF RAG Assistant</h1>
        <p style="margin-top: -10px;">Chạy trên Google Colab</p>
    </div>
    """, unsafe_allow_html=True)

    if not MODULES_AVAILABLE:
        st.error("❌ Vui lòng đảm bảo tất cả các file cần thiết đã có trong project!")
        st.info("💡 Chạy lại setup hoặc kiểm tra các file: model_embedding/, src/, vectordb/")
        return

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Cài đặt")
        st.success("🚀 Đang chạy trên Google Colab")

        # Load models
        if not st.session_state.models_loaded:
            st.warning("⏳ Đang tải models...")
            with st.spinner("Đang tải AI models..."):
                try:
                    st.session_state.embeddings = load_embeddings()
                    st.session_state.llm = load_llm()
                    st.session_state.models_loaded = True
                except Exception as e:
                    st.error(f"Lỗi tải models: {str(e)}")
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
                        add_message("assistant", f"✅ Đã xử lý thành công file **{uploaded_file.name}**!\\n\\n📊 Tài liệu được chia thành {num_chunks} phần.")
                    except Exception as e:
                        st.error(f"Lỗi xử lý PDF: {str(e)}")
                st.rerun()

        # PDF status
        if st.session_state.pdf_processed:
            st.success(f"📄 Đã tải: {st.session_state.pdf_name}")
        else:
            st.info("📄 Chưa có tài liệu")

        st.markdown("---")

        # Chat controls
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            clear_chat()
            st.rerun()

    # Main content
    st.markdown("*Trò chuyện với AI về nội dung tài liệu PDF*")

    # Chat container
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
                            answer = output.strip()
                            st.write(answer)
                            add_message("assistant", answer)
                        except Exception as e:
                            error_msg = f"Xin lỗi, đã có lỗi xảy ra: {str(e)}"
                            st.error(error_msg)
                            add_message("assistant", error_msg)
        else:
            st.info("🔄 Vui lòng upload và xử lý file PDF trước!")
            st.chat_input("Nhập câu hỏi của bạn...", disabled=True)
    else:
        st.info("⏳ Đang tải AI models...")
        st.chat_input("Nhập câu hỏi của bạn...", disabled=True)

if __name__ == "__main__":
    main()
'''
        
        with open("main.py", "w", encoding="utf-8") as f:
            f.write(main_content)
        
        print("✅ main.py created!")

def run_streamlit_with_ngrok():
    """Run Streamlit with ngrok tunnel"""
    print("🚀 Starting Streamlit with ngrok...")
    
    # Import ngrok
    try:
        from pyngrok import ngrok
    except ImportError:
        print("❌ pyngrok not installed!")
        return
    
    def run_streamlit():
        """Run Streamlit server"""
        subprocess.run([
            "streamlit", "run", "main.py", 
            "--server.port=8501", 
            "--server.headless=true"
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
        
        print(f"\n🎉 SUCCESS!")
        print(f"🔗 Public URL: {public_url}")
        print(f"\n📱 Instructions:")
        print(f"1. Click the URL above")
        print(f"2. Open in new tab")
        print(f"3. Use Streamlit normally")
        print(f"\n⚠️ Keep this running to maintain connection")
        
        # Show tunnel info
        tunnels = ngrok.get_tunnels()
        if tunnels:
            print(f"\n🔧 Tunnel info:")
            for tunnel in tunnels:
                print(f"   Local: {tunnel.config['addr']}")
                print(f"   Public: {tunnel.public_url}")
        
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
            
    except Exception as e:
        print(f"❌ Error creating tunnel: {str(e)}")
        print(f"💡 Try running again or check ngrok setup")

def main():
    """Main function"""
    print("🚀 RAG QA PDF - Google Colab Runner")
    print("=" * 50)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("⚠️ Not in Google Colab - some features may not work")
    
    # Install dependencies
    install_dependencies()
    
    # Setup ngrok
    setup_ngrok()
    
    # Check/create main.py
    check_main_py()
    
    # Run Streamlit with ngrok
    run_streamlit_with_ngrok()

if __name__ == "__main__":
    main()