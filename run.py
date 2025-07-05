#!/usr/bin/env python3
"""
Fixed run.py for Google Colab - No ngrok needed
Usage: !python run_fixed.py
"""

import os
import sys
import subprocess
import threading
import time

from pyngrok import ngrok
ngrok.set_auth_token("25ewFLxDjazgSkTITOAkIJlV4g7_7xHVgr5iVdDsT2gg11Vvb")

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        "streamlit",
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

def create_project_files():
    """Create all necessary project files"""
    print("📁 Creating project files...")
    
    # Create directories
    os.makedirs('model_embedding', exist_ok=True)
    os.makedirs('src', exist_ok=True)
    os.makedirs('vectordb', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Create model_embedding/model.py
    model_embedding_code = '''
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

class ModelEmbedding:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading {self.model_name}, using backup model")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_name = 'all-MiniLM-L6-v2'
    
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
            print(f"Loading LLM: {self.model_name}")
            
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
                device_map="auto"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
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
                temperature=0.7
            )
            
            self.llm = HuggingFacePipeline(pipeline=self.model_pipeline)
            print("✅ LLM loaded successfully!")
            
        except Exception as e:
            print(f"Error loading LLM: {str(e)}")
            raise
    
    def generate(self, prompt: str):
        return self.llm(prompt)
'''
    
    with open('src/llm.py', 'w') as f:
        f.write(llm_code)
    
    # Create src/loader.py
    loader_code = '''
from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = PyPDFLoader(file_path)
    
    def load(self):
        return self.loader.load()
'''
    
    with open('src/loader.py', 'w') as f:
        f.write(loader_code)
    
    # Create src/semantic_splitter.py
    splitter_code = '''
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SemanticChunker:
    def __init__(self, model_embedding=None, chunk_size=800, chunk_overlap=100):
        self.model_embedding = model_embedding
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def split_documents(self, documents):
        return self.text_splitter.split_documents(documents)
'''
    
    with open('src/semantic_splitter.py', 'w') as f:
        f.write(splitter_code)
    
    # Create vectordb/chroma.py
    chroma_code = '''
from langchain_community.vectorstores import Chroma
import tempfile

class ChromaVectorStore:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.vectorstore = None
        self.persist_directory = tempfile.mkdtemp()
    
    def add_documents(self, documents):
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(documents)
    
    def as_retriever(self, search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
'''
    
    with open('vectordb/chroma.py', 'w') as f:
        f.write(chroma_code)
    
    # Create .env file
    with open('.env', 'w') as f:
        f.write('MODEL_NAME=microsoft/DialoGPT-medium')
    
    # Create FIXED main.py
    main_content = '''
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
    try:
        model = ModelEmbedding("bkai-foundation-models/vietnamese-bi-encoder")
        return model.return_model()
    except:
        model = ModelEmbedding("all-MiniLM-L6-v2")
        return model.return_model()

@st.cache_resource
def load_embedding_model():
    """Load the actual embedding model object for SemanticChunker"""
    try:
        model = ModelEmbedding("bkai-foundation-models/vietnamese-bi-encoder")
        return model.model
    except:
        model = ModelEmbedding("all-MiniLM-L6-v2")
        return model.model

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

    # Use the actual embedding model for SemanticChunker
    semantic_splitter = SemanticChunker(model_embedding=st.session_state.embedding_model)
    docs = semantic_splitter.split_documents(documents)

    vector_db = ChromaVectorStore(embedding_function=st.session_state.embeddings)
    vector_db.add_documents(docs)
    retriever = vector_db.as_retriever()

    # Create prompt template (fallback if hub.pull fails)
    try:
        from langchain import hub
        prompt = hub.pull("rlm/rag-prompt")
    except:
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}

        Helpful Answer:"""
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
        <p style="margin-top: -10px;">Chạy trên Google Colab - Fixed Version</p>
    </div>
    """, unsafe_allow_html=True)

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
                    st.session_state.embedding_model = load_embedding_model()
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
                with st.spinner("��ang xử lý PDF..."):
                    try:
                        st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = uploaded_file.name
                        clear_chat()
                        add_message("assistant", f"✅ Đã xử lý thành công file **{uploaded_file.name}**!\\n\\n📊 Tài liệu được chia thành {num_chunks} phần.")
                    except Exception as e:
                        st.error(f"L��i xử lý PDF: {str(e)}")
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

        st.markdown("---")
        
        # Instructions
        st.subheader("📋 Hướng dẫn")
        st.markdown("""
        **Cách sử dụng:**
        1. Upload file PDF
        2. Nhấn "Xử lý PDF"
        3. Đặt câu hỏi trong chat
        
        **🔧 Fixed Issues:**
        - SemanticChunker embedding model
        - format_docs function
        - Error handling
        """)

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
                            if 'Answer:' in output:
                                answer = output.split('Answer:')[1].strip()
                            else:
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
    
    with open('main.py', 'w') as f:
        f.write(main_content)
    
    print("✅ All project files created with fixes!")

def run_streamlit():
    """Run Streamlit server"""
    print("🚀 Starting Streamlit...")
    
    # Instructions
    print("\n" + "="*50)
    print("📋 INSTRUCTIONS:")
    print("1. Streamlit will start automatically")
    print("2. Look for 'Open in new tab' button in output")
    print("3. Or check the port 8501 in Colab")
    print("4. Upload PDF and start chatting!")
    print("="*50 + "\n")
    
    # Run Streamlit with Colab-friendly settings
    subprocess.run([
        "streamlit", "run", "main.py",
        "--server.port=8501",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.allowRunOnSave=true"
    ])

def main():
    """Main function"""
    print("🚀 RAG QA PDF - Fixed Colab Runner")
    print("=" * 50)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        
        # Enable Colab's port forwarding
        from google.colab import output
        print("🌐 Enabling Colab port forwarding...")
        
    except ImportError:
        print("⚠️ Not in Google Colab")
    
    # Install dependencies
    install_dependencies()
    
    # Create project files with fixes
    create_project_files()
    
    # Run Streamlit
    run_streamlit()

if __name__ == "__main__":
    main()