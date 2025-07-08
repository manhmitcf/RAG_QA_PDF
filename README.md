# 🤖 RAG QA PDF

Hệ thống hỏi đáp thông minh với tài liệu PDF sử dụng RAG (Retrieval-Augmented Generation) và AI

## ✨ Tính năng chính

- 📄 **Upload & Xử lý PDF**: Tải lên và phân tích tài liệu PDF tự động
- 💬 **Chat thông minh**: Đặt câu hỏi và nhận trả lời dựa trên nội dung tài liệu
- 🇻🇳 **Hỗ trợ tiếng Việt**: Tối ưu cho ngôn ngữ tiếng Việt với Vietnamese-bi-encoder
- 🌐 **Chạy trên Google Colab**: Sử dụng GPU miễn phí với ngrok tunnel
- 🔍 **Semantic Search**: Tìm kiếm ngữ nghĩa chính xác với ChromaDB
- ⚡ **4-bit Quantization**: Tối ưu bộ nhớ với BitsAndBytesConfig

## 🏗️ Kiến trúc hệ thống

```
📄 PDF Input → 🔄 Document Loader → ✂️ Semantic Splitter → 🗄️ Vector Store → 🔍 Retriever
                                                                                    ↓
🤖 LLM Response ← 📝 Prompt Template ← 🔗 RAG Chain ← 🎯 Context + Question ←────┘
```

## 🚀 Cách sử dụng

### Phương pháp 1: Google Colab (Khuyến nghị)

1. **Lấy Ngrok Authtoken**:
   ```
   - Đăng ký tại: https://dashboard.ngrok.com/signup
   - Copy authtoken từ dashboard
   ```

2. **Cập nhật file .env**:
   ```env
   MODEL_NAME=microsoft/DialoGPT-medium
   NGROK_AUTHTOKEN=your_authtoken_here
   ```

3. **Chạy trên Colab**:
   ```python
   # Upload project files lên Colab
   !python run_ngrok.py
   ```

4. **Sử dụng**:
   - Click vào public URL từ ngrok
   - Upload file PDF qua giao diện
   - Bắt đầu chat với tài liệu!

### Phương pháp 2: Máy local

```bash
# Clone repository
git clone <repository-url>
cd RAG_QA_PDF

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy ứng dụng
streamlit run main.py
```

## 📁 Cấu trúc project

```
RAG_QA_PDF/
├── main.py                    # Ứng dụng Streamlit chính
├── run_ngrok.py              # Script chạy trên Google Colab với ngrok
├── requirements.txt          # Dependencies
├── .env                      # Cấu hình (MODEL_NAME, NGROK_AUTHTOKEN)
├── create_logo.py           # Script tạo logo (optional)
├── data/                    # Thư mục chứa PDF files
├── src/                     # Source code chính
│   ├── llm.py              # Language Model với quantization
│   ├── loader.py           # PDF Document Loader
│   └── semantic_splitter.py # Semantic Document Splitter
├── model_embedding/         # Embedding models
│   └── model.py            # Vietnamese & multilingual embeddings
└── vectordb/               # Vector database
    └── chroma.py           # ChromaDB implementation
```

## 🔧 Cấu hình chi tiết

### Models được hỗ trợ

**Language Models:**
- `microsoft/DialoGPT-medium` (khuyến nghị cho Colab - ~1.5GB)
- `microsoft/DialoGPT-small` (backup model - ~500MB)
- `lmsys/vicuna-7b-v1.5` (chất lượng cao - ~13GB RAM)

**Embedding Models:**
- `bkai-foundation-models/vietnamese-bi-encoder` (chính - tiếng Việt)
- `all-MiniLM-L6-v2` (backup - đa ngôn ngữ)

### File .env

```env
# Language Model (chọn 1 trong các options trên)
MODEL_NAME=microsoft/DialoGPT-medium

# Ngrok Authtoken (lấy từ https://dashboard.ngrok.com/)
NGROK_AUTHTOKEN=your_actual_authtoken_here
```

### Dependencies chính

```
transformers==4.52.4          # Hugging Face Transformers
bitsandbytes==0.46.0          # 4-bit quantization
langchain==0.3.25             # RAG framework
langchain-chroma==0.2.4       # ChromaDB integration
streamlit                     # Web interface
pypdf                         # PDF processing
torch                         # PyTorch backend
```

## 🎯 Workflow hoạt động

1. **Model Loading**: Tải embedding model và LLM với quantization
2. **PDF Processing**: Upload → PyPDFLoader → Semantic splitting
3. **Vector Storage**: Embedding documents → ChromaDB → Retriever
4. **RAG Chain**: Question → Retrieval → Context + Prompt → LLM → Answer
5. **Chat Interface**: Streamlit UI với chat history và controls

## ❓ Troubleshooting

### Lỗi thường gặp

**🔴 Ngrok authtoken không hợp lệ**
```
❌ NGROK_AUTHTOKEN not found in .env file!
```
**Giải pháp**: Đăng ký ngrok.com và thêm authtoken vào .env

**🔴 Out of Memory**
```
❌ CUDA out of memory
```
**Giải pháp**: 
- Dùng model nhỏ hơn: `MODEL_NAME=microsoft/DialoGPT-small`
- Restart Colab runtime
- Sử dụng High-RAM runtime (Colab Pro)

**🔴 Model loading failed**
```
❌ Error loading LLM: Connection timeout
```
**Giải pháp**:
- Kiểm tra kết nối internet
- Thử model backup tự động
- Clear Hugging Face cache: `!rm -rf ~/.cache/huggingface/`

**🔴 SemanticChunker error**
```
❌ SemanticChunker model_embedding error
```
**Giải pháp**: Đã fix trong code với fallback mechanism

### Performance Tips

- 🔋 **GPU Runtime**: Sử dụng GPU T4 trên Colab
- 📝 **Câu hỏi cụ thể**: Đặt câu hỏi rõ ràng để có kết quả tốt
- ⏰ **Kiên nhẫn**: Model loading lần đầu mất 5-10 phút
- 🔄 **Restart**: Restart runtime nếu gặp memory issues

## 🌟 Tính năng nâng cao

### Semantic Splitting
- Sử dụng `SemanticChunker` thay vì split cố định
- Tự động phát hiện ranh giới ngữ nghĩa
- Chunk size linh hoạt: 500-1500 tokens

### Smart Fallbacks
- Tự động chuyển sang backup model nếu main model fail
- Fallback prompt template nếu hub.pull() fail
- Error handling toàn diện

### Optimizations
- 4-bit quantization giảm 75% memory usage
- Streamlit caching cho models
- Efficient vector retrieval với ChromaDB

## 📞 Hỗ trợ

- 📖 **Setup chi tiết**: Đọc file `run_ngrok.py` comments
- 🐛 **Bug reports**: Tạo issue với error logs
- 💬 **Questions**: Liên hệ team development
- 🔧 **Customization**: Modify models trong .env file

## 👥 Contributors

- Trinh Nam Thuan
- Trần Văn Mạnh

## 📊 Benchmark

| Component | Model | Size | Speed | Accuracy |
|-----------|-------|------|-------|----------|
| Embedding | vietnamese-bi-encoder | ~400MB | Fast | High (VN) |
| LLM | DialoGPT-medium | ~1.5GB | Medium | Good |
| Vector DB | ChromaDB | Variable | Fast | High |
| Total RAM | - | ~4-6GB | - | - |

---

**🎉 Chúc bạn sử dụng thành công RAG QA PDF!**

*Developed with ❤️ for Vietnamese AI community*