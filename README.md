# RAG QA PDF - Hệ thống Hỏi Đáp Thông minh với PDF

Một ứng dụng RAG (Retrieval-Augmented Generation) cho phép bạn trò chuyện với tài liệu PDF bằng tiếng Việt.

## 🚀 Tính năng

- **Upload PDF**: Tải lên và xử lý tài liệu PDF
- **Trò chuyện thông minh**: Đặt câu hỏi về nội dung tài liệu
- **Hỗ trợ tiếng Việt**: Tối ưu cho ngôn ngữ tiếng Việt
- **Giao diện thân thiện**: Sử dụng Streamlit với UI đẹp mắt
- **Semantic Search**: Tìm kiếm ngữ nghĩa chính xác

## 📋 Yêu cầu hệ thống

- Python 3.8+
- CUDA (khuyến nghị cho GPU)
- RAM: tối thiểu 8GB (khuyến nghị 16GB+)

## 🛠️ Cài đặt

1. **Clone repository**:
```bash
git clone <repository-url>
cd RAG_QA_PDF
```

2. **Tạo virtual environment**:
```bash
python -m venv rag_qa_pdf
source rag_qa_pdf/bin/activate  # Linux/Mac
# hoặc
rag_qa_pdf\Scripts\activate     # Windows
```

3. **Cài đặt dependencies**:
```bash
pip install -r requirements.txt
```

4. **Cấu hình môi trường**:
```bash
cp .env.example .env
# Chỉnh sửa file .env nếu cần
```

## 🎯 Cách sử dụng

### Chạy ứng dụng Streamlit

```bash
streamlit run main.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

### Sử dụng giao diện

1. **Tải models**: Đợi hệ thống tải các AI models (lần đầu sẽ mất thời gian)
2. **Upload PDF**: Chọn file PDF từ sidebar
3. **Xử lý tài liệu**: Nhấn "Xử lý PDF" để phân tích tài liệu
4. **Đặt câu hỏi**: Nhập câu hỏi trong ô chat
5. **Nhận trả lời**: AI sẽ trả lời dựa trên nội dung PDF

### Demo với Python script

```bash
python demo.py
```

## 📁 Cấu trúc dự án

```
RAG_QA_PDF/
├── main.py                 # Ứng dụng Streamlit chính
├── demo.py                 # Script demo
├── requirements.txt        # Dependencies
├── .env                   # Cấu hình môi trường
├── README.md              # Hướng dẫn này
├── src/                   # Source code
│   ├── llm.py            # Language Model
│   ├── loader.py         # PDF Loader
│   ├── semantic_splitter.py  # Document Splitter
│   └── rag_chain.py      # RAG Chain
├── model_embedding/       # Embedding models
│   └── model.py
├── vectordb/             # Vector database
│   └── chroma.py
└── data/                 # Thư mục chứa PDF files
```

## ⚙️ Cấu hình

### File .env

```env
MODEL_NAME=lmsys/vicuna-7b-v1.5
```

### Thay đổi model

Bạn có thể thay đổi model trong file `.env`:
- `lmsys/vicuna-7b-v1.5` (mặc định)
- `microsoft/DialoGPT-medium`
- Hoặc bất kỳ model nào tương thích với Hugging Face

## 🔧 Troubleshooting

### Lỗi thường gặp

1. **Out of Memory**:
   - Giảm batch size
   - Sử dụng quantization
   - Chuyển sang model nhỏ hơn

2. **CUDA not available**:
   - Cài đặt PyTorch với CUDA support
   - Hoặc chạy trên CPU (chậm hơn)

3. **Model loading failed**:
   - Kiểm tra kết nối internet
   - Xóa cache: `~/.cache/huggingface/`

### Performance Tips

- **GPU**: Sử dụng GPU để tăng tốc độ
- **RAM**: Đảm bảo đủ RAM cho model
- **SSD**: Sử dụng SSD để tăng tốc I/O

## 📊 Hiệu suất

| Component | Model | Size | Speed |
|-----------|-------|------|-------|
| Embedding | vietnamese-bi-encoder | ~400MB | Fast |
| LLM | vicuna-7b-v1.5 | ~13GB | Medium |
| Vector DB | ChromaDB | Variable | Fast |

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 🆘 Hỗ trợ

Nếu gặp vấn đề, hãy:
1. Kiểm tra phần Troubleshooting
2. Tạo issue trên GitHub
3. Liên hệ team phát triển

## 🔄 Cập nhật

Để cập nhật lên phiên bản mới:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

**Phát triển bởi**: RAG Team  
**Phiên bản**: 1.0.0  
**Cập nhật**: 2024