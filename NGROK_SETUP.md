# 🌐 Hướng dẫn setup Ngrok cho RAG QA PDF

## 🎯 Tổng quan

Ngrok cho phép tạo public URL để truy cập ứng dụng Streamlit từ bất kỳ đâu. Từ 2024, ngrok yêu cầu authtoken (miễn phí).

---

## 🔑 Bước 1: Lấy Ngrok Authtoken

### 1.1 Đăng ký tài khoản
1. Truy cập: https://dashboard.ngrok.com/signup
2. Đăng ký bằng email hoặc GitHub/Google
3. Xác nhận email (nếu cần)

### 1.2 Lấy authtoken
1. Đăng nhập vào: https://dashboard.ngrok.com/
2. Vào mục: **Getting Started** → **Your Authtoken**
3. Copy authtoken (dạng: `2abc123def456ghi789jkl`)

---

## 📝 Bước 2: Cấu hình .env file

### 2.1 Cập nhật file .env
```env
MODEL_NAME=microsoft/DialoGPT-medium
NGROK_AUTHTOKEN=your_actual_authtoken_here
```

### 2.2 Thay thế authtoken
- Thay `your_actual_authtoken_here` bằng authtoken thực tế
- Ví dụ: `NGROK_AUTHTOKEN=2abc123def456ghi789jkl`

---

## �� Bước 3: Chạy ứng dụng

### 3.1 Trên Google Colab
```python
# Upload file .env với authtoken đã cập nhật
# Sau đó chạy:
!python run_ngrok.py
```

### 3.2 Workflow
```
run_ngrok.py
    ↓
Install dependencies
    ↓
Setup ngrok with authtoken
    ↓
Create project files
    ↓
Start Streamlit + ngrok tunnel
    ↓
Get public URL
    ↓
Access from anywhere
```

---

## 🔧 Troubleshooting

### ❌ "authentication failed"
```
ERROR: authentication failed: Usage of ngrok requires a verified account and authtoken.
```

**Giải pháp:**
1. Kiểm tra authtoken trong .env file
2. Đảm bảo không có khoảng trắng thừa
3. Authtoken phải là chuỗi dài ~50 ký tự

### ❌ "NGROK_AUTHTOKEN not found"
```
❌ NGROK_AUTHTOKEN not found in .env file!
```

**Giải pháp:**
1. Tạo/cập nhật file .env
2. Thêm dòng: `NGROK_AUTHTOKEN=your_token`
3. Upload lại file .env lên Colab

### ❌ "tunnel session failed"
```
ERROR: tunnel session failed
```

**Giải pháp:**
1. Kiểm tra kết nối internet
2. Thử chạy lại script
3. Restart Colab runtime nếu cần

### ❌ "account limit exceeded"
```
ERROR: account limit exceeded
```

**Giải pháp:**
1. Tài khoản miễn phí có giới hạn
2. Đợi 1 giờ và th��� lại
3. Hoặc upgrade lên tài khoản trả phí

---

## 💡 Tips và Best Practices

### 🎯 Sử dụng hiệu quả
- **Bookmark URL**: Lưu public URL để dùng lại
- **Share safely**: Chỉ chia sẻ URL với người tin tưởng
- **Monitor usage**: Theo dõi bandwidth usage
- **Keep running**: Giữ cell chạy để duy trì tunnel

### 🔒 Bảo mật
- **Không commit authtoken**: Không push .env lên git
- **Regenerate token**: Tạo lại token nếu bị lộ
- **Use HTTPS**: Ngrok tự động dùng HTTPS
- **Session timeout**: Tunnel tự động đóng khi session end

### 🚀 Performance
- **Close unused tunnels**: Đóng tunnel không dùng
- **Use nearest region**: Chọn region gần nhất
- **Monitor latency**: Kiểm tra độ trễ
- **Optimize model**: Dùng model nhỏ hơn nếu cần

---

## 📊 So sánh các phương pháp

| Phương pháp | Setup | Ổn định | Tốc độ | Chia sẻ | Khuyến nghị |
|-------------|-------|---------|--------|---------|-------------|
| Ngrok + auth | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ✅ | ✅ Tốt nhất |
| Colab built-in | ⭐ | ⭐⭐ | ⭐⭐⭐ | ❌ | ⚠️ Local only |
| Localtunnel | ⭐⭐ | ⭐ | ⭐⭐ | ✅ | ⚠️ Không ổn định |

---

## 🎁 Ngrok Free Plan

### ✅ Được phép
- **1 online tunnel** cùng lúc
- **40 connections/minute**
- **HTTPS support**
- **Custom subdomain** (random)
- **Basic analytics**

### ❌ Giới hạn
- Chỉ 1 tunnel đồng thời
- Bandwidth limit
- Session timeout
- Không custom domain

---

## 🔗 Links hữu ích

- **Ngrok Dashboard**: https://dashboard.ngrok.com/
- **Ngrok Docs**: https://ngrok.com/docs
- **Pricing**: https://ngrok.com/pricing
- **Status Page**: https://status.ngrok.com/

---

## 📋 Checklist

### ✅ Trước khi bắt đầu
- [ ] Đã đăng ký tài khoản ngrok
- [ ] Đã lấy authtoken
- [ ] Đã cập nhật .env file
- [ ] Đã upload .env lên Colab

### ✅ Khi chạy
- [ ] Dependencies installed thành công
- [ ] Ngrok setup không lỗi
- [ ] Streamlit server started
- [ ] Public URL được tạo
- [ ] Có thể truy cập từ browser

### ✅ Khi sử dụng
- [ ] Upload PDF thành công
- [ ] Models load không lỗi
- [ ] Chat hoạt động bình thường
- [ ] Response time chấp nhận được

---

## 🎉 Kết luận

Với ngrok authtoken, bạn có thể:

🌟 **Truy cập từ mọi nơi** - Public URL hoạt động ở bất kỳ đâu  
🌟 **Chia sẻ dễ dàng** - Gửi URL cho người khác sử dụng  
🌟 **HTTPS secure** - Kết nối được mã hóa  
🌟 **Ổn đ���nh** - Tunnel ổn định hơn các phương pháp khác  

**Chúc bạn thành công với RAG QA PDF + Ngrok!** 🚀