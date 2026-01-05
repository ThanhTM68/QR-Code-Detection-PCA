# QR Code Detection & Localization using PCA (YOLO Data Supported)

> **Hệ thống phát hiện và định vị mã QR (QR Code Localization) hiệu năng cao, sử dụng thuật toán Phân tích thành phần chính (PCA) và Xử lý ảnh số. Hỗ trợ huấn luyện trực tiếp từ bộ dữ liệu gán nhãn YOLO.**

---

## 1. Giới thiệu

Dự án này giải quyết bài toán phát hiện mã QR trong môi trường phức tạp mà không cần sử dụng mạng Neural (Deep Learning) nặng nề.

Thay vào đó, chúng tôi sử dụng phương pháp Statistical Pattern Recognition (Nhận dạng mẫu thống kê). Hệ thống tự động học đặc trưng của Finder Pattern (hoa văn định vị ở 3 góc QR) từ dữ liệu gán nhãn YOLO, tạo ra một không gian vector riêng (Eigenspace) để phân loại và định vị mã QR với tốc độ xử lý thời gian thực.

### Mục tiêu bài toán

1.  **Detection:** Xác định chính xác vị trí mã QR trong ảnh thiếu sáng, nghiêng, xoay, nhiễu nhẹ đến trung bình.
2.  **Robustness:** Khử nhiễu và loại bỏ các đối tượng có hình dáng giống QR (False Positives).
3.  **Optimization:** Tối ưu hóa thuật toán để đạt tốc độ xử lý nhanh với chi phí tính toán thấp.

---

## 2. Cấu trúc thư mục

```text
QR-Code-PCA-Project/
├── train/
│   └── finderPatterns/       # Thư mục huấn luyện
│       ├── images/           # Chứa ảnh gốc (.jpg, .png)
│       ├── labels/           # Chứa file nhãn tọa độ (.txt)
│       └── classes.txt       # Định nghĩa tên lớp (FinderPattern)
├── QRCode/                   # Dữ liệu kiểm thử
├── output_qr/               # Kết quả sau khi chạy mô hình
├── venv/                     # Môi trường ảo Python
├── .gitignore                # File cấu hình Git
├── QR_Detection_PCA.ipynb    # Source code chính (Jupyter Notebook)
└── README.md                 # Tài liệu dự án
```

---

## 3. Phương pháp & Quy trình (Pipeline)

Hệ thống hoạt động qua 2 giai đoạn: Huấn luyện (Training) và Suy luận (Inference)

### Giai đoạn 1: Huấn luyện

1. Data Loading: Quét thư mục images và labels
2. Auto-Cropping: Cắt các vùng Finder Pattern dựa trên tọa độ YOLO.
3. PCA Computation
    - Chuyển dữ liệu sang dạng vector phẳng
    - Tính toán Mean Vector và Eigenvectors (các thành phần chính)
    - Thiết lập ngưỡng sai số tái tạo (T_D​=Mean + 3σ) để làm mốc phân loại

### Giai đoạn 2: Suy luận

1. Tiền xử lý: Grayscale → Median Blur → Adaptive Threshold → Morphology
2. Trích xuất ứng viên: Tìm contours, lọc theo diện tích và cấu trúc lồng nhau (Hierarchy)
3. Kiểm tra PCA:
    - Chiếu ứng viên lên không gian PCA đã học.
    - Nếu Distance ≤ T_D ​→ Là Finder Pattern thật
4. Gom nhóm hình học: Tìm bộ 3 điểm tạo thành hình Tam giác vuông cân (đặc trưng L-shape của QR) và vẽ Bounding Box.

---

## 4. Cài đặt

Bước 1: Chuẩn bị môi trường

```text
# Tạo môi trường ảo (khuyên dùng môi trường ảo)
py -3.10 -m venv venv
.\venv\Scripts\activate  # trên Windows

# Cài đặt thư viện
pip install opencv-python numpy matplotlib pandas

```

---

## 5. Đánh giá (Evaluation)

Hệ thống được đánh giá trên tập test với các chỉ số:

-   Accuracy: Tỷ lệ phát hiện đúng QR Code.
-   Precision/Recall: Đánh giá khả năng lọc nhiễu của mô hình PCA.
-   Processing Time: Thời gian xử lý trung bình trên mỗi ảnh (FPS).
