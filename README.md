# 🛡️ Real-Time Fall Detection System (YOLOv11-Pose)

Hệ thống phát hiện té ngã thời gian thực qua Camera/Webcam dựa trên mô hình **YOLOv11 Pose Estimation (17 Keypoints)**, phân tích góc nghiêng sinh trắc học sống lưng, khử báo động giả và tự động trích xuất video sự cố.

---

## ⚡ Tính Năng Cốt Lõi

* **Khung xương Pose Estimation (17 Keypoints)**: Đo góc sống lưng $\theta$ từ trung điểm vai đến hông.
  * $\theta > 45^\circ$: Bình thường (Đứng/Ngồi) $\rightarrow$ Khung xanh ngọc + Skeleton.
  * $\theta \le 38^\circ$: Nằm sàn $\rightarrow$ Kích hoạt cảnh báo ngã (Khung đỏ rực).
* **Khử báo động giả 2 lớp (`TemporalFallFilter`)**: Yêu cầu duy trì tư thế ngã $\ge 1.5$s liên tục (loại bỏ trường hợp cúi nhặt đồ hoặc ngồi bệt); có cooldown chống chập chờn tín hiệu.
* **Tự động lưu video sự cố (Pre/Post Buffer)**: Tự trích xuất video clip MP4 (4s trước + 4s sau khi ngã) vào `storage/snapshots/fall_<timestamp>.mp4` không gây trễ camera.
* **Hệ thống cảnh báo đa kênh**:
  * Còi hú âm thanh tức thì qua loa máy tính (`winsound.Beep`) chạy ngầm bất đồng bộ.
  * Tự động lưu nhật ký sự cố chi tiết vào `storage/incidents/incidents.json`.

---

## 🛠️ Cài Đặt (Installation)

### 1. Yêu cầu hệ thống
* **Python**: 3.9 trở lên (Khuyên dùng 3.10 hoặc 3.11).
* **Thiết bị**: Webcam/Camera tích hợp hoặc USB.
* **Phần cứng**: Hỗ trợ tốt cả **CPU** (tối ưu hóa qua YOLO11n nhẹ) và **GPU NVIDIA (CUDA)**.

### 2. Các bước cài đặt

```bash
# 1. Clone repository về máy
git clone https://github.com/mihtan05/Fall-Detection.git
cd Fall-Detection

# 2. Tạo và kích hoạt môi trường ảo (Khuyên dùng)
python -m venv .venv

# Kích hoạt trên Windows:
.\.venv\Scripts\activate
# Hoặc trên Linux/macOS:
source .venv/bin/activate

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Khởi Chạy Camera
* **Cách 1 (Nhanh nhất):** Nhấp đúp chuột vào file **`run.bat`**.
* **Cách 2 (Terminal):**
  ```powershell
  cd "D:\Fall Detect"
  & .\.venv\Scripts\python.exe run_webcam.py
  ```
