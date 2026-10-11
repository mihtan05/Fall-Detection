"""
Main Application: Chạy Camera Thật Giám Sát Té Ngã
Khởi chạy hệ thống AI Fall Detection thời gian thực với Webcam/Camera máy tính.
Tích hợp ghi lại video 5-10 giây trước & sau khi ngã và lưu log JSON sự cố.
"""
import os
import sys
import time
import threading
import winsound
import cv2

# Thêm thư mục hiện tại vào sys.path để import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detector import FallDetector
from src.temporal_filter import TemporalFallFilter
from src.notifier import NotificationManager, TelegramNotifier
from src.video_recorder import IncidentVideoRecorder


def play_beep_async(frequency: int = 2500, duration: int = 200):
    """Phát âm thanh cảnh báo bất đồng bộ (không làm khựng/lag video)."""
    def _beep():
        try:
            winsound.Beep(frequency, duration)
        except Exception:
            pass

    threading.Thread(target=_beep, daemon=True).start()


def open_camera(camera_index: int = 0):
    """Mở camera với DirectShow trên Windows để khởi động nhanh hơn."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    return cap


def main():
    print("=" * 65)
    print("  HE THONG AI GIAM SAT & PHAT HIEN TE NGA THOI GIAN THUC (YOLOv11-POSE)")
    print("=" * 65)

    # 1. Khởi tạo các thành phần cốt lõi
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Model", "weights", "yolo11n-pose.pt")

    detector = FallDetector(model_path=MODEL_PATH, conf_threshold=0.45)
    temporal_filter = TemporalFallFilter(fall_duration_threshold=1.5, aspect_ratio_min=0.0)
    notifier = NotificationManager()

    # Bộ ghi video sự cố: 4 giây trước khi ngã + 4 giây sau khi ngã
    video_recorder = IncidentVideoRecorder(pre_seconds=4.0, post_seconds=4.0, fps_estimate=25.0)

    # Nếu muốn dùng Telegram Bot sau này:
    # notifier.add_notifier(TelegramNotifier(bot_token="YOUR_TOKEN", chat_id="YOUR_CHAT_ID"))

    # Thư mục lưu video & ảnh bằng chứng
    snapshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage", "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    # 2. Kết nối vào Webcam
    print("[*] Đang kết nối vào Webcam...")
    cap = open_camera(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[!] KHÔNG THỂ MỞ CAMERA (Device 0). Đang thử thiết bị Camera 1...")
        cap = open_camera(1)
        if not cap.isOpened():
            print("[!] LỖI: Không tìm thấy camera nào. Vui lòng kiểm tra kết nối thiết bị!")
            return

    print("[+] KHỞI ĐỘNG THÀNH CÔNG!")
    print("    - Nhấn phím 'q' trên cửa sổ video để dừng chương trình.")
    print("    - Đứng / Ngồi bình thường : Khung XANH & Skeleton (Góc thân > 45°).")
    print("    - Ngã nằm sàn >= 1.5s     : Khung ĐỎ & CÒI HÚ BÁO ĐỘNG (Góc thân <= 38°).")
    print("    - Tự động lưu video 4s trước và 4s sau khi ngã vào 'storage/snapshots/'.\n")

    window_name = "AI Fall Detection System - YOLOv11 Pose Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    last_beep_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Không nhận được tín hiệu từ camera. Đang thử lại...")
                time.sleep(0.1)
                continue

            # Tính FPS thực tế
            curr_time = time.time()
            delta_time = curr_time - prev_time
            fps = int(1.0 / delta_time) if delta_time > 0.001 else 30
            prev_time = curr_time

            # 3. Nhận diện qua mô hình YOLOv11-Pose
            det_result = detector.detect(frame)

            # 4. Lọc qua bộ đếm thời gian (chống báo động giả)
            filter_status = temporal_filter.update(
                detected_fall=det_result["has_fall"],
                aspect_ratio=det_result["max_aspect_ratio"]
            )

            # 5. Xử lý sự cố té ngã
            if filter_status.is_new_incident:
                timestamp_int = int(time.time())
                snapshot_filename = f"fall_{timestamp_int}.jpg"
                snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
                video_filename = f"fall_{timestamp_int}.mp4"
                video_path = os.path.join(snapshot_dir, video_filename)

                # Lưu ảnh chụp bằng chứng (thumbnail)
                cv2.imwrite(snapshot_path, frame)

                # Còi hú báo động không chặn luồng video
                play_beep_async(frequency=2500, duration=350)
                last_beep_time = curr_time

                # Kích hoạt ghi lại video sự cố (4s trước + 4s sau khi ngã)
                incident_time = curr_time
                def _on_video_completed(v_path):
                    notifier.update_incident_video(incident_time, v_path)

                video_recorder.trigger(
                    output_video_path=video_path,
                    post_seconds=4.0,
                    on_complete=_on_video_completed
                )

                # Gửi thông báo đến các kênh đăng ký và ghi log JSON
                incident_data = {
                    "timestamp": curr_time,
                    "snapshot_path": snapshot_path,
                    "video_path": video_path,
                    "duration": filter_status.duration,
                    "confidence": det_result["max_fall_conf"]
                }
                notifier.notify_all(incident_data)

            elif filter_status.is_confirmed_fall:
                # Phát tiếng bíp lặp lại mỗi 1.2 giây khi người đó vẫn đang nằm ngã
                if (curr_time - last_beep_time) >= 1.2:
                    play_beep_async(frequency=2200, duration=150)
                    last_beep_time = curr_time

            # 6. Render giao diện HUD
            display_frame = detector.annotate_frame(frame, det_result, filter_status, fps=fps)

            # Nếu đang ghi video sự cố, hiển thị chấm đỏ nhấp nháy REC
            if len(video_recorder.active_sessions) > 0:
                cv2.circle(display_frame, (display_frame.shape[1] - 30, 60), 7, (0, 0, 255), -1)
                cv2.putText(display_frame, "REC VIDEO", (display_frame.shape[1] - 125, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # 7. Nạp frame vào bộ đệm video sự cố (lưu video chất lượng kèm khung xương HUD)
            video_recorder.add_frame(display_frame, timestamp=curr_time)

            cv2.imshow(window_name, display_frame)

            # Phím 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[*] Người dùng yêu cầu dừng hệ thống.")
                break

    except KeyboardInterrupt:
        print("\n[*] Đã nhận lệnh ngắt từ bàn phím (Ctrl+C).")
    finally:
        print("[*] Đang hoàn tất lưu các video sự cố còn dở dang (nếu có)...")
        video_recorder.flush_and_close()
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Đã giải phóng camera và đóng hệ thống an toàn.")


if __name__ == "__main__":
    main()
