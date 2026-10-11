"""
Fall Detector Module (Lõi Nhận Diện Té Ngã Qua Khung Xương Pose Estimation)
Tích hợp mô hình YOLOv11-Pose (yolo11n-pose.pt) theo dõi 17 điểm khớp xương người.
Xác định tư thế té ngã thông qua góc nghiêng sống lưng (Torso Angle) so với mặt đất.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO


# Danh sách các cặp nối xương người (COCO 17 Keypoints)
SKELETON_PAIRS = [
    # Đầu & mặt
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Thân (Vai & Hông)
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Cánh tay trái
    (5, 7), (7, 9),
    # Cánh tay phải
    (6, 8), (8, 10),
    # Chân trái
    (11, 13), (13, 15),
    # Chân phải
    (12, 14), (14, 16)
]


class FallDetector:
    def __init__(self, model_path: str = "Model/weights/yolo11n-pose.pt", conf_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold

        # Nếu model chưa có trong đường dẫn chỉ định, tải hoặc nạp tự động
        if not os.path.exists(self.model_path):
            print(f"[*] FallDetector: Không tìm thấy tại {self.model_path}, chuyển sang nạp 'yolo11n-pose.pt'...")
            self.model_path = "yolo11n-pose.pt"

        print(f"[*] FallDetector: Đang khởi tạo YOLOv11 Pose Estimation từ: {self.model_path}...")
        self.model = YOLO(self.model_path)
        print("[+] FallDetector: Đã sẵn sàng! Chế độ Pose Estimation (17 Keypoints).")

    @staticmethod
    def calculate_torso_angle(kpts: np.ndarray) -> tuple:
        """
        Tính góc nghiêng của sống lưng (từ trung điểm 2 hông lên trung điểm 2 vai) so với mặt sàn.
        Trả về (angle_deg, conf_score):
          - Đứng thẳng: ~75° - 90°
          - Ngồi thẳng: ~65° - 85°
          - Nằm sàn / Té ngã: <= 38°
        """
        # Keypoints: 5: vai trái, 6: vai phải, 11: hông trái, 12: hông phải
        ls_x, ls_y, ls_conf = kpts[5]
        rs_x, rs_y, rs_conf = kpts[6]
        lh_x, lh_y, lh_conf = kpts[11]
        rh_x, rh_y, rh_conf = kpts[12]

        # Xác định trung điểm 2 vai
        if ls_conf > 0.25 and rs_conf > 0.25:
            mid_shoulder = ((ls_x + rs_x) / 2.0, (ls_y + rs_y) / 2.0)
        elif ls_conf > 0.25:
            mid_shoulder = (ls_x, ls_y)
        elif rs_conf > 0.25:
            mid_shoulder = (rs_x, rs_y)
        else:
            mid_shoulder = None

        # Xác định trung điểm 2 hông
        if lh_conf > 0.25 and rh_conf > 0.25:
            mid_hip = ((lh_x + rh_x) / 2.0, (lh_y + rh_y) / 2.0)
        elif lh_conf > 0.25:
            mid_hip = (lh_x, lh_y)
        elif rh_conf > 0.25:
            mid_hip = (rh_x, rh_y)
        else:
            mid_hip = None

        # Khi nhìn thấy cả vai và hông (chính xác nhất)
        if mid_shoulder is not None and mid_hip is not None:
            dx = abs(mid_shoulder[0] - mid_hip[0])
            dy = abs(mid_shoulder[1] - mid_hip[1])
            angle_rad = np.arctan2(dy, dx + 1e-6)
            angle_deg = float(np.degrees(angle_rad))
            conf_score = min(max(ls_conf, rs_conf), max(lh_conf, rh_conf))
            return angle_deg, conf_score

        # Dự phòng khi bị che khuất hông: đo từ mũi (0) đến đầu gối (13, 14)
        nose_x, nose_y, nose_conf = kpts[0]
        lk_x, lk_y, lk_conf = kpts[13]
        rk_x, rk_y, rk_conf = kpts[14]
        if nose_conf > 0.3 and (lk_conf > 0.25 or rk_conf > 0.25):
            mid_k_x = (lk_x + rk_x) / 2.0 if (lk_conf > 0.25 and rk_conf > 0.25) else (lk_x if lk_conf > 0.25 else rk_x)
            mid_k_y = (lk_y + rk_y) / 2.0 if (lk_conf > 0.25 and rk_conf > 0.25) else (lk_y if lk_conf > 0.25 else rk_y)
            dx = abs(nose_x - mid_k_x)
            dy = abs(nose_y - mid_k_y)
            angle_rad = np.arctan2(dy, dx + 1e-6)
            return float(np.degrees(angle_rad)), min(nose_conf, max(lk_conf, rk_conf))

        # Nếu không đủ keypoint, trả về giá trị mặc định đứng thẳng
        return 90.0, 0.0

    def detect(self, frame: np.ndarray) -> dict:
        """
        Nhận diện người và 17 khớp xương trên frame.
        Đánh giá tư thế ngã dựa trên góc nghiêng sống lưng và tỷ lệ khung bao.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        has_fall = False
        max_fall_conf = 0.0
        max_aspect_ratio = 1.0

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is None or len(boxes) == 0:
                continue

            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy

                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                aspect_ratio = w / h

                # Lấy 17 điểm khớp xương của người này
                if keypoints is not None and len(keypoints) > i:
                    kpts_xy = keypoints.xy[i].cpu().numpy()  # (17, 2)
                    kpts_conf = keypoints.conf[i].cpu().numpy() if keypoints.conf is not None else np.ones(17)
                    # Gộp thành mảng (17, 3) [x, y, conf]
                    kpts = np.hstack([kpts_xy, kpts_conf.reshape(-1, 1)])
                else:
                    kpts = np.zeros((17, 3))

                # 1. Tính góc nghiêng thân người
                angle_deg, torso_conf = self.calculate_torso_angle(kpts)

                # 2. Điều kiện xác nhận tư thế ngã
                # - Góc sống lưng <= 38° (thân người nằm ngang so với sàn)
                # - Hoặc góc <= 45° kết hợp khung hình rộng hơn hoặc gần bằng cao (W/H >= 0.95)
                # - Hoặc nếu mất dấu keypoint nhưng khung bao người rất bẹp ngang (W/H >= 1.35)
                is_fall_by_angle = (angle_deg <= 38.0 and torso_conf > 0.2)
                is_fall_by_posture = (angle_deg <= 46.0 and aspect_ratio >= 0.95)
                is_fall_by_ratio = (aspect_ratio >= 1.35)

                is_fall = bool(is_fall_by_angle or is_fall_by_posture or is_fall_by_ratio)

                fall_conf = conf
                if is_fall:
                    has_fall = True
                    # Điểm tin cậy ngã càng cao khi góc thân càng sát mặt sàn
                    angle_confidence = max(0.5, (90.0 - angle_deg) / 90.0)
                    fall_conf = round(conf * 0.4 + angle_confidence * 0.6, 2)
                    if fall_conf > max_fall_conf:
                        max_fall_conf = fall_conf
                        max_aspect_ratio = max(aspect_ratio, 1.2)

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "fall_conf": fall_conf,
                    "angle": angle_deg,
                    "aspect_ratio": aspect_ratio,
                    "is_fall": is_fall,
                    "keypoints": kpts
                })

        return {
            "has_fall": has_fall,
            "max_fall_conf": max_fall_conf,
            "max_aspect_ratio": max_aspect_ratio,
            "detections": detections
        }

    def annotate_frame(self, frame: np.ndarray, detect_result: dict, filter_status, fps: int = 0) -> np.ndarray:
        """
        Vẽ khung xương người 17 keypoints và giao diện HUD chuyên nghiệp lên frame.
        """
        h_frame, w_frame = frame.shape[:2]
        output = frame.copy()

        # 1. Vẽ khung xương & Bounding Box cho từng người
        for det in detect_result.get("detections", []):
            x1, y1, x2, y2 = det["box"]
            is_fall = det["is_fall"]
            angle = det["angle"]
            kpts = det["keypoints"]

            # Chọn bảng màu: Đỏ rực nếu ngã, Xanh ngọc lục bảo nếu bình thường
            if is_fall:
                color_box = (0, 0, 255)       # Đỏ
                color_bone = (0, 80, 255)     # Cam đỏ
                color_joint = (0, 255, 255)   # Vàng
                text = f"FALL! ({angle:.0f} deg)"
            else:
                color_box = (0, 220, 100)     # Xanh lục bảo
                color_bone = (0, 240, 150)    # Xanh sáng
                color_joint = (255, 255, 0)   # Xanh lơ
                text = f"Normal ({angle:.0f} deg)"

            # Vẽ các đường nối xương (Skeleton bones)
            for p1, p2 in SKELETON_PAIRS:
                if p1 < len(kpts) and p2 < len(kpts):
                    x_a, y_a, c_a = kpts[p1]
                    x_b, y_b, c_b = kpts[p2]
                    if c_a > 0.25 and c_b > 0.25:
                        pt_a = (int(x_a), int(y_a))
                        pt_b = (int(x_b), int(y_b))
                        cv2.line(output, pt_a, pt_b, color_bone, 2, cv2.LINE_AA)

            # Vẽ các điểm khớp xương (Joint points)
            for pt in kpts:
                px, py, pc = pt
                if pc > 0.25:
                    cv2.circle(output, (int(px), int(py)), 4, color_joint, -1, cv2.LINE_AA)
                    cv2.circle(output, (int(px), int(py)), 5, (0, 0, 0), 1, cv2.LINE_AA)

            # Vẽ Bounding Box
            cv2.rectangle(output, (x1, y1), (x2, y2), color_box, 2)

            # Vẽ nhãn góc thân người có nền mờ
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(output, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(th + 8, y1)), color_box, -1)
            cv2.putText(output, text, (x1 + 3, max(th + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # 2. Hiển thị cảnh báo theo trạng thái bộ lọc thời gian
        if filter_status.is_confirmed_fall:
            # Viền đỏ cảnh báo toàn màn hình
            cv2.rectangle(output, (0, 0), (w_frame, h_frame), (0, 0, 255), 8)

            # Banner cảnh báo nổi bật phía trên
            banner_h = 60
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (w_frame, banner_h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.82, output, 0.18, 0, output)

            warn_text = f"CANH BAO: XAC NHAN TE NGA! (Duy tri: {filter_status.duration:.1f}s)"
            cv2.putText(output, warn_text, (30, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3, cv2.LINE_AA)

        elif detect_result.get("has_fall", False) and filter_status.duration > 0:
            # Thanh tiến trình đếm giây xác nhận (tránh báo động giả)
            bar_w = int(220 * filter_status.progress)
            cv2.rectangle(output, (20, 60), (240, 78), (40, 40, 40), -1)
            cv2.rectangle(output, (20, 60), (20 + bar_w, 78), (0, 165, 255), -1)
            cv2.putText(output, f"Kiem tra nga: {filter_status.duration:.1f}s / {filter_status.progress*100:.0f}%", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 2, cv2.LINE_AA)

        # 3. Thông tin HUD (FPS & Trạng thái hệ thống)
        status_text = "DANG THEO DOI" if not filter_status.is_confirmed_fall else "CO SU CO!"
        status_color = (0, 255, 0) if not filter_status.is_confirmed_fall else (0, 0, 255)

        cv2.putText(output, f"FPS: {fps}", (w_frame - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(output, f"POSE AI: {status_text}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        return output
