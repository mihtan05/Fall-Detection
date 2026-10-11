"""
Incident Video Recorder (Bộ Ghi Video Sự Cố Pre-Event & Post-Event)
Duy trì bộ đệm tròn (Rolling Buffer) lưu trữ 5-8 giây TRƯỚC khi ngã.
Khi có sự cố té ngã, tự động thu thập thêm 5-8 giây SAU khi ngã.
Ghép nối và xuất thành tệp video MP4 hoàn chỉnh chạy ngầm không làm lag camera.
"""
import os
import time
import threading
from collections import deque
import cv2
import numpy as np


class IncidentVideoRecorder:
    def __init__(self, pre_seconds: float = 4.0, post_seconds: float = 4.0, fps_estimate: float = 20.0):
        """
        :param pre_seconds: Số giây ghi trước khi ngã (mặc định 4.0 giây).
        :param post_seconds: Số giây ghi sau khi ngã (mặc định 4.0 giây).
        :param fps_estimate: Ước lượng FPS để tính dung lượng buffer.
        """
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.fps_estimate = fps_estimate

        # Bộ đệm tròn lưu tối đa số frame tương ứng pre_seconds + 2s dư phòng
        max_buffer_frames = int((pre_seconds + 2.0) * max(fps_estimate, 30.0))
        self.buffer = deque(maxlen=max_buffer_frames)

        self.active_sessions = []
        self.lock = threading.Lock()
        self.last_frame_time = time.time()
        self.measured_fps = fps_estimate

    def add_frame(self, frame: np.ndarray, timestamp: float = None):
        """
        Nạp từng frame vào bộ đệm và thu thập frame sau ngã cho các session đang hoạt động.
        """
        if frame is None:
            return

        now = timestamp if timestamp is not None else time.time()

        # Đo FPS thực tế trung bình động
        dt = now - self.last_frame_time
        if 0.005 < dt < 1.0:
            instant_fps = 1.0 / dt
            self.measured_fps = 0.9 * self.measured_fps + 0.1 * instant_fps
        self.last_frame_time = now

        frame_copy = frame.copy()

        with self.lock:
            # Lưu vào bộ đệm tròn
            self.buffer.append((now, frame_copy))

            # Xử lý các phiên ghi sự cố đang đợi gom đủ frame sau ngã
            completed_sessions = []
            for session in self.active_sessions:
                session["post_frames"].append((now, frame_copy))
                if now >= session["end_time"]:
                    completed_sessions.append(session)

            for session in completed_sessions:
                self.active_sessions.remove(session)
                # Bắn luồng ghi video ra đĩa bất đồng bộ (không chặn camera)
                threading.Thread(
                    target=self._write_video_worker,
                    args=(session, max(12.0, min(self.measured_fps, 35.0))),
                    daemon=True
                ).start()

    def trigger(self, output_video_path: str, post_seconds: float = None, on_complete=None) -> bool:
        """
        Kích hoạt ghi lại sự cố:
        Lấy toàn bộ frame trong pre_seconds trước đó + chờ thêm post_seconds tiếp theo.
        """
        duration = post_seconds if post_seconds is not None else self.post_seconds
        now = time.time()
        min_pre_time = now - self.pre_seconds

        with self.lock:
            # Lọc các frame trong khoảng pre_seconds vừa qua
            pre_frames = [f for (t, f) in self.buffer if t >= min_pre_time]

            # Đảm bảo có ít nhất 10 frame dự phòng nếu máy vừa khởi động
            if not pre_frames and len(self.buffer) > 0:
                pre_frames = [f for (t, f) in self.buffer]

            session = {
                "output_path": output_video_path,
                "pre_frames": pre_frames,
                "post_frames": [],
                "end_time": now + duration,
                "callback": on_complete
            }
            self.active_sessions.append(session)

        print(f"[*] IncidentVideoRecorder: Bắt đầu thu thập video sự cố (Pre: {len(pre_frames)} frames, chờ thêm {duration:.1f}s)...")
        return True

    def _write_video_worker(self, session: dict, fps: float):
        """Hàm ghi file video MP4 chạy ngầm trong background thread."""
        output_path = session["output_path"]
        pre_frames = session["pre_frames"]
        post_frames = [f for (t, f) in session["post_frames"]]
        all_frames = pre_frames + post_frames

        if not all_frames:
            print(f"[!] IncidentVideoRecorder: Không có frame nào để xuất video: {output_path}")
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            h, w = all_frames[0].shape[:2]

            # Dùng codec mp4v chuẩn cho file MP4 tương thích mọi thiết bị
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            if not writer.isOpened():
                # Dự phòng codec XVID dạng .avi nếu môi trường Windows lỗi mp4v
                alt_path = os.path.splitext(output_path)[0] + ".avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(alt_path, fourcc, fps, (w, h))
                if writer.isOpened():
                    output_path = alt_path

            for frame in all_frames:
                writer.write(frame)

            writer.release()
            total_duration = len(all_frames) / fps
            print(f"[+] IncidentVideoRecorder: Đã xuất video bằng chứng thành công: {output_path} ({len(all_frames)} frames, ~{total_duration:.1f}s, {fps:.1f} FPS)")

            if session.get("callback"):
                session["callback"](output_path)

        except Exception as e:
            print(f"[!] IncidentVideoRecorder: Lỗi khi lưu file video: {e}")

    def flush_and_close(self):
        """Lưu lại các phiên ghi còn dở dang trước khi tắt chương trình."""
        with self.lock:
            for session in list(self.active_sessions):
                self._write_video_worker(session, max(15.0, min(self.measured_fps, 30.0)))
            self.active_sessions.clear()
