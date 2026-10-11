"""
Notification Module (Module Thông Báo)
Cung cấp cơ chế thông báo đa kênh khi có sự cố té ngã.
Hỗ trợ ghi log sự cố và đường dẫn video vào storage/incidents/incidents.json.
"""
import os
import json
import time
from abc import ABC, abstractmethod


class BaseNotifier(ABC):
    """Lớp cơ sở cho tất cả các kênh thông báo."""

    @abstractmethod
    def notify(self, incident_data: dict) -> bool:
        """Gửi thông báo sự cố."""
        pass


class LocalLogNotifier(BaseNotifier):
    """Module thông báo ghi log cục bộ (Lưu incidents.json)."""

    def __init__(self, log_dir: str = "storage/incidents"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.json_file = os.path.join(self.log_dir, "incidents.json")

    def notify(self, incident_data: dict) -> bool:
        try:
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(incident_data.get("timestamp", time.time())))
            video_path = incident_data.get("video_path", "")
            snapshot_path = incident_data.get("snapshot_path", "")

            print(f"\n[THÔNG BÁO HỆ THỐNG] >>> PHÁT HIỆN TÉ NGÃ lúc {timestamp_str}!")
            if snapshot_path:
                print(f"                   >>> Ảnh chụp bằng chứng: {snapshot_path}")
            if video_path:
                print(f"                   >>> Video clip bằng chứng: {video_path}")
            print(f"                   >>> Thời gian duy trì ngã: {incident_data.get('duration', 0):.2f}s")
            print(f"                   >>> Độ tin cậy: {incident_data.get('confidence', 0)*100:.1f}%\n")

            history = []
            if os.path.exists(self.json_file):
                try:
                    with open(self.json_file, "r", encoding="utf-8") as f:
                        history = json.load(f)
                except Exception:
                    history = []

            history.append({
                "timestamp": incident_data.get("timestamp", time.time()),
                "timestamp_str": timestamp_str,
                "snapshot_path": snapshot_path,
                "video_path": video_path,
                "duration": round(incident_data.get("duration", 0), 2),
                "confidence": round(incident_data.get("confidence", 0), 2),
                "status": "recording_video" if not video_path else "completed"
            })

            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            print(f"[!] Lỗi ghi log sự cố: {e}")
            return False

    def update_incident_video(self, timestamp: float, video_path: str):
        """Cập nhật đường dẫn file video vào incidents.json sau khi xuất video xong."""
        try:
            if not os.path.exists(self.json_file):
                return
            with open(self.json_file, "r", encoding="utf-8") as f:
                history = json.load(f)

            # Cập nhật incident khớp thời gian gần nhất
            for inc in reversed(history):
                if abs(inc.get("timestamp", 0) - timestamp) < 3.0 or not inc.get("video_path"):
                    inc["video_path"] = video_path
                    inc["status"] = "completed"
                    break

            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            print(f"[+] LocalLogNotifier: Đã cập nhật video_path vào {self.json_file}")
        except Exception as e:
            print(f"[!] Lỗi cập nhật video_path vào log: {e}")


class TelegramNotifier(BaseNotifier):
    """
    Module thông báo qua Telegram Bot (Cấu hình sau).
    Để sử dụng: Truyền bot_token và chat_id.
    """

    def __init__(self, bot_token: str = "", chat_id: str = ""):
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.is_enabled = bool(self.bot_token and self.chat_id)

    def notify(self, incident_data: dict) -> bool:
        if not self.is_enabled:
            return False

        try:
            import requests
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            caption = (
                f"🚨 *CẢNH BÁO TÉ NGÃ!* 🚨\n\n"
                f"⏰ Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"⏱️ Duy trì ngã: {incident_data.get('duration', 0):.1f}s\n"
                f"🎯 Độ tin cậy: {incident_data.get('confidence', 0)*100:.1f}%\n"
                f"⚠️ Vui lòng kiểm tra tình trạng ngay lập tức!"
            )

            photo_path = incident_data.get("snapshot_path", "")
            if photo_path and os.path.exists(photo_path):
                with open(photo_path, "rb") as photo_file:
                    resp = requests.post(
                        url,
                        data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "Markdown"},
                        files={"photo": photo_file},
                        timeout=5
                    )
                    return resp.status_code == 200
            else:
                msg_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                resp = requests.post(
                    msg_url,
                    json={"chat_id": self.chat_id, "text": caption, "parse_mode": "Markdown"},
                    timeout=5
                )
                return resp.status_code == 200
        except Exception as e:
            print(f"[!] Lỗi gửi Telegram: {e}")
            return False


class NotificationManager:
    """Điều phối tất cả các kênh thông báo."""

    def __init__(self):
        self.notifiers = []
        # Mặc định luôn kích hoạt LocalLogNotifier
        self.add_notifier(LocalLogNotifier())

    def add_notifier(self, notifier: BaseNotifier):
        self.notifiers.append(notifier)

    def notify_all(self, incident_data: dict):
        results = []
        for notifier in self.notifiers:
            res = notifier.notify(incident_data)
            results.append(res)
        return any(results)

    def update_incident_video(self, timestamp: float, video_path: str):
        for notifier in self.notifiers:
            if hasattr(notifier, "update_incident_video"):
                notifier.update_incident_video(timestamp, video_path)
