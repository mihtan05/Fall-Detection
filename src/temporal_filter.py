"""
Temporal Fall Filter (Bộ Lọc Báo Động Giả Theo Thời Gian & Hình Học)
Loai bo cac truong hop cui nhat do, ngoi xuong san tam thoi.
Chi xac nhan te nga khi nguoi do duy tri tu the nam san >= threshold_seconds.
"""
import time
from dataclasses import dataclass


@dataclass
class FilterStatus:
    is_confirmed_fall: bool   # Da vuot nguong thoi gian -> Xac nhan nga
    is_new_incident: bool     # Day la frame dau tien xac nhan nga (kich hoat alert 1 lan)
    duration: float           # Thoi gian nam san hien tai (giay)
    progress: float           # Tien trinh tu 0.0 -> 1.0 den khi xac nhan


class TemporalFallFilter:
    def __init__(self, fall_duration_threshold: float = 2.0, aspect_ratio_min: float = 0.85, cooldown_seconds: float = 0.5):
        """
        :param fall_duration_threshold: So giay nga lien tuc de xac nhan (mac dinh 2.0s).
        :param aspect_ratio_min: Ty le Chieu Rong / Chieu Cao toi thieu cua Bounding Box.
        :param cooldown_seconds: Thoi gian khong thay nga de reset bo dem (chong nhap nhay flicker).
        """
        self.fall_duration_threshold = fall_duration_threshold
        self.aspect_ratio_min = aspect_ratio_min
        self.cooldown_seconds = cooldown_seconds

        self.fall_start_time = None
        self.last_fall_time = None
        self.already_triggered = False

    def update(self, detected_fall: bool, aspect_ratio: float = 1.0) -> FilterStatus:
        now = time.time()

        # Dieu kien hop le: Model bao 'fall' VÀ hop thoa man dang nam (rong hon cao hoac gan bang)
        is_valid_fall_posture = detected_fall and (aspect_ratio >= self.aspect_ratio_min)

        if is_valid_fall_posture:
            self.last_fall_time = now

            if self.fall_start_time is None:
                self.fall_start_time = now

            duration = now - self.fall_start_time
            progress = min(1.0, duration / self.fall_duration_threshold)
            is_confirmed = duration >= self.fall_duration_threshold

            is_new = False
            if is_confirmed and not self.already_triggered:
                is_new = True
                self.already_triggered = True

            return FilterStatus(
                is_confirmed_fall=is_confirmed,
                is_new_incident=is_new,
                duration=duration,
                progress=progress
            )
        else:
            # Khong thay nga
            if self.fall_start_time is not None:
                # Kiem tra xem da qua cooldown ke tu lan cuoi phat hien nga chua
                time_since_last_fall = now - (self.last_fall_time if self.last_fall_time else now)

                if time_since_last_fall >= self.cooldown_seconds:
                    # Da qua cooldown -> Nguoi do da dung day -> Reset he thong
                    self.reset()
                    return FilterStatus(is_confirmed_fall=False, is_new_incident=False, duration=0.0, progress=0.0)
                else:
                    # Vua mat dau vet tam thoi (flicker) trong khoang cooldown -> giu trang thai
                    duration = now - self.fall_start_time
                    progress = min(1.0, duration / self.fall_duration_threshold)
                    is_confirmed = duration >= self.fall_duration_threshold
                    return FilterStatus(
                        is_confirmed_fall=is_confirmed,
                        is_new_incident=False,
                        duration=duration,
                        progress=progress
                    )

            return FilterStatus(is_confirmed_fall=False, is_new_incident=False, duration=0.0, progress=0.0)

    def reset(self):
        self.fall_start_time = None
        self.last_fall_time = None
        self.already_triggered = False
