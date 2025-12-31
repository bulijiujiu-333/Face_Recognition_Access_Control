import cv2
import numpy as np
import threading
import time
import os
from typing import Optional, Callable


class CameraCapture:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.callback: Optional[Callable[[np.ndarray], None]] = None

    def _capture_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            with self.frame_lock:
                self.current_frame = frame.copy()
            if self.callback:
                self.callback(frame)
            time.sleep(1/30)

    def start_capture(self, callback: Optional[Callable[[np.ndarray], None]] = None):
        if self.is_running:
            print("摄像头已运行")
            return
        self.is_running = True
        self.callback = callback
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("✅ 摄像头已启动")

    def stop_capture(self):
        self.is_running = False
        if hasattr(self, "capture_thread"):
            self.capture_thread.join(timeout=2)
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ 摄像头已停止")

    def get_current_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def capture_member_frames(self, member_name: str, data_type: str, count: int = 20):
        """采集指定成员的照片（支持三人）"""
        save_dir = f"D:/PycharmProjects/Face_Recognition_Access_Control/data/{data_type}/{member_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.start_capture()
        collected = 0
        print(f"开始采集{member_name}的{count}张{data_type}照片，按ESC中断")
        while collected < count and self.is_running:
            frame = self.get_current_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            # 显示进度
            cv2.putText(frame, f"采集：{collected+1}/{count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("采集窗口", frame)

            # 保存照片
            filename = f"{member_name}_{data_type}_{collected+1:02d}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), frame)
            collected += 1

            if cv2.waitKey(500) & 0xFF == 27:
                break
        self.stop_capture()
        print(f"采集完成：{collected}张照片已保存到{save_dir}")


# 测试采集三人照片（示例：周吉）
if __name__ == "__main__":
    camera = CameraCapture()
    # 替换为需要采集的成员（JingHanying/ZhouJi/ChuWenjie）
    camera.capture_member_frames(member_name="ZhouJi", data_type="register", count=20)