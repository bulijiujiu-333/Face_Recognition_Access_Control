import cv2
import dlib
import numpy as np
import os
from typing import Tuple, Optional  # 导入类型注解


class BlinkDetector:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        # 修复模型路径：使用当前文件目录拼接，避免路径错误
        current_dir = os.path.dirname(os.path.abspath(__file__))
        landmark_model_dir = os.path.join(current_dir, "model_data")
        # 创建model_data目录（若不存在）
        if not os.path.exists(landmark_model_dir):
            os.makedirs(landmark_model_dir)
        landmark_model_path = os.path.join(landmark_model_dir, "shape_predictor_68_face_landmarks.dat")
        self.landmark_predictor = dlib.shape_predictor(landmark_model_path)

        # 眨眼检测参数（可调整）
        self.EAR_THRESHOLD = 0.25  # 眼睛长宽比阈值（低于此值判定为闭眼）
        self.EAR_CONSEC_FRAMES = 2  # 连续闭眼帧数（超过此值判定为有效眨眼）
        self.frame_count = 0  # 连续闭眼帧计数器
        self.blink_detected = False  # 眨眼检测结果

    def _calculate_EAR(self, eye_landmarks: np.ndarray) -> float:
        """计算眼睛长宽比（EAR）"""
        # 眼睛关键点：[0,1,2,3,4,5] → 对应左眼36-41或右眼42-47
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (A + B) / (2.0 * C)

    def detect_blink(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        检测单帧中的眨眼动作
        :param frame: 摄像头帧（BGR格式）
        :return: (是否检测到有效眨眼, 标注后的帧)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        annotated_frame = frame.copy()  # 标注后的帧（用于展示）

        self.blink_detected = False
        if len(faces) == 0:
            return False, annotated_frame

        # 处理每个人脸（此处只处理第一个）
        face = faces[0]
        landmarks = self.landmark_predictor(gray, face)
        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

        # 提取左右眼关键点
        left_eye = landmarks_np[36:42]
        right_eye = landmarks_np[42:48]

        # 计算EAR
        left_EAR = self._calculate_EAR(left_eye)
        right_EAR = self._calculate_EAR(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # 绘制眼睛框
        cv2.polylines(annotated_frame, [left_eye.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(annotated_frame, [right_eye.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # 判断是否闭眼
        if avg_EAR < self.EAR_THRESHOLD:
            self.frame_count += 1
            cv2.putText(annotated_frame, "Eye Closed", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # 连续闭眼帧数达标，判定为有效眨眼
            if self.frame_count >= self.EAR_CONSEC_FRAMES:
                self.blink_detected = True
                cv2.putText(annotated_frame, "Blink Detected!", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.frame_count = 0

        # 显示EAR值
        cv2.putText(annotated_frame, f"EAR: {avg_EAR:.2f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        return self.blink_detected, annotated_frame


# 测试代码
if __name__ == "__main__":
    from src.data_process import CameraCapture  # 正确导入

    blink_detector = BlinkDetector()
    camera = CameraCapture()
    camera.start_capture()

    print("开始眨眼检测，按ESC键退出...")
    while True:
        frame = camera.get_current_frame()
        if frame is None:
            continue

        blink_detected, annotated_frame = blink_detector.detect_blink(frame)
        cv2.imshow("Blink Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.stop_capture()
    cv2.destroyAllWindows()