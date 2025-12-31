import cv2
import dlib
import numpy as np
from typing import Optional, Tuple


class FacePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size
        # 关键点模型路径（替换为你的实际路径）
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(
            "D:/PycharmProjects/Face_Recognition_Access_Control/shape_predictor_68_face_landmarks.dat"
        )

    def _detect_face_landmarks(self, img: np.ndarray) -> Optional[dlib.full_object_detection]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        return self.landmark_predictor(gray, faces[0]) if faces else None

    def face_alignment(self, img: np.ndarray) -> Optional[np.ndarray]:
        landmarks = self._detect_face_landmarks(img)
        if not landmarks:
            return None

        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)
        left_eye = landmarks_np[36:42].mean(axis=0)
        right_eye = landmarks_np[42:48].mean(axis=0)

        # 计算旋转角度
        dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        # 计算缩放与中心
        eye_center = ((left_eye[0] + right_eye[0])/2, (left_eye[1] + right_eye[1])/2)
        scale = self.target_size[0] / (2 * np.linalg.norm(right_eye - left_eye) * 1.5)

        # 旋转+裁剪
        M = cv2.getRotationMatrix2D(eye_center, angle, scale)
        aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
        aligned_landmarks = self._detect_face_landmarks(aligned_img)
        if not aligned_landmarks:
            return None

        aligned_landmarks_np = np.array([[p.x, p.y] for p in aligned_landmarks.parts()])
        x_min, x_max = max(0, int(aligned_landmarks_np[:, 0].min() - 10)), min(aligned_img.shape[1], int(aligned_landmarks_np[:, 0].max() + 10))
        y_min, y_max = max(0, int(aligned_landmarks_np[:, 1].min() - 10)), min(aligned_img.shape[0], int(aligned_landmarks_np[:, 1].max() + 10))
        return cv2.resize(aligned_img[y_min:y_max, x_min:x_max], self.target_size)

    def preprocess(self, img: np.ndarray) -> Optional[np.ndarray]:
        """完整预处理：对齐→灰度→去噪→归一化（支持三人图片）"""
        aligned_img = self.face_alignment(img)
        if not aligned_img:
            print("警告：未检测到人脸")
            return None

        gray_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        denoised_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        return denoised_img / 255.0


# 测试三人图片预处理（示例：敬韩颖）
if __name__ == "__main__":
    preprocessor = FacePreprocessor()
    # 替换为任意一人的图片路径
    img_path = "D:/PycharmProjects/Face_Recognition_Access_Control/data/register/JingHanying/JingHanying_register_01.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print(f"错误：未找到图片{img_path}")
    else:
        processed_img = preprocessor.preprocess(img)
        if processed_img is not None:
            print(f"✅ 预处理成功！尺寸：{processed_img.shape}")
            cv2.imshow("Processed Face", processed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()