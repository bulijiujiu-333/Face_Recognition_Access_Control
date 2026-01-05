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
        # 只返回第一个检测到的人脸关键点
        return self.landmark_predictor(gray, faces[0]) if len(faces) > 0 else None

    def face_alignment(self, img: np.ndarray) -> Optional[np.ndarray]:
        landmarks = self._detect_face_landmarks(img)
        if landmarks is None:
            return None

        landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)
        left_eye = landmarks_np[36:42].mean(axis=0)
        right_eye = landmarks_np[42:48].mean(axis=0)

        # 计算旋转角度
        dy, dx = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        # 计算缩放与中心
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        scale = self.target_size[0] / (2 * np.linalg.norm(right_eye - left_eye) * 1.5)

        # 旋转+裁剪
        M = cv2.getRotationMatrix2D(eye_center, angle, scale)
        aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
        aligned_landmarks = self._detect_face_landmarks(aligned_img)

        if aligned_landmarks is None:
            return None

        aligned_landmarks_np = np.array([[p.x, p.y] for p in aligned_landmarks.parts()])
        x_min, x_max = max(0, int(aligned_landmarks_np[:, 0].min() - 10)), min(aligned_img.shape[1], int(
            aligned_landmarks_np[:, 0].max() + 10))
        y_min, y_max = max(0, int(aligned_landmarks_np[:, 1].min() - 10)), min(aligned_img.shape[0], int(
            aligned_landmarks_np[:, 1].max() + 10))

        # 裁剪并缩放至目标尺寸
        cropped_img = aligned_img[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0:  # 检查裁剪后的图片是否为空
            return None
        return cv2.resize(cropped_img, self.target_size)

    def preprocess(self, img: np.ndarray) -> Optional[np.ndarray]:
        """完整预处理：对齐→灰度→去噪→归一化（支持三人图片）"""
        aligned_img = self.face_alignment(img)
        # 修复核心：先判断是否为None，再判断数组是否有效
        if aligned_img is None or aligned_img.size == 0:
            print("警告：人脸对齐失败，未检测到有效人脸")
            return None

        gray_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        denoised_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        # 归一化到0-1区间
        return denoised_img / 255.0


# 测试三人图片预处理（动态读取手动放入的图片）
if __name__ == "__main__":
    from data_manager import FaceDataManager  # 引入数据管理类

    preprocessor = FacePreprocessor()
    manager = FaceDataManager()

    # 随机选一张周吉的register图片（你手动放入的）
    zhou_imgs = manager.get_member_data("ZhouJi", "register")
    if not zhou_imgs:
        print("错误：周吉的register目录无图片")
    else:
        img_path = zhou_imgs[0]  # 取第一张测试
        img = cv2.imread(img_path)
        if img is None:
            print(f"错误：未找到图片{img_path}，请检查路径是否正确")
        else:
            processed_img = preprocessor.preprocess(img)
            if processed_img is not None:
                print(f"✅ 预处理成功！尺寸：{processed_img.shape}")
                cv2.imshow("Processed Face (手动数据集)", processed_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("❌ 预处理失败：未检测到人脸或图片无效")