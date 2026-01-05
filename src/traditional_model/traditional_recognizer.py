import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional  # 导入类型注解
from src.data_process import FaceDataManager, FacePreprocessor


class TraditionalFaceRecognizer:
    def __init__(self, model_type: str = "lbph"):
        """
        初始化传统识别模型
        :param model_type: 模型类型（lbph/pca）
        """
        self.model_type = model_type.lower()
        self.preprocessor = FacePreprocessor()
        self.data_manager = FaceDataManager()
        self.model = self._init_model()
        self.classes = []  # 存储组员姓名（标签）
        self.train_features = None  # 训练特征集
        self.scaler = StandardScaler()  # PCA用标准化器
        self.y_train = np.array([])  # 保存训练标签，供PCA预测使用

    def _init_model(self):
        """初始化模型"""
        if self.model_type == "lbph":
            # LBPH模型（OpenCV内置）
            return cv2.face.LBPHFaceRecognizer_create(
                radius=1,  # 邻域半径
                neighbors=8,  # 邻域像素数
                grid_x=8,  # 水平网格数
                grid_y=8  # 垂直网格数
            )
        elif self.model_type == "pca":
            # PCA+余弦相似度
            return PCA(n_components=50)  # 保留50个主成分
        else:
            raise ValueError("模型类型仅支持lbph和pca")

    def _load_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载训练数据（注册照）并预处理"""
        all_register_data = self.data_manager.get_all_data("register")
        X = []  # 特征矩阵
        y = []  # 标签（组员索引）

        for idx, (member, img_paths) in enumerate(all_register_data.items()):
            self.classes.append(member)
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                # 预处理
                processed_img = self.preprocessor.preprocess(img)
                if processed_img is not None:
                    X.append(processed_img.flatten())  # 展平为一维向量
                    y.append(idx)

        self.y_train = np.array(y)  # 保存训练标签为实例变量
        return np.array(X), self.y_train

    def train(self) -> float:
        """训练模型并返回训练准确率"""
        X_train, y_train = self._load_train_data()
        if len(X_train) == 0:
            raise ValueError("训练数据为空，请先采集注册照")

        print(f"开始训练{self.model_type}模型，训练样本数：{len(X_train)}")

        if self.model_type == "lbph":
            # LBPH要求输入为uint8类型的2D图像（非展平）
            X_train_lbph = [cv2.normalize(x.reshape(128, 128), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            for x in X_train]
            self.model.train(X_train_lbph, y_train)
            # 计算训练准确率
            correct = 0
            for img, label in zip(X_train_lbph, y_train):
                pred_label, _ = self.model.predict(img)
                if pred_label == label:
                    correct += 1
            train_acc = correct / len(X_train_lbph)

        else:  # PCA
            # 标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            # 降维
            self.train_features = self.model.fit_transform(X_train_scaled)
            # 计算训练准确率（用余弦相似度）
            correct = 0
            for i in range(len(self.train_features)):
                similarity = cosine_similarity([self.train_features[i]], self.train_features)[0]
                pred_idx = np.argmax(similarity)
                if y_train[pred_idx] == y_train[i]:
                    correct += 1
            train_acc = correct / len(self.train_features)

        print(f"{self.model_type}模型训练完成，训练准确率：{train_acc:.2f}")
        # 保存模型（创建目录，避免不存在报错）
        if self.model_type == "lbph":
            model_dir = "../models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            self.model.save(os.path.join(model_dir, "lbph_model.yml"))
        return train_acc

    def predict(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        预测人脸身份
        :param img: 输入图像（BGR格式）
        :return: (匹配姓名, 相似度得分)，未匹配返回(None, 0.0)
        """
        # 预处理
        processed_img = self.preprocessor.preprocess(img)
        if processed_img is None:
            return None, 0.0

        if self.model_type == "lbph":
            img_lbph = cv2.normalize(processed_img.reshape(128, 128), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            pred_label, confidence = self.model.predict(img_lbph)
            # LBPH的confidence是距离（越小越相似），转换为相似度（0-1）
            similarity = 1 - min(confidence / 100, 1.0)
            if similarity < 0.6:  # 相似度阈值（可调整）
                return None, similarity
            return self.classes[pred_label], similarity

        else:  # PCA
            if self.train_features is None:
                raise ValueError("模型未训练，请先调用train()")
            # 预处理+标准化+降维
            img_flatten = processed_img.flatten().reshape(1, -1)
            img_scaled = self.scaler.transform(img_flatten)
            img_feature = self.model.transform(img_scaled)
            # 计算余弦相似度
            similarities = cosine_similarity(img_feature, self.train_features)[0]
            max_sim_idx = np.argmax(similarities)
            max_sim = similarities[max_sim_idx]
            if max_sim < 0.7:  # 相似度阈值
                return None, max_sim
            # 调用实例变量self.y_train获取标签
            pred_label = self.y_train[max_sim_idx]
            return self.classes[pred_label], max_sim


# 测试代码
if __name__ == "__main__":
    # 初始化模型（选择lbph或pca）
    recognizer = TraditionalFaceRecognizer(model_type="lbph")
    # 训练
    recognizer.train()
    # 测试验证照
    test_img_path = "../data/verify/member1/member1_verify_01.jpg"
    test_img = cv2.imread(test_img_path)
    if test_img is not None:
        name, sim = recognizer.predict(test_img)
        print(f"测试结果：姓名={name}，相似度={sim:.2f}")