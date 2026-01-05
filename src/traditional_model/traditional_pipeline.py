import cv2
import time
from typing import Dict  # 导入类型注解
from src.data_process import CameraCapture, FacePreprocessor
from .traditional_recognizer import TraditionalFaceRecognizer
from .blink_detector import BlinkDetector


class TraditionalPipeline:
    def __init__(self, model_type: str = "lbph"):
        self.camera = CameraCapture()
        self.recognizer = TraditionalFaceRecognizer(model_type=model_type)
        self.blink_detector = BlinkDetector()
        self.preprocessor = FacePreprocessor()

        # 状态变量
        self.is_running = False
        self.recognition_result: Dict = {
            "match_name": None,
            "similarity": 0.0,
            "blink_status": "未检测",
            "is_valid": False
        }

    def train_model(self) -> float:
        """训练传统识别模型"""
        return self.recognizer.train()

    def run(self):
        """启动识别+眨眼检测流程"""
        self.is_running = True
        self.camera.start_capture()
        print("传统模型流程启动，按ESC键退出...")

        while self.is_running:
            frame = self.camera.get_current_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # 1. 眨眼检测
            blink_detected, annotated_frame = self.blink_detector.detect_blink(frame)
            self.recognition_result["blink_status"] = "有效" if blink_detected else "无效"

            # 2. 人脸识别（仅在检测到人脸时执行）
            # 兼容组员A的face_alignment方法（若方法名为face_align，可在此修改）
            face_img = self.preprocessor.face_alignment(frame)
            if face_img is not None:
                match_name, similarity = self.recognizer.predict(face_img)
                self.recognition_result["match_name"] = match_name
                self.recognition_result["similarity"] = round(similarity, 2)
                # 判定是否有效（匹配成功+眨眼有效）
                self.recognition_result["is_valid"] = (match_name is not None) and blink_detected

            # 3. 在画面上标注识别结果
            if self.recognition_result["match_name"]:
                cv2.putText(annotated_frame, f"Match: {self.recognition_result['match_name']}", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Similarity: {self.recognition_result['similarity']}", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "No Match", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(annotated_frame, f"Blink: {self.recognition_result['blink_status']}", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 显示最终判定
            if self.recognition_result["is_valid"]:
                cv2.putText(annotated_frame, "ACCESS GRANTED", (400, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                cv2.putText(annotated_frame, "ACCESS DENIED", (400, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            cv2.imshow("Traditional Model + Blink Detection", annotated_frame)

            # 退出逻辑
            if cv2.waitKey(1) & 0xFF == 27:
                self.stop()
                break

    def stop(self):
        """停止流程"""
        self.is_running = False
        self.camera.stop_capture()
        cv2.destroyAllWindows()
        print("传统模型流程已停止")

    def get_result(self) -> Dict:
        """获取当前结果（供组员C集成）"""
        return self.recognition_result.copy()


# 测试代码
if __name__ == "__main__":
    pipeline = TraditionalPipeline(model_type="lbph")
    # 训练模型
    pipeline.train_model()
    # 启动流程
    pipeline.run()