import os
import shutil
from typing import List, Dict


class FaceDataManager:
    def __init__(self):
        # 绝对路径（根据你的实际路径修改）
        self.root_dir = "D:/PycharmProjects/Face_Recognition_Access_Control/data"
        self.register_dir = os.path.join(self.root_dir, "register")
        self.verify_dir = os.path.join(self.root_dir, "verify")
        # 三人拼音姓名
        self.members = ["JingHanying", "ZhouJi", "ChuWenjie"]

    def add_face_data(self, member_name: str, data_type: str, img_path: str) -> bool:
        """添加三人的人脸数据"""
        if member_name not in self.members:
            print(f"错误：组员{member_name}不在名单中")
            return False
        if data_type not in ["register", "verify"]:
            print(f"错误：数据类型{data_type}无效")
            return False

        target_dir = os.path.join(self.register_dir if data_type == "register" else self.verify_dir, member_name)
        if not os.path.exists(target_dir):
            print(f"错误：目标目录{target_dir}不存在，请手动创建")
            return False

        # 自动按规范命名（支持三人）
        file_count = len([f for f in os.listdir(target_dir) if f.endswith((".jpg", ".png"))]) + 1
        new_filename = f"{member_name}_{data_type}_{file_count:02d}.jpg"
        target_path = os.path.join(target_dir, new_filename)

        try:
            shutil.copy(img_path, target_path)
            print(f"成功添加：{new_filename} -> {target_dir}")
            return True
        except Exception as e:
            print(f"添加失败：{str(e)}")
            return False

    def get_member_data(self, member_name: str, data_type: str) -> List[str]:
        """获取单人的图片路径（支持三人）"""
        target_dir = os.path.join(self.register_dir if data_type == "register" else self.verify_dir, member_name)
        if not os.path.exists(target_dir):
            return []
        return [
            os.path.join(target_dir, f)
            for f in os.listdir(target_dir)
            if f.endswith((".jpg", ".png"))
        ]

    def get_all_data(self, data_type: str) -> Dict[str, List[str]]:
        """获取三人的所有图片路径（用于模型训练）"""
        all_data = {}
        for member in self.members:
            all_data[member] = self.get_member_data(member, data_type)
        return all_data

    def delete_data(self, img_path: str) -> bool:
        """删除指定图片"""
        if not os.path.exists(img_path) or not img_path.startswith(self.root_dir):
            print(f"错误：文件{img_path}不存在或不在允许目录内")
            return False
        try:
            os.remove(img_path)
            print(f"成功删除：{img_path}")
            return True
        except Exception as e:
            print(f"删除失败：{str(e)}")
            return False


# 测试三人数据读取
if __name__ == "__main__":
    manager = FaceDataManager()
    # 测试敬韩颖、周吉、褚文洁的注册照
    print("=== 三人注册照列表 ===")
    for member in manager.members:
        data = manager.get_member_data(member, "register")
        print(f"{member}：{data if data else '暂无照片'}")