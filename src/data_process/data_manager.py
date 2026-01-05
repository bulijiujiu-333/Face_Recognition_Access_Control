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
        """添加三人的人脸数据（手动放图时无需调用此函数）"""
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
        """获取单人的图片路径（支持三人，适配手动命名的图片）"""
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

    def check_data_completeness(self, expected_count: int = 20) -> None:
        """验证手动放入的数据集完整性（每人register/verify各20张）"""
        print("=== 手动数据集完整性检查 ===")
        for data_type in ["register", "verify"]:
            print(f"\n【{data_type} 目录】")
            for member in self.members:
                img_paths = self.get_member_data(member, data_type)
                actual_count = len(img_paths)
                status = "✅" if actual_count == expected_count else "❌"
                print(f"{member}：实际{actual_count}张 | 预期{expected_count}张 {status}")
                if actual_count != expected_count:
                    print(
                        f"  缺失/多余：需检查 {self.register_dir if data_type == 'register' else self.verify_dir}/{member} 目录")


# 测试三人数据读取 + 验证手动数据集完整性
if __name__ == "__main__":
    manager = FaceDataManager()
    # 1. 验证你手动放入的数据集是否各20张
    manager.check_data_completeness(expected_count=20)

    # 2. 读取并打印任意一人的图片路径（示例：褚文洁的verify图片）
    print("\n=== 褚文洁的verify图片路径示例 ===")
    chu_verify_imgs = manager.get_member_data("ChuWenjie", "verify")
    # 打印前3张路径（避免输出太长）
    for i, path in enumerate(chu_verify_imgs[:3]):
        print(f"第{i + 1}张：{path}")