"""
人脸门禁系统 - 主入口文件
当前阶段：项目框架初始化，完成环境校验和基础组件加载
后续扩展：人脸检测、识别、门禁控制等核心逻辑
"""
import os
import sys

# -------------------------- 基础配置（和项目结构对应） --------------------------
# 模型文件路径（和.gitignore中排除的文件对应）
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
# 依赖库列表（校验环境是否安装成功）
REQUIRED_LIBS = ["cv2", "dlib", "numpy"]


# -------------------------- 环境校验函数 --------------------------
def check_environment():
    """校验项目运行的基础环境（依赖+模型文件）"""
    print("🔍 开始校验人脸门禁系统运行环境...")

    # 1. 校验依赖库是否安装
    missing_libs = []
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
            print(f"✅ 依赖库 {lib} 已安装")
        except ImportError:
            missing_libs.append(lib)

    if missing_libs:
        print(f"❌ 缺少依赖库：{missing_libs}")
        print("💡 解决方案：执行 pip install opencv-python dlib-bin numpy")
        sys.exit(1)  # 依赖缺失则退出

    # 2. 校验模型文件路径（仅检查路径，不加载模型，避免占用资源）
    if os.path.exists(MODEL_PATH):
        print(f"✅ 模型文件 {MODEL_PATH} 存在")
    else:
        print(f"⚠️  模型文件 {MODEL_PATH} 未找到（已在.gitignore中排除，需手动下载）")
        print("💡 提示：后续运行前需手动下载该模型并放到项目根目录")

    print("✅ 环境校验完成，项目框架可正常扩展开发！")


# -------------------------- 主函数（项目入口） --------------------------
def main():
    """项目主函数"""
    print("🚪 人脸门禁系统框架初始化完成")
    # 第一步：校验环境
    check_environment()

    # 第二步：预留核心逻辑入口（后续补充）
    print("\n📌 后续开发方向：")
    print("1. 补充人脸检测/识别核心逻辑")
    print("2. 对接门禁硬件控制代码")
    print("3. 实现授权用户特征库管理")


# -------------------------- 启动入口 --------------------------
if __name__ == "__main__":
    main()