import os
from PIL import Image
import numpy as np

# 定义输入和输出文件夹路径
input_folder = "/home2/zxp/Projects/FoundationPose/demo_data/square_table_leg/depth_2"  # 替换为你的输入文件夹路径
output_folder = "/home2/zxp/Projects/FoundationPose/demo_data/square_table_leg/depth_3"  # 替换为你的输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有图片文件
for filename in os.listdir(input_folder):
    # 构造完整文件路径
    input_path = os.path.join(input_folder, filename)

    # 确保文件是图片（可以根据需要扩展支持的文件类型）
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
        # 打开图片并转换为 NumPy 数组
        image = Image.open(input_path).convert("I;16")  # "I" 模式支持 uint16
        image_array = np.array(image, dtype=np.uint16)

        # 用 65535 减去每个像素值
        inverted_array = 65535 - image_array

        # 转换回图片并保存到输出文件夹
        output_image = Image.fromarray(inverted_array, mode="I;16")
        output_path = os.path.join(output_folder, filename)
        output_image.save(output_path)

        print(f"已处理并保存：{output_path}")
