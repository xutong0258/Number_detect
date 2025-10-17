import cv2
import numpy as np
import csv
import os


def process_image(image_path):
    """读取灰度图，调整为28x28，并返回一维像素数组"""
    # 读取灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 调整尺寸为28x28
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # 将二维数组转换为一维数组 (28x28 -> 784)
    img_flattened = img_resized.flatten()

    return img_flattened


def save_to_csv(image_array, csv_path, label=None):
    """将图像数组保存到CSV文件的一行中"""
    # 如果文件不存在，创建并写入表头
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # 写入表头（仅当文件不存在时）
        if not file_exists:
            # 表头：如果有标签则第一个字段为"label"，其余为像素位置
            header = []
            if label is not None:
                header.append("label")
            header.extend([f"pixel_{i}" for i in range(len(image_array))])
            writer.writerow(header)

        # 准备要写入的行
        row = []
        if label is not None:
            row.append(label)
        row.extend(image_array.tolist())

        # 写入数据行
        writer.writerow(row)


if __name__ == "__main__":
    # 配置参数
    files = ['4.png', '9.png']
    # image_path = "4.png"  # 输入图像路径
    csv_path = "image_data.csv"  # 输出CSV文件路径
    label = None  # 可选：为图像添加标签，如0-9的数字

    for file in files:
        try:
            # 处理图像
            image_array = process_image(file)
            print(f"图像处理完成，像素数量: {len(image_array)}")

            # 保存到CSV
            label = file.replace(".png", "")
            save_to_csv(image_array, csv_path, label)
            print(f"已成功保存到CSV文件: {csv_path}")

        except Exception as e:
            print(f"处理失败: {str(e)}")
