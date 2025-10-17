import torch
import cv2
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

# 自定义 CSV 数据集
class CSVDataset(Dataset):
    def __init__(self, file_path):
        # 读取 CSV 文件
        self.data = pd.read_csv(file_path)

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 使用 .iloc 明确基于位置索引
        row = self.data.iloc[idx]
        # 将特征和标签分开
        features = torch.tensor(row.iloc[:-1].to_numpy(), dtype=torch.float32)  # 特征
        label = torch.tensor(row.iloc[-1], dtype=torch.float32)  # 标签
        # print(f'features: {features}, label: {label}')
        return features, label

def preprocess_image(image_path):
    # 读取灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 调整尺寸为28x28（根据模型结构反推的输入尺寸）
    # 模型最后展平层为64*7*7，反推输入尺寸：(7*2*2) = 28
    img_resized = cv2.resize(img, (28, 28))

    # 转换为float32并归一化到[0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    # print(f'img_normalized: {img_normalized}')

    # 扩展维度：(28, 28) → (1, 28, 28) → (1, 1, 28, 28)
    # 对应PyTorch的输入格式：(batch_size, channels, height, width)
    img_tensor = torch.from_numpy(img_normalized)
    img_tensor = img_tensor.unsqueeze(0)  # 添加通道维度
    img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度

    return img, img_tensor

# # 实例化数据集和 DataLoader
# dataset = CSVDataset("image_data.csv")
# print(len(dataset))
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
#
# # 遍历 DataLoader
# for features, label in dataloader:
#     print("特征:", features)
#     print("标签:", label)
#     break