import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
from base.util import *

from SimpleCNN import SimpleCNN


# 创建模型实例
model = SimpleCNN()

# 把参数塞进模型里面
model.load_state_dict(torch.load('./mymodel.pt'))

# 2. 读取并预处理图片
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


# 6. 可视化测试结果
# 3. 预处理图片


files = ['13.png', '9.png']
# files = ['4.png']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i in range(2):
    try:
        img, input_tensor = preprocess_image(files[i])
    except ValueError as e:
        print(e)
        exit()

    outputs = model(input_tensor)
    _, predictions = torch.max(outputs, 1)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Pred: {predictions.item()}")
    axes[i].axis('off')
plt.show()