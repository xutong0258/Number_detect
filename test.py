import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# import cv2
import numpy as np
# from base.util import *

from SimpleCNN import SimpleCNN


# 创建模型实例
model = SimpleCNN()
# 把参数塞进模型里面
model.load_state_dict(torch.load('./mymodel.pt'))

# 定义数据预处理的流水线
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 将图像调整为 128x128
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485], std=[0.229])  # 标准化
])

# # 加载图像
# image = Image.open('image.jpg')
#
# # 应用预处理
# image_tensor = transform(image)
# print(image_tensor.shape)  # 输出张量的形状

files = ['13.png', '9.png']
# files = ['4.png']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i in range(2):
    image = Image.open(files[i])

    input_tensor = transform(image)

    outputs = model(input_tensor)
    _, predictions = torch.max(outputs, 1)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Pred: {predictions.item()}")
    axes[i].axis('off')
plt.show()