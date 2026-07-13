import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from SimpleCNN import SimpleCNN
from CSVDataset import *
from image_dataset import *

# # 1. 数据加载与预处理
# transform = transforms.Compose([
#     transforms.ToTensor(),  # 转为张量
#     transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
# ])
#
# # 加载 MNIST 数据集
# train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
#
# dataset = CSVDataset("image_data.csv")
# # print(len(dataset))
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 示例：创建数据集实例
import glob

# 假设图像文件在以下目录，按类别组织
# dataset/class1/*.jpg
# dataset/class2/*.jpg
data_dir = "dataset"

# 获取所有图像路径和标签
image_paths = []
labels = []
class_names = os.listdir(data_dir)
print(f'class_names: {class_names}')
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
print(f'class_to_idx: {class_to_idx}')

for cls_name in class_names:
    cls_dir = os.path.join(data_dir, cls_name)
    if os.path.isdir(cls_dir):
        # 获取该类别下的所有图像文件
        imgs = glob.glob(os.path.join(cls_dir, "*.jpg")) + \
               glob.glob(os.path.join(cls_dir, "*.png"))

        print(f'imgs: {imgs}')
        image_paths.extend(imgs)
        # labels.extend([class_to_idx[cls_name]] * len(imgs))
        labels.extend(cls_name)
        print(f'labels: {labels}')

# 定义数据转换 - 包括数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

print(f'labels_2: {labels}')
# 创建数据集实例
dataset = ImageDataset(
    image_paths=image_paths,
    labels=labels,
    transform=train_transform,
    img_mode='RGB'
)

dataset_len = len(dataset)
# 创建模型实例
model = SimpleCNN()


# 3. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 4. 模型训练
num_epochs = 2
model.train()  # 设置模型为训练模式

total_loss = 0
for i in range(dataset_len) :
    # logger.info(f'images[0] shape: {images[0].size()}')
    images, labels = dataset[i]
    images = images.view(-1, 1, 28, 28)  # 关键步骤：-1表示自动计算批量大小（这里是2）
    outputs = model(images)  # 前向传播
    labels = torch.tensor(eval(labels)).long()
    loss = criterion(outputs, labels)  # 计算损失

    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    total_loss += loss.item()

print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


# 保存整个模型
torch.save(model.state_dict(), './mymodel.pt')
