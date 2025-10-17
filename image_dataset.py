import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np


class ImageDataset(Dataset):
    """
    自定义图像数据集类，继承自PyTorch的Dataset

    参数:
        image_paths (list): 图像文件路径的列表
        labels (list): 对应图像的标签列表
        transform (callable, optional): 应用于图像的转换函数
        target_transform (callable, optional): 应用于标签的转换函数
        img_mode (str): 图像打开模式，'RGB'或'L'等
    """

    def __init__(self, image_paths, labels, transform=None, target_transform=None, img_mode='RGB'):
        # 检查图像路径和标签数量是否一致
        if len(image_paths) != len(labels):
            raise ValueError("图像路径数量与标签数量不一致")

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.img_mode = img_mode

        # 检查所有图像文件是否存在
        self._check_files_exist()

    def _check_files_exist(self):
        """检查所有图像文件是否存在"""
        for path in self.image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"图像文件不存在: {path}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的一个样本

        参数:
            idx (int): 样本索引

        返回:
            image (torch.Tensor): 处理后的图像张量
            label: 处理后的标签
        """
        # 获取图像路径和标签
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        # print(f'self.labels: {self.labels}')

        # 打开图像
        try:
            image = Image.open(img_path).convert(self.img_mode)
        except Exception as e:
            raise RuntimeError(f"无法打开图像 {img_path}: {str(e)}")

        # 应用图像转换
        if self.transform:
            image = self.transform(image)

        # 应用标签转换
        if self.target_transform:
            label = self.target_transform(label)

        # print(f'label: {label}')
        return image, label


# 示例用法
if __name__ == "__main__":
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

    print(f"数据集: {len(dataset)}")

    # 测试获取一个样本
    if len(dataset) > 0:
        sample_img, sample_label = dataset[0]
        print(f"图像张量形状: {sample_img.shape}")
        print(f"标签: {sample_label}")
