# Number_detect 项目规则

## 1. 项目概述

基于 PyTorch 的手写数字识别项目，使用 SimpleCNN 模型实现 0-9 数字分类。

## 2. 环境与依赖

### 2.1 运行环境
- **操作系统**: Windows
- **Python 版本**: 3.10.20
- **解释器路径**: `C:/anaconda/envs/yolov8/python.exe`

### 2.2 核心依赖
| 依赖包 | 版本 | 用途 |
|--------|------|------|
| torch | 2.13.0+cpu | 深度学习框架 |
| opencv-python | 5.0.0 | 图像处理 |
| numpy | 2.2.6 | 数值计算 |
| torchvision | - | 数据转换 |
| pandas | - | CSV 数据读取 |
| matplotlib | - | 可视化 |

### 2.3 安装命令
```bash
conda activate yolov8
C:/anaconda/envs/yolov8/python.exe -m pip install torch opencv-python numpy torchvision pandas matplotlib
```

**注意**: 始终使用完整解释器路径调用 pip，避免环境漂移。

## 3. 项目结构

```
Number_detect/
├── .trae/rules/           # 项目规则配置
├── base/                  # 工具模块（fileOP.py, util.py）
├── data/MNIST/            # MNIST 数据集
├── dataset/               # 自定义图片数据集（按类别组织目录）
├── SimpleCNN.py           # CNN 模型定义
├── CSVDataset.py          # CSV 数据集类
├── image_dataset.py       # 图像数据集类
├── number_detect_sava_mode_my_dataset.py  # 训练入口
├── mymodel.pt             # 训练好的模型权重
```

## 4. 代码规范

### 4.1 注释规范
- 使用中文注释
- 函数和类添加文档字符串说明

### 4.2 命名风格
- 类名: PascalCase（如 `SimpleCNN`, `CSVDataset`）
- 函数/方法名: snake_case（如 `preprocess_image`, `_check_files_exist`）
- 变量名: snake_case（如 `image_paths`, `class_to_idx`）
- 常量: UPPER_CASE（如 `data_dir`）

### 4.3 缩进
- 使用 4 空格缩进

## 5. 模型规范

### 5.1 模型架构（SimpleCNN）
- 输入: 1x28x28 灰度图像
- 卷积层: conv1(1→32) → conv2(32→64)
- 池化层: 2x2 最大池化
- 全连接层: fc1(64*7*7→128) → fc2(128→10)
- 输出: 10 类概率（数字 0-9）

### 5.2 模型保存/加载
- 保存格式: `torch.save(model.state_dict(), './mymodel.pt')`
- 加载方式:
  ```python
  model = SimpleCNN()
  model.load_state_dict(torch.load('./mymodel.pt'))
  ```

## 6. 数据集规范

### 6.1 MNIST 数据集
- 存储路径: `./data/MNIST/`
- 通过 `torchvision.datasets.MNIST` 加载

### 6.2 自定义图片数据集
- 目录结构: `dataset/{class_name}/*.png`
- 使用 `ImageDataset` 类加载
- 标签从目录名自动提取

## 7. 运行命令

### 训练模型
```bash
conda activate yolov8
C:/anaconda/envs/yolov8/python.exe d:/Number_detect/number_detect_sava_mode_my_dataset.py
```

### 图像预处理工具
```python
from CSVDataset import preprocess_image
img, img_tensor = preprocess_image(image_path)
```

## 8. 注意事项

1. **环境一致性**: 所有命令必须使用 `C:/anaconda/envs/yolov8/python.exe` 作为解释器
2. **numpy 版本**: 当前使用 numpy 2.x，如引入 tensorflow 需注意版本兼容性
3. **模型输入尺寸**: SimpleCNN 期望 28x28 灰度图输入
4. **数据格式**: CSV 数据集最后一列为标签，其余为特征