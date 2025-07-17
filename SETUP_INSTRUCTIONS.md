# 手语识别项目运行说明

## 系统要求

- Python 3.8 或更高版本
- 建议使用 Python 3.9-3.11 以获得最佳兼容性

## 环境设置

### 方法1：使用 pip 安装（推荐）

1. **创建虚拟环境**（可选但推荐）：
```bash
python -m venv slr_env
source slr_env/bin/activate  # Linux/Mac
# 或者
slr_env\Scripts\activate     # Windows
```

2. **安装依赖**：
```bash
pip install -r requirements.txt
```

### 方法2：使用 conda 安装

1. **创建conda环境**：
```bash
conda create -n slr_env python=3.9
conda activate slr_env
```

2. **安装依赖**：
```bash
pip install -r requirements.txt
```

## 项目结构

```
slr_project/
├── data/                          # 数据目录
│   └── frames/                    # 视频帧数据
│       ├── train/                 # 训练数据
│       ├── validation/            # 验证数据
│       └── test/                  # 测试数据
├── models/                        # 模型定义
│   ├── mobilenetv3.py            # MobileNetV3特征提取器
│   └── lstm_attention.py         # BiLSTM+注意力模型
├── preprocessing/                 # 数据预处理
│   └── dataset.py                # 数据集类
├── experiments/                   # 实验notebook
│   ├── 01_data_extract.ipynb     # 数据提取
│   ├── 02_dataset_augment.ipynb  # 数据增强
│   ├── 03_mobilenetv3_feature.ipynb  # 特征提取
│   ├── 04_lstm_attention.ipynb   # 模型训练
│   ├── 05_train_baseline2.ipynb  # 基础训练
│   ├── 05_train_baseline3.ipynb  # 改进训练
│   ├── 06_train_augmented.ipynb  # 增强数据训练
│   ├── 07_performance_evaluation.ipynb  # 性能评估
│   └── evaluate.ipynb            # 模型评估
├── utils/                         # 工具函数
│   └── frame_utils.py            # 帧处理工具
├── requirements.txt              # 依赖列表
└── README.md                     # 项目说明
```

## 运行步骤

### 1. 启动 Jupyter Notebook

```bash
jupyter notebook
# 或者
jupyter lab
```

### 2. 按顺序运行实验

建议按以下顺序运行notebook：

1. **01_data_extract.ipynb** - 数据提取和预处理
2. **02_dataset_augment.ipynb** - 数据增强设置
3. **03_mobilenetv3_feature.ipynb** - 特征提取器训练
4. **04_lstm_attention.ipynb** - 注意力模型设计
5. **05_train_baseline2.ipynb** - 基础模型训练
6. **06_train_augmented.ipynb** - 增强数据模型训练
7. **07_performance_evaluation.ipynb** - 性能对比评估

### 3. 主要实验对比

最终的性能评估在 `07_performance_evaluation.ipynb` 中，对比了：
- **Experiment1_Basic**: 基础模型（MobileNetV3 + BiLSTM + Attention）
- **Experiment2_Augmented**: 增强数据训练的模型

## 注意事项

1. **数据路径**: 确保 `data/frames/` 目录包含正确的训练、验证和测试数据
2. **GPU支持**: 如果有GPU，代码会自动使用CUDA加速
3. **内存要求**: 建议至少8GB RAM
4. **模型文件**: 训练完成的模型会保存为 `.pth` 文件

## 常见问题

### 问题1: 模块导入错误
```python
ModuleNotFoundError: No module named 'xxx'
```
**解决方案**: 重新运行 `pip install -r requirements.txt`

### 问题2: CUDA相关错误
**解决方案**: 如果没有GPU，代码会自动切换到CPU模式

### 问题3: 数据路径错误
**解决方案**: 检查 `data/frames/` 目录结构是否正确

## 联系信息

如果在运行过程中遇到问题，请检查：
1. Python版本是否兼容
2. 所有依赖是否正确安装
3. 数据文件是否存在

## 项目说明

这是一个基于深度学习的手语识别项目，使用了：
- **MobileNetV3**: 轻量级特征提取
- **BiLSTM**: 时序建模
- **注意力机制**: 关键帧识别
- **数据增强**: 提高模型泛化能力

项目实现了对100个手语词汇的识别，并对比了不同训练策略的效果。 