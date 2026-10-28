<div align="center">

# 🖼️ ISR_GAN_System

**基于改进 SRGAN 的图像超分辨率重建系统**

从低分辨率图像恢复出具有丰富纹理细节的高分辨率图像

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![tests](https://github.com/slw-cuddlebear/ISR-GAN-System/actions/workflows/tests.yml/badge.svg)](https://github.com/slw-cuddlebear/ISR-GAN-System/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](#-license)

![PSNR](https://img.shields.io/badge/PSNR-29.31_dB_(MSE%E9%A2%84%E8%AE%AD%E7%BB%83)-6d5dfc?style=flat-square)
![SSIM](https://img.shields.io/badge/SSIM-0.7702-6d5dfc?style=flat-square)
![LPIPS](https://img.shields.io/badge/LPIPS-0.2253_(GAN%2C%E6%84%9F%E7%9F%A5%E6%9C%80%E4%BC%98)-ef8b2c?style=flat-square)
![Scale](https://img.shields.io/badge/Upscale-4%C3%97-f59e0b?style=flat-square)

</div>
Contributor: JJ-05-H (主要负责数据采集与清洗等部分)
---

## 📖 目录

- [✨ 项目简介](#-项目简介)
- [🚀 算法改进](#-算法改进)
- [📊 实验结果](#-实验结果)
- [🗂️ 项目结构](#️-项目结构)
- [⚙️ 环境安装](#️-环境安装)
- [📦 数据准备](#-数据准备)
- [🏋️ 训练流程](#️-训练流程)
- [🖥️ 运行 Web Demo](#️-运行-web-demo)
- [💾 模型权重说明](#-模型权重说明)
- [🎨 界面预览](#-界面预览)
- [🧰 技术栈](#-技术栈)
- [📄 License](#-license)

---

## ✨ 项目简介

图像超分辨率重建是计算机视觉领域的一项经典任务，旨在从低分辨率图像恢复出高分辨率图像，广泛应用于医疗影像、卫星遥感、安防监控等场景。

传统插值方法（如双三次插值）虽然算法简单、速度快，但**无法生成新的高频细节**，重建图像边缘模糊、纹理失真。近年来，以 SRGAN 为代表的生成对抗网络能够通过学习 LR→HR 的非线性映射生成视觉上更真实的图像。

本项目以 SRGAN 为基线，针对以下问题进行了针对性改进：

- ❌ 模型参数量大，难以在普通设备上快速部署
- ❌ GAN 训练不稳定，容易模式崩溃
- ❌ 单一 MSE 损失导致重建图像过于平滑，缺乏高频纹理

最终构建了一个**功能完备、交互友好**的超分辨率重建演示系统。

### 实测效果（DIV2K test 划分，11,634 对）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓（感知） |
|---|---:|---:|---:|
| 双三次插值 | 28.581 | 0.7458 | 0.4000 |
| MSE 预训练生成器 | **29.311** | **0.7702** | 0.2542 |
| GAN 对抗微调生成器 | 27.257 | 0.6981 | **0.2253** |

**两个模型各赢一半**：MSE 预训练模型保真度最高（PSNR/SSIM 均超双三次基线），
GAN 微调模型感知质量最好（LPIPS 比预训练低 11.4%，逐样本比较有 73% 的图像更优，p < 1e-66）。
这是对抗训练的**预期代价**而非训练失败 —— 详见 [📊 实验结果](#-实验结果)。

---

## 🚀 算法改进

### 1️⃣ 生成器网络：轻量化残差密集块

在残差块的基础上增加密集连接，允许网络每一层的输出作为后续所有层的输入。同时引入**简化通道注意力（SCA）**模块，以更少的参数量实现特征重用和融合，缓解深层网络的梯度消失问题。

| 参数 | 值 |
|---|---|
| 残差密集块数量 | **4** |
| 特征通道数 | **64** |
| 内部增长通道 | **32** |
| 上采样方式 | 两次 `PixelShuffle(2)`，共 4 倍 |

### 2️⃣ 判别器网络：条件判别器

采用 4 通道输入（RGB + 语义掩码），通过 5 层卷积逐级下采样，最后经全局平均池化与全连接层输出**真伪 logit**（配合 `BCEWithLogitsLoss`，不做 sigmoid）。

> ⚠️ **当前实现状态**：语义掩码尚未真正接入。训练脚本中传入的是**全 1 掩码**（`src/train/train_gan.py`），即第 4 通道为常量，因此判别器目前**退化为普通（无条件）判别器**。条件分支的输入接口已预留，但要真正生效需要引入语义分割图作为条件。

### 3️⃣ 复合损失函数

论文中总损失由三部分加权构成：

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{mse}} \cdot \mathcal{L}_{\text{MSE}} + \lambda_{\text{adv}} \cdot \mathcal{L}_{\text{Adv}} + \lambda_{\text{percep}} \cdot \mathcal{L}_{\text{Percep}}
$$

| 损失项 | 作用 | 权重 | 当前是否启用 |
|---|---|---|---|
| 🎯 MSE 损失 | 监督像素级一致性，保证基础保真度 | `λ_mse = 1.0` | ✅ 已启用 |
| ⚔️ 对抗损失 | 非饱和对抗损失，驱动生成器生成真实高频纹理 | `λ_adv = 1e-3` | ✅ 已启用 |
| 👁️ 感知损失 | 基于预训练 VGG19 特征图欧氏距离，约束全局语义 | `λ_percep = 1e-2` | ❌ **未接入** |

> ⚠️ **当前实现状态**：`src/losses/losses.py` 中已实现 `PerceptualLoss`，但 `src/train/train_gan.py` 的训练目标是
> `λ_mse · MSE + λ_adv · Adv`，**没有把感知损失计入**，`config.py` 中的 `lambda_percep` 目前是未生效的配置项。
> 因此本项目当前实际训练的是「MSE + 对抗」两项损失。

### 4️⃣ 两阶段训练策略

```
第一阶段 ──▶ 仅用 MSE 损失预训练生成器 50 epoch
                │
                ▼
第二阶段 ──▶ 加载预训练权重，引入判别器
           使用复合损失进行 20 epoch 联合对抗训练
```

### 5️⃣ 数据预处理

- ✅ 对 DIV2K 数据集做完整性检查与清洗
- ✅ 滑动窗口裁剪成 256×256 HR patch，双三次下采样生成 64×64 LR patch
- ✅ 在线数据增强：随机水平翻转、垂直翻转、90° 旋转
- ✅ 像素归一化到 `[-1, 1]`

---

## 📊 实验结果

### 评估方式

指标由仓库内的 `evaluate.py` 计算，**可复现**：

```bash
# 1. 先生成 test 划分（需要 DIV2K 原图）
python -m src.data.preprocess --source-dir <DIV2K所在目录> --splits test

# 2. 三方对比：双三次 / MSE 预训练 / GAN 微调
python evaluate.py --split test
```

- **数据**：DIV2K 全部 900 张按原始图像 8:1:1 划分（`random_state=42`，同一原图的 patch 不跨 split），test 划分共 **11,634 对 256×256 patch**
- **指标**：PSNR、SSIM 与 **LPIPS**；四周 shave 4px（标准 SR 评测做法）
- **PSNR 两种口径都给出**：
  - **口径 A「单张均值」**：先算每对图的 PSNR 再取平均 —— SR 论文最常用的报告方式
  - **口径 B「池化」**：先对全测试集求平均 MSE 再换算 —— 对高误差样本更敏感
  - 本项目两者相差约 5 dB。原因是误差在高误差 patch 上集中（前 5% 的高误差块贡献了约 22% 的总误差），属于预期现象。**引用数字时必须注明口径。**
- **为什么必须测 LPIPS**：PSNR/SSIM 是保真度指标，而对抗训练的卖点正是「牺牲保真度换感知质量」。
  只用 PSNR 评价 GAN 会系统性地低估它——这一点在本项目上得到了直接验证（见下文结论）。

### 实测结果（DIV2K test 划分，11,634 对）

| 方法 | PSNR (dB) 口径A ↑ | PSNR (dB) 口径B | SSIM ↑ | **LPIPS ↓（感知）** |
|---|---:|---:|---:|---:|
| 🔵 双三次插值（基线） | 28.581 | 23.533 | 0.7458 | 0.4000 |
| 🟢 **MSE 预训练生成器** | **29.311** | **24.627** | **0.7702** | 0.2542 |
| 🟠 **GAN 对抗微调生成器** | 27.257 | 23.484 | 0.6981 | **0.2253** |

相对双三次基线的增益：

| 方法 | ΔPSNR (dB) | ΔSSIM | ΔLPIPS（负值更好） |
|---|---:|---:|---:|
| MSE 预训练生成器 | **+0.731** | **+0.0244** | −0.1458 |
| GAN 对抗微调生成器 | −1.324 | −0.0477 | **−0.1747** |

### 关键结论：这是一次教科书式的「保真度 ↔ 感知质量」权衡

**两个模型各赢一半，而且赢的方向符合超分领域的普遍规律：**

- **MSE 预训练模型在保真度上获胜**（PSNR +0.731 dB、SSIM +0.0244，均优于双三次基线）
- **GAN 微调模型在感知质量上获胜**（LPIPS 0.2253，比预训练模型低 **11.4%**，比双三次低 **43.7%**）

这不是「训练失败」，而是**对抗训练的预期代价**：判别器迫使生成器产出更接近自然图像统计的纹理，
代价是像素级误差增大，于是 PSNR/SSIM 下降、LPIPS 改善。
SRGAN / ESRGAN 等经典工作全是这个模式——**它们的 PSNR 普遍低于同结构的 MSE 版本**。

### 显著性检验（避免只看均值下结论）

在 1,500 个随机配对样本上的配对检验：

| 对比 | 配对 t 检验 | Wilcoxon 符号秩 | GAN 逐样本更优占比 |
|---|---|---:|---:|
| MSE 预训练 vs **GAN** | t = 18.18，**p = 6.9e-67** | p = 1.3e-85 | **73.2%** |
| 双三次 vs **GAN** | t = 58.28，p ≈ 0 | — | **91.6%** |

GAN 的感知优势**不是被少数样本拉动的均值假象**：逐样本比较有 73% 的图像 GAN 更好，
p 值远低于任何常规阈值。

### 如何看待这些数字

1. **用 PSNR 评价 GAN 会系统性地低估它。**
   本项目最初只测 PSNR/SSIM，那时的结论是「GAN 比双三次还差 1.3 dB」，看起来像训练失败；
   补上感知指标 LPIPS 后，结论变为「GAN 用保真度换到了感知质量」。
   **评价指标的选取会直接决定结论**——这是本项目在方法学上最有价值的一点。

2. **PSNR 下降 2.05 dB 放在领域内看偏大**，说明对抗权重或训练轮数仍有调优空间
   （同类工作通常在 0.5~1.5 dB）。可验证的改进方向：
   - 补上感知损失（`PerceptualLoss` 已实现但未接入训练目标，见上文「复合损失函数」）
   - 调整 `λ_adv` 或减少对抗训练轮数
   - 加入判别器特征匹配损失
   这些都有望在**保持感知质量**的前提下把 PSNR 拉回来。

3. **绝对数值低于部分论文报道，与评测协议有关。**
   本项目在 256×256 patch 上评测（LR 由 HR 经双三次 4 倍下采样得到）。
   分辨率越低的输入，4 倍放大的信息损失越严重，PSNR 天然偏低；
   标准基准（Set5/Set14）通常对完整 2K 图像下采样后再评测。

4. **主观效果**：SR 输出可见高频网格伪影（`results/` 下的对比图），
   这在 GAN 超分中常见，与「LPIPS 改善 + PSNR 下降」的组合一致。

> 📌 上表数字由 `python evaluate.py --split test` 实测得到，原始结果见 `results/eval_test_*.json` / `.csv`。
> 此前 README 曾列出 29.85 dB / 0.876，但仓库中缺少产生该数字的评测代码，无法复现，故替换为本表实测值。

---

## 🗂️ 项目结构

```
ISR-GAN-System/
├── 📄 config.py                     # 全局配置
├── 🚀 run.py                        # 一键启动脚本
├── 📋 requirements.txt              # 依赖列表
├── 📖 README.md
├── 🚫 .gitignore
├── 🖼️ img.png                       # README 预览图
│
├── 📁 src/
│   ├── 🧠 models/
│   │   ├── generator.py             # 生成器网络
│   │   └── discriminator.py         # 条件判别器
│   ├── 📊 data/
│   │   ├── preprocess.py            # 数据预处理
│   │   └── dataset.py               # 训练数据加载
│   ├── 📉 losses/
│   │   └── losses.py                # 复合损失函数
│   └── 🏋️ train/
│       ├── train_generator.py       # MSE 预训练
│       └── train_gan.py             # GAN 联合训练
│
├── 🌐 backend/
│   ├── app.py                       # Flask 主程序
│   ├── model_loader.py              # 模型加载
│   └── utils.py                     # 图像预处理/后处理
│
├── 🎨 frontend/
│   └── index.html                   # Web 演示页面
│
├── 📏 evaluate.py                   # 量化评估（PSNR/SSIM，多方对比）← 新增
├── 🧪 tests/                        # 离线单元测试 ← 新增
│   ├── test_metrics.py              #   指标实现正确性（与暴力实现交叉验证）
│   └── test_model.py                #   生成器结构、判别器、推理链路、权重加载
├── 🤖 .github/workflows/tests.yml   # CI：ruff + pytest ← 新增
│
├── 💾 checkpoints/                  # 模型权重（.pth）
├── 📦 dataset/                      # 原始数据集（需自行下载）
├── ⚙️ processed/                    # 预处理产物
├── 📈 results/                      # 评估结果与对比图
└── 📝 logs/                         # 训练日志与损失曲线
```

---

## ⚙️ 环境安装

推荐使用 Python 3.11 和虚拟环境：

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` 已按实际验证过的版本**精确锁定**（`==`），避免依赖漂移。
测试需要额外安装：`pip install pytest ruff`。

### 运行测试

```bash
python -m pytest tests/ -v          # 全部离线，不需要数据集；权重缺失时相关用例自动跳过
ruff check . --select F,E9          # 代码风格检查
```

---

## 📦 数据准备

下载 [DIV2K 数据集](https://data.vision.ee.ethz.ch/cvl/DIV2K/)，将高分辨率原图放到：

```
dataset/
├── DIV2K_train_HR/    # 800 张训练原图
└── DIV2K_valid_HR/    # 100 张验证原图
```

运行预处理脚本：

```bash
# 默认读取 dataset/，生成全部 train/val/test 划分
python -m src.data.preprocess

# 原始图在其他位置时，用 --source-dir 指定，无需复制数据
python -m src.data.preprocess --source-dir /path/to/DIV2K

# 只生成某个划分（完整 train 划分的 patch 数量很大、体积上百 GB）
python -m src.data.preprocess --splits test
```

**划分规则**：DIV2K 的 train 与 valid 原图合并后按 **8:1:1** 随机划分，
`random_state=42` 固定，**以原始图像为单位**（同一张原图切出的 patch 不会跨 split，杜绝数据泄露）。
900 张原图 → train 720 / val 90 / test 90 张；其中 test 划分实测产出 **11,634 对** patch。

预处理完成后，`processed/` 目录下会生成：

```
processed/
├── train/{HR,LR}/    # 训练集图像块
├── val/{HR,LR}/      # 验证集图像块
├── test/{HR,LR}/     # 测试集图像块
├── train_index.json  # 训练集索引
├── val_index.json    # 验证集索引
└── test_index.json   # 测试集索引
```

---

## 🏋️ 训练流程

### 第一阶段：MSE 预训练

```bash
python -m src.train.train_generator
```

- 仅使用 MSE 损失，训练 **50** 个 epoch
- 每 10 个 epoch 保存一次中间权重
- 最终权重保存至 `checkpoints/generator_pretrain.pth`
- 损失曲线保存至 `logs/generator_pretrain_loss.png`

### 第二阶段：GAN 联合训练

```bash
python -m src.train.train_gan
```

- 加载 `checkpoints/generator_epoch20.pth` 作为起点
- 使用复合损失，训练 **20** 个 epoch
- 每 5 个 epoch 保存一次检查点
- 最终权重保存至 `checkpoints/generator_final.pth`
- **部署使用** `checkpoints/generator_gan_epoch20.pth`

---

## 🖥️ 运行 Web Demo

确保 `checkpoints/generator_gan_epoch20.pth` 已存在，然后执行：

```bash
python run.py
```

脚本会**自动打开浏览器**并访问：

```
http://127.0.0.1:5000
```

页面支持：

- 🖱️ 拖拽或点击上传图片
- ⚡ 一键超分，**4 倍放大**
- 🎚️ 原图 / 超分结果滑块对比
- 💾 结果下载

> 💡 后端会**自动把任意尺寸的输入调整到合理范围**（短边 ≥ 32，长边 ≤ 512，宽高对齐到 4 的倍数），所以你上传任何尺寸的图片都可以正常处理。

### 手动启动后端

```bash
python backend/app.py
```

**API 接口**：

| 路由 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 前端页面 |
| `/health` | GET | 健康检查 |
| `/superres_base64` | POST | 上传图片，返回 base64 编码的超分结果 |

**请求示例**：

```bash
curl -X POST -F "image=@example.jpg" http://127.0.0.1:5000/superres_base64
```

**响应示例**：

```json
{
  "image_base64": "iVBORw0KGgo...",
  "input_size": [256, 256],
  "output_size": [1024, 1024]
}
```

---

## 💾 模型权重说明

本仓库已包含以下权重，**clone 后可直接运行**：

| 文件 | 大小 | 用途 |
|---|---|---|
| `checkpoints/generator_gan_epoch20.pth` | ~3.2 MB | 🟢 部署推理（推荐使用） |
| `checkpoints/generator_pretrain.pth` | ~3.2 MB | 🟡 MSE 预训练最终模型，可用于重新训练 |

> 📌 `discriminator_*.pth` 和中间检查点（`generator_epoch10/30/40/50`、`generator_gan_epoch5/10/15`）未上传，推理时不需要。

---

## 🎨 界面预览

**主界面**：

![img.png](img.png)

**使用步骤**：

1. 🖱️ 拖拽或点击上传本地图片
2. ⚡ 点击"开始超分"
3. ⏳ 等待数秒至数十秒（取决于设备）
4. 🎚️ 拖动滑块对比原图与超分结果
5. 💾 点击"下载结果"保存到本地

---

## 🧰 技术栈

| 类别 | 技术 |
|---|---|
| 🐍 编程语言 | Python 3.11（已在 3.11.0 上验证） |
| 🔥 深度学习框架 | PyTorch 2.14 |
| 👁️ 计算机视觉 | OpenCV, NumPy |
| 🌐 后端框架 | Flask 3.1 |
| 🎨 前端 | 原生 HTML5 / CSS3 / JavaScript |
| 📊 数据处理 | scikit-learn, Pillow |
| 📈 可视化 | Matplotlib, tqdm |
| ✅ 质量保障 | pytest（离线单测）+ GitHub Actions |

---

## 📄 License

MIT

---

如果这个项目对你有帮助，欢迎 Star ⭐
