# 项目名称

> 一句话描述这个项目是做什么的（例如：基于 YOLOv7 的 XX 目标检测系统）。

## 环境要求

- Python 3.8+
- CUDA 11.1+（GPU 训练必须）
- 推荐显存 ≥ 8GB

## 快速启动

```bash
# 1. 克隆仓库（含 YOLOv7 子模块）
git clone https://github.com/WongKinYiu/yolov7.git
# 将本仓库的文件覆盖到 yolov7/ 目录，或按队内约定的目录结构使用

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入数据集路径等

# 5. 获取数据集（见下方说明）

# 6. 下载预训练权重
# 从 YOLOv7 官方 Release 页面下载 yolov7.pt，放到项目根目录
# https://github.com/WongKinYiu/yolov7/releases

# 7. 开始训练
python train.py --workers 4 --device 0 \
  --batch-size 16 --epochs 100 \
  --data data/dataset.yaml \
  --img 640 640 \
  --cfg cfg/training/yolov7.yaml \
  --weights yolov7.pt \
  --name exp

# 8. 推理
python detect.py \
  --weights runs/train/exp/weights/best.pt \
  --source <图片/视频路径> --img-size 640
```

## 复现论文实验结果

> 论文评审会验证可复现性，以下步骤应能精确还原提交结果。

```bash
python train.py --workers 4 --device 0 \
  --batch-size 16 --epochs 100 \
  --data data/dataset.yaml \
  --img 640 640 \
  --cfg cfg/training/yolov7.yaml \
  --weights yolov7.pt \
  --name final_run

# 评估（输出 mAP 等指标）
python test.py \
  --weights runs/train/final_run/weights/best.pt \
  --data data/dataset.yaml \
  --img-size 640
```

论文中所有图表均可由 `paper/figures/` 目录下的脚本生成。

## 数据集获取

**数据集由比赛主办方在比赛开始后统一下发。**

拿到数据集后：
1. 解压到项目根目录的 `data/` 文件夹
2. 确认目录结构如下，不符合则需转换格式：

```
data/
├── dataset.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

3. 更新 `.env` 中的 `DATA=data/dataset.yaml`

## 大文件共享方式

因为大家在同一间屋子，优先用 **U盘** 传输，速度最快：

| 文件类型 | 共享方式 |
|----------|----------|
| 数据集（主办方下发） | U盘拷贝给每人 |
| 训练中的权重 | U盘 或 局域网共享 |
| **最终提交权重** | **直接提交给主办方**（见下方） |

## 最终提交

比赛结束时需提交：
1. **代码**：本 git 仓库（含 `paper/`）
2. **论文**：`paper/main.pdf`（编译自 `paper/main.tex`）
3. **模型权重**：`runs/train/final_run/weights/best.pt`，直接提交给主办方

提交前检查：
- [ ] `best.pt` 能正常加载并推理
- [ ] README 中的复现步骤亲测可跑通
- [ ] 论文里的数字与实际评估输出一致

## 项目结构

```
main-repo/
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── .env.example
├── paper/                    # 论文源文件
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── data/                     # 数据集（不进 git，U盘共享）
└── runs/                     # 训练输出（不进 git，最终 best.pt 直接提交）
```

## 模块分工

| 模块 | 负责人 | 目录 / 文件 |
|------|--------|-------------|
| 数据处理 | | `data/` |
| 模型训练 | | `train.py` |
| 推理 / 后处理 | | `detect.py` |
| 评估 & 指标 | | `test.py` |
| 论文写作 | | `paper/` |

## 团队成员

- （在此添加团队成员）
