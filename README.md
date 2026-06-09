# RecBole 推荐实验平台

这是一个面向电商推荐实验的 RecBole 平台封装。项目统一支持数据转换、基线模型训练、自定义 CrossDomainNeuMF、全库排序评估和结果归档。

## 目录结构

```text
recbole_platform/
+-- README.md
+-- requirements.txt
+-- run_train.py                    # 统一训练 / 续训 / EVAL_ONLY 入口
+-- app.py                          # Streamlit 演示前端（python app.py）
+-- build_sequential_dataset.py      # SASRec 序列数据生成
+-- configs/                         # 模型配置
+-- crossdomain_neumf/               # 自定义 CrossDomainNeuMF 实现
+-- scripts/                         # 数据转换与评估脚本
+-- reports/                         # 主报告与模型归档
+-- results/logs/                    # 精选指标 JSON
+-- data/sample/                     # 小样例数据
+-- datasets/.gitkeep                # 数据目录占位；全量数据不进仓库
```

## 环境

在 `recbole_platform/` 目录下执行：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如需 GPU，请先按 PyTorch 官方说明安装匹配 CUDA 的 `torch`，再安装其余依赖。前端另需 `streamlit`、`plotly`（已写入 `requirements.txt`）。

## 演示前端（Streamlit）

在 `recbole_platform/` 目录下：

```powershell
python app.py
```

浏览器打开后有三个 Tab：**我的首页**（test 用户 + SASRec 推荐）、**新客发现**（热门浏览 + 勾选后 SASRec 推荐）、**状态**（权重与离线指标）。

### 演示前必需内容

| 项目 | 路径 / 说明 |
| --- | --- |
| SASRec 权重 | `D:\recbole_checkpoints\movies_tv_seq\SASRec\best.pth` |
| 交互数据 | `datasets/movies_tv/*.inter` |
| 序列数据 | `datasets/movies_tv_seq/*.inter`（选用户、SASRec 推理） |
| 展示元数据 | `datasets/movies_tv/display/`（封面、标题、用户昵称） |

说明：

- **我的首页**与点击「生成推荐」后的列表由 **SASRec** 在线推理（可选命中 `results/sasrec_future/` 预计算缓存，约 1.7GB，不必上传 Git）。
- **新客发现**的「热门发现」浏览区按全站交互频次 **Pop 排序**展示候选（统计 `.inter`，不需 Pop 模型权重）；用户勾选后，再由 **SASRec** 根据勾选序列生成推荐。
- 上述 `datasets/` 与 D 盘权重默认不进 Git，需本地准备或网盘分发。

## 数据格式

原始数据目录需要包含三份 CSV：

| 文件 | 必需列 |
| --- | --- |
| train.csv | user_id, item_id |
| valid.csv | user_id, item_id |
| test.csv | user_id, item_id |

可选列：

| 列名 | 说明 |
| --- | --- |
| rating | 缺失时默认 1.0 |
| timestamp | 缺失时默认 0 |

转换为 RecBole benchmark 数据：

```powershell
python scripts/convert_csv_to_recbole.py --src "D:\path\to\Movies_and_TV" --name movies_tv
```

生成结果：

```text
datasets/movies_tv/
+-- movies_tv.train.inter
+-- movies_tv.valid.inter
+-- movies_tv.test.inter
```

`--name` 对应后续配置里的 `dataset`。

## 训练入口

统一入口是 `run_train.py`。在文件底部可调区域修改 `MODEL`，然后运行：

```powershell
python run_train.py
```

支持的模型键：

| MODEL | 配置文件 | 说明 |
| --- | --- | --- |
| pop | configs/pop_movies_tv_full.yaml | 热门基线 |
| itemknn | configs/itemknn_movies_tv_full.yaml | ItemKNN 基线 |
| bpr | configs/bpr_movies_tv_full.yaml | MF / BPR loss |
| neumf | configs/neumf_movies_tv_full.yaml | RecBole 内置 NeuMF |
| lightgcn | configs/lightgcn_movies_tv_full.yaml | 图协同过滤 |
| sasrec | configs/sasrec_movies_tv_full.yaml | K=50 序列模型 |
| sasrec_k5 | configs/sasrec_movies_tv_k5.yaml | K=5 序列模型 |
| sasrec_k2 | configs/sasrec_movies_tv_k2.yaml | K=2 序列模型 |
| bert4rec | configs/bert4rec_movies_tv_full.yaml | BERT4Rec 序列模型 |
| crossdomain_neumf | configs/crossdomain_neumf_movies_tv_full.yaml | 自定义优化 NeuMF |

常用开关在 `run_train.py` 可调区域：

| 参数 | 作用 |
| --- | --- |
| MODEL | 选择模型 |
| DEBUG | 快速小规模运行 |
| EVAL_ONLY | 只加载已有权重评估 |
| RESUME_FROM | 断点续训权重路径 |
| EPOCHS | 覆盖 yaml 中训练轮数 |
| TAG | 指标 JSON 文件名前缀 |

默认 checkpoint 根目录在 `run_train.py` 的 `CHECKPOINT_ROOT` 中配置。当前实验使用 `D:\recbole_checkpoints`

## 评估

训练结束后，`run_train.py` 会写入：

```text
results/logs/<dataset>_<model>_<tag>.json
```

全库排序评估：

```powershell
python scripts/eval_full_catalog_hr.py
```

自定义 CrossDomainNeuMF 的全库评估：

```powershell
python scripts/eval_crossdomain_full_catalog.py
```

级联和融合评估脚本位于：

```text
scripts/eval_cascade_rank.py
scripts/eval_rrf_rank.py
```

这些脚本顶部有实验配置区，运行前按需要修改模型、checkpoint、用户 cap 和 topk。

## 脚本说明

| 脚本 | 用途 |
| --- | --- |
| scripts/convert_csv_to_recbole.py | train/valid/test.csv 转 RecBole .inter 数据 |
| scripts/build_sequential_dataset.py | 从 movies_tv 生成 SASRec/BERT4Rec 序列数据 |
| scripts/eval_full_catalog_hr.py | RecBole 模型全库排序评估 |
| scripts/eval_crossdomain_full_catalog.py | CrossDomainNeuMF 全库排序评估 |
| scripts/eval_cascade_rank.py | 粗排 -> 精排级联评估 |
| scripts/eval_rrf_rank.py | 多模型 RRF 融合评估 |
| scripts/eval_seq_len_curve.py | SASRec 推理阶段历史长度截断分析 |
| scripts/compare_sasrec_topk_overlap.py | 对比不同 SASRec 模型的 Top-K 推荐重叠 |
| scripts/init_sequential_from_general.py | 用 BPR/NeuMF item embedding 初始化序列模型 |
| scripts/precompute_sasrec_future_recs.py | 预计算 SASRec 推荐缓存（可选，加速前端） |
| scripts/bench_per_user_latency.py | 单用户推理耗时基准 |

`scripts/legacy/` 中是早期 RecBole quick_start 封装，保留用于回看历史实验，不建议作为新实验入口。

`results/sasrec_future/` 为前端可选缓存目录，体积大，已在 `.gitignore` 中忽略。

## 结果文件

| 路径 | 内容 |
| --- | --- |
| reports/Main_results.md | 最终主结果表 |
| reports/model_reports/ | 每个最终模型的报告与定稿 JSON 拷贝 |
| results/logs/ | 原始指标 JSON |

`reports/model_reports/` 是便于提交和阅读的归档目录；原始训练日志、TensorBoard、全量数据和权重不建议进入 Git。

## 复现实验的最短路径

```powershell
cd recbole_platform
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts/convert_csv_to_recbole.py --src "D:\path\to\Movies_and_TV" --name movies_tv
python run_train.py
```

运行前确认：

| 检查项 | 说明 |
| --- | --- |
| dataset | yaml 中 dataset 与转换时 --name 一致 |
| data_path | 默认 datasets/ |
| EVAL_ONLY | 没有已有权重时应设为 False |
| eval_*_user_cap | 控制验证 / 测试抽样用户数 |

