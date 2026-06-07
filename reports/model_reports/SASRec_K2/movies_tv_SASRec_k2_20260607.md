# 训练报告：movies_tv · SASRec K=2（阶段一）

| 项目 | 阶段一 |
|------|--------|
| 运行标签 | `movies_tv_seq_k2_SASRec_full_20260607_105351` |
| 时间 | 2026-06-07 10:53 — 12:25+ |
| 日志 | `log/SASRec/SASRec-movies_tv_seq_k2-Jun-07-2026_10-53-51-640218.log` |
| 指标 JSON | `results/logs/movies_tv_seq_k2_SASRec_full_20260607_105351.json` |
| Epoch 范围 | **0 — 9** |
| 结束原因 | **KeyboardInterrupt**（ep9 存 best 后跑 test） |
| **阶段一 best** | valid NDCG **0.6526** @ **epoch 9** |

配置文件：`configs/sasrec_movies_tv_k2.yaml`  
序列数据：`datasets/movies_tv_seq_k2`（`build_sequential_dataset.py --max-len 2 --seq-name movies_tv_seq_k2`）  
权重：`D:/recbole_checkpoints/movies_tv/SASRec_k2/best.pth`

> 阶段二（续训）**尚未进行**。下文为阶段一完整记录；与 K=5 报告结构对齐，便于 K 消融横向对比。

---

## 1. 与 K=5 / K=50 的差异

| 项目 | K=50 | K=5 | **K=2（本报告）** |
|------|------|-----|-------------------|
| `MAX_ITEM_LIST_LENGTH` | 50 | 5 | **2** |
| `position_embedding` 行数 | 50 | 5 | **2** |
| checkpoint | `.../SASRec/` | `.../SASRec_k5/` | **`.../SASRec_k2/`** |
| 结构其余 | hidden 128, 2L2H, CE | 相同 | **相同** |

训练/评估协议：valid/test **uni100**，用户 cap **20,000**，`eval_sample_seed=42`。

**数据含义（K=2）**：test 前历史 median=6、但序列宽度仅 2 → **绝大多数用户** 训练/推理时只看到 **最近 2 次点击**（仅 5-core 下限 N=5 用户中 hist=4 的 24.8% 会看到完整 2 条前的 4 条被截成 2 条）。

---

## 2. 模型与训练超参（阶段一）

### 2.1 结构

| 参数 | 值 |
|------|-----|
| `model` | SASRec |
| `hidden_size` / `n_layers` / `n_heads` / `inner_size` | 128 / 2 / 2 / 256 |
| `loss_type` | CE |
| **`MAX_ITEM_LIST_LENGTH`** | **2** |
| `train_batch_size` / `eval_batch_size` | 512 / 2048 |
| `seed` | 2020 |

### 2.2 优化（阶段一实际）

| 参数 | yaml | **阶段一实际** |
|------|------|----------------|
| `learning_rate` | 0.001 | **0.0001**（`run_train.py` 中 `RESUME_LR=1e-4` 覆盖 yaml） |
| `weight_decay` | 0.0 | **1e-5**（`RESUME_WEIGHT_DECAY=1e-5` 覆盖 yaml） |
| `epochs` 上限 | 30 | 30（ep9 中断） |
| `stopping_step` | 3 | 3（未触发早停） |
| `RESUME_FROM` | — | **None**（从头训练） |

> 与 K=5 阶段一相比：lr 同为 **1e-4**；K=5 阶段一 wd 为 **1e-6**，K=2 为 **1e-5**（当前 `run_train` 可调区默认）。跨 K 对比时请注明 **非完全同一超参网格**。

### 2.3 平台评估参数

| 参数 | 值 |
|------|-----|
| `num_negatives_eval` | 100 → uni100 |
| `eval_valid_user_cap` / `eval_test_user_cap` | 20000 / 20000 |
| `eval_sample_seed` | 42 |
| 每 epoch 训练 batch 数 | **10,683** |

---

## 3. 训练过程（逐 epoch · 阶段一）

valid：**uni100 · cap 20,000**；`valid_score` = NDCG@10。

- **结束**：ep9 valid 后 **KeyboardInterrupt**；加载 **best.pth（ep9）** 跑 uni100 test。
- 每 epoch 约 **train ~528 s（~8.8 min）+ valid ~6.3 s**。
- **阶段一累计（ep0–9）**：约 **89 min**。

| Epoch | 训练耗时 | train loss | NDCG@10 | HR@10 | MRR@10 | GAUC | 存 best |
|------:|---------:|-----------:|--------:|------:|-------:|-----:|:-------:|
| 0 | 522.4s | 114466.22 | 0.5378 | 0.7300 | 0.4776 | 0.8977 | 是 |
| 1 | 519.1s | 106826.95 | 0.5910 | 0.7716 | 0.5343 | 0.9165 | 是 |
| 2 | 533.3s | 101903.87 | 0.6223 | 0.7960 | 0.5676 | 0.9260 | 是 |
| 3 | 525.7s | 98469.30 | 0.6379 | 0.8107 | 0.5836 | 0.9307 | 是 |
| 4 | 518.7s | 96304.62 | 0.6445 | 0.8157 | 0.5905 | 0.9324 | 是 |
| 5 | 536.3s | 94850.50 | 0.6501 | 0.8180 | 0.5972 | 0.9330 | 是 |
| 6 | 541.5s | 93844.14 | 0.6488 | 0.8188 | 0.5951 | 0.9334 | — |
| 7 | 534.7s | 93107.69 | 0.6515 | 0.8185 | 0.5988 | 0.9325 | 是 |
| 8 | 529.0s | 92560.76 | 0.6508 | 0.8194 | 0.5976 | 0.9320 | — |
| **9** | 522.2s | 92137.31 | **0.6526** | 0.8188 | **0.6001** | 0.9322 | **是** |

**阶段一结束 test**（best @ ep9，来源 `105351.json` / `best_meta.json`）：

| 集合 | HR@10 | NDCG@10 | MRR@10 | GAUC |
|------|-------|---------|--------|------|
| Valid | 0.8188 | 0.6526 | 0.6001 | 0.9322 |
| Test | 0.7738 | 0.6011 | 0.5469 | 0.9120 |

---

## 4. K 消融快照（uni100 · cap 20k test）

| 模型 | 训练阶段 | Valid NDCG@10 | Test NDCG@10 | Test HR@10 |
|------|----------|---------------|--------------|------------|
| K=50 | 全局 best（续训后） | 0.6904 | **0.6347** | **0.8010** |
| K=5 | 全局 best（续训后） | 0.6761 | 0.6233 | 0.7895 |
| **K=2** | **阶段一 ep9** | 0.6526 | 0.6011 | 0.7738 |

趋势：**K 越小，uni100 指标越低**；K=2 阶段一 test NDCG 较 K=5 全局 best **−0.0222**，较 K=50 **−0.0336**。

全库 HR（表 7）：**待跑**（`eval_full_catalog_hr.py`，`MODEL=sasrec_k2`）。

---

## 5. 复现命令

```powershell
cd recbole_platform

# 训练阶段一（当前 run_train 默认 RESUME_LR=1e-4, RESUME_WEIGHT_DECAY=1e-5）
# MODEL = "sasrec_k2"

python scripts/build_sequential_dataset.py --dataset movies_tv --max-len 2 --seq-name movies_tv_seq_k2
```

---

## 6. 待办

- [ ] 阶段二续训（`RESUME_FROM=SASRec_k2/best.pth`，与 K=5 同策略）  
- [ ] 全库 test（`eval_full_catalog_hr.py`）  
- [ ] 与 K=5/K=50 的 Top-100 列表对比（`compare_sasrec_topk_overlap.py`，需扩展多模型）  
- [ ] 可选：固定 lr/wd 的 strict K 消融（三 K 均 `RESUME_LR=None`，yaml lr=0.001）
