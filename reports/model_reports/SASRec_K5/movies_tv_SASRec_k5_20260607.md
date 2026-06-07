# 训练报告：movies_tv · SASRec K=5

| 项目 | 阶段一 | 阶段二（续训） |
|------|--------|----------------|
| 运行标签 | `movies_tv_seq_k5_SASRec_full_20260606_234929` | `movies_tv_seq_k5_SASRec_full_20260607_014216` |
| 时间 | 2026-06-06 23:49 — 01:38+ | 2026-06-07 01:42 — 02:12+ |
| 日志 | `log/SASRec/SASRec-movies_tv_seq_k5-Jun-06-2026_23-49-29-ab19e0.log` | `log/SASRec/SASRec-movies_tv_seq_k5-Jun-07-2026_01-42-16-071b60.log` |
| 指标 JSON | `results/logs/movies_tv_seq_k5_SASRec_full_20260606_234929.json` | `results/logs/movies_tv_seq_k5_SASRec_full_20260607_014216.json` |
| Epoch 范围 | **0 — 11** | **12 — 14**（续训；日志计数连续） |
| 结束原因 | **KeyboardInterrupt**（ep11 后 test） | **早停**（`stopping_step=3`，best @ **epoch 13**） |
| 续训起点 | — | `SASRec_k5/best.pth`（阶段一 ep11 权重） |
| **全局 best** | valid NDCG **0.6755** @ ep11 | valid NDCG **0.6761** @ ep13 → **覆盖** `best.pth` |

配置文件：`configs/sasrec_movies_tv_k5.yaml`  
序列数据：`datasets/movies_tv_seq_k5`（`build_sequential_dataset.py --max-len 5 --seq-name movies_tv_seq_k5`）  
权重：`D:/recbole_checkpoints/movies_tv/SASRec_k5/best.pth`

---

## 1. 与 K=50 原模型的差异

| 项目 | K=50（`sasrec` / `movies_tv_seq`） | **K=5（本报告）** |
|------|-----------------------------------|-------------------|
| `MAX_ITEM_LIST_LENGTH` | 50 | **5** |
| `position_embedding` 行数 | 50 | **5** |
| checkpoint | `.../SASRec/best.pth` | `.../SASRec_k5/best.pth` |
| 结构其余部分 | hidden 128, 2L2H, CE | **相同** |

训练/评估协议：valid/test **uni100**，用户 cap **20,000**，`eval_sample_seed=42`（与主表 BPR/LightGCN/SASRec 同族）。

---

## 2. 数据背景（movies_tv 扁平集）

| 统计量 | 值 |
|--------|-----|
| 用户数 | 657,203 |
| 每用户总交互 N（均值 / 中位数） | **11.32 / 7** |
| test 前历史（train+valid，均值 / 中位数） | **10.32 / 6** |
| N=5（5-core 下限）用户占比 | **24.8%** |
| test 前历史 ≤5 的用户 | **41.2%** → 对这部分用户，K=5 序列 **等于全历史** |

K=5 序列文件与 K=50 行数相同（增广步数一致），仅 `item_id_list` 最大宽度为 5。

---

## 3. 模型与训练超参

### 3.1 结构

| 参数 | 值 |
|------|-----|
| `model` | SASRec |
| `hidden_size` / `n_layers` / `n_heads` / `inner_size` | 128 / 2 / 2 / 256 |
| `loss_type` | CE |
| **`MAX_ITEM_LIST_LENGTH`** | **5** |
| `train_batch_size` / `eval_batch_size` | 512 / 2048 |
| `seed` | 2020 |

### 3.2 优化（两阶段）

| 参数 | yaml | **阶段一实际** | **阶段二实际** |
|------|------|----------------|----------------|
| `learning_rate` | 0.001 | **0.0001** | **0.0001** |
| `weight_decay` | 0.0 | **1e-6** | **1e-5** |
| `epochs` 上限 | 30 | 30（ep11 中断） | 30（ep14 早停） |
| `stopping_step` | 3 | 3 | 3 |

> 阶段一/二学习率均低于 yaml 默认 0.001，与 K=50 续训时的小 lr 策略类似。

### 3.3 平台评估参数（两阶段相同）

| 参数 | 值 |
|------|-----|
| `num_negatives_eval` | 100 → uni100 |
| `eval_valid_user_cap` / `eval_test_user_cap` | 20000 / 20000 |
| `eval_sample_seed` | 42 |
| `metrics` / `topk` / `valid_metric` | Hit, NDCG, MRR, Recall, GAUC / [10] / NDCG@10 |
| 每 epoch 训练 batch 数 | **10,683**（5,469,520 ÷ 512） |

---

## 4. 训练过程（逐 epoch）

valid 均为 **uni100 · cap 20,000**；`valid_score` = NDCG@10。

### 4.1 阶段一（ep0 — ep11）

- **结束**：ep11 valid 后 **KeyboardInterrupt**；平台加载 **best.pth（ep11）** 跑 test（见 JSON `234929`）。
- 每 epoch 耗时约 **train ~530 s（~8.8 min）+ valid ~6.5 s**。

| Epoch | 训练耗时 | train loss | NDCG@10 | HR@10 | MRR@10 | GAUC | 存 best |
|------:|---------:|-----------:|--------:|------:|-------:|-----:|:-------:|
| 0 | 525.3s | 113781.89 | 0.5490 | 0.7403 | 0.4889 | 0.8999 | 是 |
| 1 | 525.4s | 106100.56 | 0.6014 | 0.7812 | 0.5450 | 0.9183 | 是 |
| 2 | 527.8s | 101167.32 | 0.6381 | 0.8101 | 0.5839 | 0.9294 | 是 |
| 3 | 525.8s | 97620.56 | 0.6554 | 0.8240 | 0.6021 | 0.9350 | 是 |
| 4 | 525.4s | 95286.10 | 0.6633 | 0.8287 | 0.6109 | 0.9371 | 是 |
| 5 | 547.9s | 93618.34 | 0.6689 | 0.8324 | 0.6171 | 0.9390 | 是 |
| 6 | 544.1s | 92391.68 | 0.6691 | 0.8338 | 0.6169 | 0.9396 | 是 |
| 7 | 528.7s | 91461.49 | 0.6715 | 0.8345 | 0.6199 | 0.9394 | 是 |
| 8 | 527.3s | 90730.89 | 0.6729 | 0.8368 | 0.6210 | 0.9401 | 是 |
| 9 | 537.0s | 90152.79 | 0.6748 | 0.8359 | 0.6238 | 0.9399 | 是 |
| 10 | 540.5s | 89692.02 | 0.6754 | 0.8357 | 0.6247 | 0.9398 | 是 |
| **11** | 530.8s | 89314.78 | **0.6755** | 0.8358 | 0.6247 | 0.9392 | **是** |

**阶段一累计（ep0–11）**：约 **109 min**（train+valid）。

**阶段一结束 test**（best @ ep11，来源 `movies_tv_seq_k5_SASRec_full_20260606_234929.json`）：

| 集合 | HR@10 | NDCG@10 | MRR@10 | GAUC |
|------|-------|---------|--------|------|
| Valid | 0.8358 | 0.6755 | 0.6247 | 0.9392 |
| Test | 0.7912 | 0.6211 | 0.5676 | 0.9195 |

### 4.2 阶段二（续训 ep12 — ep14）

- **启动**：`Checkpoint loaded. Resume training from epoch 12`；`lr=0.0001`，`weight_decay=1e-5`。
- **结束**：ep14 后 **早停**（ep13 为全局 best 后连续 2 轮未提升 + 本轮未提升，`stopping_step=3`）；加载 **best.pth（ep13）** 跑 test。
- 每 epoch 仍约 **train ~540 s + valid ~6.5 s**。

| Epoch | 训练耗时 | train loss | NDCG@10 | HR@10 | MRR@10 | GAUC | 存 best |
|------:|---------:|-----------:|--------:|------:|-------:|-----:|:-------:|
| 12 | 541.2s | 88868.18 | 0.6749 | 0.8341 | 0.6246 | 0.9392 | — |
| **13** | 540.2s | 88681.26 | **0.6761** | **0.8373** | **0.6252** | 0.9397 | **是** |
| 14 | 539.8s | 88536.80 | 0.6749 | 0.8360 | 0.6238 | 0.9395 | — |

**阶段二累计（ep12–14）**：约 **28 min**。

**阶段二结束 test**（best @ ep13，来源 `best_meta.json` / JSON `014216`）：

| 集合 | HR@10 | NDCG@10 | MRR@10 | GAUC |
|------|-------|---------|--------|------|
| Valid | 0.8373 | 0.6761 | 0.6252 | 0.9397 |
| Test | 0.7895 | 0.6233 | 0.5709 | 0.9214 |

> **全局 best**：ep13（valid NDCG **0.6761**），覆盖阶段一 ep11 的 0.6755；test NDCG 由 0.6211 → **0.6233**（+0.0022）。

---

## 5. uni100 主表指标（cap 20k test）

来源：`SASRec_k5/best_meta.json`（2026-06-07），与 K=50 的 `SASRec/best_meta.json`（2026-06-06）对比。

### 5.1 Valid（best epoch）

| 模型 | NDCG@10 | HR@10 | MRR@10 |
|------|---------|-------|--------|
| K=50 SASRec | **0.6904** | 0.8456 | 0.6412 |
| **K=5 SASRec** | 0.6761 | 0.8373 | 0.6252 |
| **Δ** | **−0.0143** | −0.0083 | −0.0160 |

### 5.2 Test

| 模型 | NDCG@10 | HR@10 | MRR@10 |
|------|---------|-------|--------|
| K=50 SASRec | **0.6347** | **0.8010** | **0.5821** |
| **K=5 SASRec** | 0.6233 | 0.7895 | 0.5709 |
| **Δ** | **−0.0114** | −0.0115 | −0.0112 |

**结论（uni100）**：训测一致的 K=5 模型在主表协议下 **全面略低于 K=50**，幅度约 **1.1–1.4 个 NDCG 点**，与「更短上下文丢信息」一致。

### 5.3 全库 test（表 7 协议）

脚本：`scripts/eval_full_catalog_hr.py`（`MODEL=sasrec_k5`，`FULL_CATALOG_USER_CAP=0`，2026-06-07）

| 指标 | K=50 SASRec | **K=5 SASRec** | **Δ** |
|------|-------------|----------------|-------|
| 用户数 | 657,203 | 657,203 | 相同 |
| 物品数 | ~197,944 | ~197,944 | 相同 |
| **HR@10** | **0.0913** | **0.0874** | −0.0039 |
| **HR@50** | **0.1829** | **0.1744** | −0.0085 |
| **NDCG@10** | **0.0469** | **0.0434** | −0.0035 |
| **NDCG@50** | 待同脚本补 | **0.0624** | — |
| **MRR@10** | **0.0332** | **0.0298** | −0.0034 |
| **MRR@50** | 待同脚本补 | **0.0338** | — |

耗时约 **12 s**（321 batch × ~2048 用户，与 K=50 同量级）。

**结论（全库）**：K=5 在全库 test 上 **各指标均略低于 K=50**（HR@10 约 −0.4pp），与 uni100 结论一致；不存在「短序列在全库更优」。

---

## 6. 单用户 Top-100 全库列表对比（K50 vs K5）

脚本：`scripts/compare_sasrec_topk_overlap.py`

```powershell
python scripts/compare_sasrec_topk_overlap.py
```

可调：`USER_TOKEN`（指定用户 token）、`TOPK=100`、`MIN_HIST`（自动选用户时最少历史长度）。

### 6.1 协议

- 同一 test 用户、同一 **完整** test 前历史（来自 `movies_tv_seq.test`）。
- K=50 模型：输入最近 **50** 条；K=5 模型：输入最近 **5** 条（各用各自 checkpoint）。
- 对 **~197,944 item** 全库 `full_sort_predict`，mask 已见历史 + pad。
- 比较两路 **Top-100** item 集合：交集、差集、Jaccard、test 正例全库排名。

### 6.2 示例用户（自动抽样，`MIN_HIST=10`，seed=42）

| 字段 | 值 |
|------|-----|
| `user_token` | **493464** |
| test 前历史 | **25** 条（K50 用 25，K5 用 5） |
| test 正例 | item **30452** |

| 指标 | 值 |
|------|-----|
| Top-100 **相同** | **48 / 100（48%）** |
| **仅 K50** 有 | **52** |
| **仅 K5** 有 | **52** |
| Jaccard@100 | **0.316** |
| 正例全库排名 | K50 **#2541** · K5 **#1462** |

K50 Top-5：`1385, 32571, 49773, 63495, 5198`  
K5 Top-5：`15303, 813, 19395, 1046, 18911`（与 K50 Top-5 **无一项重合**）

JSON：`results/logs/sasrec_k50_vs_k5_top100_20260607_021643.json`

### 6.3 解读（严格）

1. **列表差异大**：即使 hist=25 的用户，Top-100 也只有约一半重合；Top-5 排序路径几乎完全不同。  
2. **不等于互补**：正例在 K5 下排名更高（1462 vs 2541）是个例；uni100 上 K5 整体更差。  
3. **融合仍需谨慎**：两路分数尺度不同，简单加权需 valid 调参；且 41% 短历史用户 K5≈全长，融合收益面更窄。  

---

## 7. 复现命令

```powershell
cd recbole_platform

# 训练
# run_train.py → MODEL = "sasrec_k5"

# 重建序列（若需）
python scripts/build_sequential_dataset.py --dataset movies_tv --max-len 5 --seq-name movies_tv_seq_k5

# Top-100 对比
python scripts/compare_sasrec_topk_overlap.py
# 指定用户：脚本内 USER_TOKEN = "493464"
```

---

## 8. 待办

- [x] K=5 **全库 HR/NDCG/MRR**（§5.3，2026-06-07）  
- [ ] 多样本 Top-100 重叠率：对 hist>15 用户抽样 N=100，报告平均 Jaccard@100  
- [ ] **K=2** 模型训练（`MODEL=sasrec_k2`，数据已生成 `movies_tv_seq_k2`）后与 K=50 同脚本对比
