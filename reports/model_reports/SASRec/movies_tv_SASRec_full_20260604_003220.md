# 训练报告：movies_tv · SASRec（阶段一 + 阶段二续训）

| 项目 | 阶段一 | 阶段二（续训） |
|------|--------|----------------|
| 运行标签 | `movies_tv_SASRec_full_20260604_003220` | `movies_tv_seq_SASRec_full_20260604_125756` |
| 时间 | 2026-06-04 00:32:20 — 02:10+ | 2026-06-04 12:57:56 — 14:09+ |
| 日志 | `log/SASRec/SASRec-movies_tv_seq-Jun-04-2026_00-32-20-de2339.log` | `log/SASRec/SASRec-movies_tv_seq-Jun-04-2026_12-57-56-217474.log` |
| 指标 JSON | `results/logs/movies_tv_seq_SASRec_full_20260604_003220.json` | `results/logs/movies_tv_seq_SASRec_full_20260604_125756.json` |
| Epoch 范围 | **0 — 10**（ep11 未开训） | **9 — 16**（从 checkpoint 续，日志计数连续） |
| 结束原因 | **KeyboardInterrupt**（未早停） | **早停**（`stopping_step=3`，best @ **epoch 12**） |
| 续训起点 | — | `RESUME_FROM` = 阶段一结束时的 `best.pth`（阶段一 best 为 ep8 权重） |
| **全局 best** | valid NDCG **0.6870** @ ep8 | valid NDCG **0.6874** @ ep12 → **覆盖** `best.pth` |

配置文件：`configs/sasrec_movies_tv_full.yaml`（两阶段共用模型结构；优化器参数见下表）

底层数据：`datasets/movies_tv` → 序列化 `datasets/movies_tv_seq`

---

## 1. 数据与协议

### 1.1 原始划分（与 BPR/LightGCN 一致）

来源：`Movies_and_TV`，按用户 **leave-one-out**（train = 前 N−2，valid = N−1，test = N）。

| 文件 | 行数 | 用户数 |
|------|------|--------|
| `movies_tv.train.inter` | 6,126,723 | 657,204 |
| `movies_tv.valid.inter` | 657,203 | 657,203 |
| `movies_tv.test.inter` | 657,203 | 657,203 |

### 1.2 序列数据集 `movies_tv_seq`（SASRec 专用）

由 `scripts/build_sequential_dataset.py` 从扁平 benchmark 生成（`MAX_ITEM_LIST_LENGTH=50`）：

| Split | 行数 | 含义 |
|-------|------|------|
| **train** | **5,469,520** | 序列增广：每个用户多步「历史 → 下一 item」 |
| **valid** | 657,203 → **cap 20,000** | 历史 = train，预测 valid 目标 item |
| **test** | 657,203 → **cap 20,000** | 历史 = train + valid，预测 test 目标 item |

`.inter` 列：`user_id`, `item_id`（目标）, `item_id_list`（空格分隔历史）, `item_length`。

加载后统计（cap 前，日志）：

| 统计量 | 值 |
|--------|-----|
| 用户数 | 657,204 |
| 物品数 | 197,944 |
| 交互行（合并后） | 5,509,520 |
| 稀疏度 | 99.996% |

### 1.3 评估协议

| 阶段 | 用户 | 候选 | 说明 |
|------|------|------|------|
| valid / test | `eval_*_user_cap: 20000` | **uni100**（1 正 + 100 负） | 与 BPR/LightGCN 同族指标 |
| **全库 HR** | **657,203**（test 全量） | **~197,944 item** 全排序 | 见 **§11**（`eval_full_catalog_hr.py`，约 **12 s**） |

**注意**：SASRec 训练为 **CE + 历史序列**，不用 BPR 式 `num_negatives_train` / 动态难负例；与通用模型 **训练机制不同**，valid/test 仍可横向比 uni100 数值，但需在报告中注明协议差异。

---

## 2. 模型与训练超参

### 2.1 SASRec 结构（两阶段相同）

| 参数 | 值 |
|------|-----|
| `model` | SASRec |
| `hidden_size` | 128 |
| `n_layers` / `n_heads` | 2 / 2 |
| `inner_size` | 256 |
| `hidden_dropout_prob` / `attn_dropout_prob` | 0.3 / 0.3 |
| `hidden_act` | gelu |
| `layer_norm_eps` | 1e-12 |
| `initializer_range` | 0.02 |
| `loss_type` | **CE**（全 item softmax） |
| `MAX_ITEM_LIST_LENGTH` | 50 |
| `train_neg_sample_args` | **none**（CE，无 BPR 负采样） |
| 可训练参数 | **25,608,448** |
| `learner` | adam |
| `train_batch_size` / `eval_batch_size` | 512 / 2048 |
| `seed` / `reproducibility` | 2020 / True |
| `gpu_id` / `device` | 0 / cuda |
| `show_progress` | True |
| `save_dataset` / `save_dataloaders` | False / False |

### 2.2 数据加载（两阶段相同）

| 参数 | 值 |
|------|-----|
| `dataset` | movies_tv_seq |
| `data_path` | datasets/ |
| `benchmark_filename` | ['train', 'valid', 'test'] |
| `field_separator` | `\t` |
| `seq_separator` | 空格 |
| `load_col.inter` | [user_id, item_id, item_id_list] |
| `alias_of_item_id` | [item_id_list] |
| `TIME_FIELD` | timestamp |
| `ITEM_LIST_LENGTH_FIELD` | item_length |
| `LIST_SUFFIX` | _list |
| `user_inter_num_interval` / `item_inter_num_interval` | [1,inf) / [1,inf) |

### 2.3 优化与调度（分阶段，互不覆盖）

| 参数 | yaml 默认 | **阶段一实际** | **阶段二实际**（`run_train.py` 可调区） |
|------|-----------|----------------|----------------------------------------|
| `epochs`（上限） | 30 | **50**（`EPOCHS=50`） | **25**（`EPOCHS=25`） |
| `learning_rate` | 0.001 | 0.001 | **0.0005**（`RESUME_LR`） |
| `weight_decay` | 0.0 | **0.0** | **1e-5**（`RESUME_WEIGHT_DECAY`） |
| `eval_step` | 1 | 1 | 1 |
| `stopping_step` | 3 | **5**（阶段一日志） | **3**（阶段二日志） |
| `RESUME_FROM` | — | None | `D:\recbole_checkpoints\movies_tv_seq\SASRec\best.pth` |
| `EVAL_ONLY` | — | False | False |
| `DEBUG` | — | False | False |

阶段二启动日志：`Checkpoint loaded. Resume training from epoch 9`（RecBole 计数器；权重为阶段一 checkpoint 内状态）。

### 2.4 平台评估参数（两阶段相同）

| 参数 | 值 |
|------|-----|
| `num_negatives_eval` | 100 → uni100 |
| `neg_sampling_dynamic` | false（对 SASRec 训练无效） |
| `neg_sampling_train` 等 | 不生效（序列 CE） |
| `eval_valid_user_cap` / `eval_test_user_cap` | 20000 / 20000 |
| `eval_sample_seed` | 42 |
| `eval_args.order` | TO |
| `eval_args.group_by` | user |
| `eval_args.split` | {'LS': 'valid_and_test'} |
| `eval_args.mode` | valid/test = uni100 |
| `metrics` | Hit, NDCG, MRR, Recall, GAUC |
| `topk` | [10] |
| `valid_metric` | NDCG@10 |
| `full_catalog_eval_enabled` | false |
| `full_catalog_eval_topk` | [10, 20, 50] |
| `full_catalog_eval_on` | test |
| `full_catalog_eval_user_cap` | 0 |

---

## 3. 阶段一训练过程

### 3.1 为何结束？

- **非早停**：best NDCG 在 ep8；ep9–10 未连续 5 轮不涨（仅 2 轮），`stopping_step=5` 未触发。
- **实际**：epoch 10 valid 评估完成后 **Ctrl+C**；平台加载 **best.pth（ep8）** 跑 uni100 test。
- 每 epoch：**10683 batch**，训练 ~**495 s**（~8.2 min）+ valid ~**14 s**。

### 3.2 阶段一逐 Epoch

| Epoch | 训练耗时 | train loss | NDCG@10 | HR@10 | MRR@10 | GAUC | NDCG/HR | 存 best |
|------:|---------:|-----------:|--------:|------:|-------:|-----:|--------:|:-------:|
| 0 | 493.2s | 106760.77 | 0.6450 | 0.8188 | 0.5900 | 0.9310 | 0.788 | 是 |
| 1 | 498.0s | 98413.62 | 0.6733 | 0.8398 | 0.6205 | 0.9393 | 0.802 | 是 |
| 2 | 493.5s | 95443.84 | 0.6825 | 0.8464 | 0.6304 | **0.9415** | 0.807 | 是 |
| 3 | 492.2s | 93627.64 | 0.6845 | **0.8466** | 0.6332 | 0.9411 | 0.809 | 是 |
| 4 | 500.5s | 92397.78 | 0.6846 | 0.8446 | 0.6339 | 0.9405 | 0.811 | 是 |
| 5 | 496.5s | 91512.26 | 0.6826 | 0.8419 | 0.6320 | 0.9400 | 0.811 | — |
| 6 | 499.5s | 90838.89 | 0.6857 | 0.8462 | 0.6347 | 0.9396 | 0.810 | 是 |
| 7 | 495.5s | 90312.82 | 0.6848 | 0.8444 | 0.6342 | 0.9393 | 0.811 | — |
| **8** | 493.3s | 89868.62 | **0.6870** | 0.8464 | **0.6364** | 0.9391 | **0.812** | **是** |
| 9 | 494.4s | 89502.12 | 0.6865 | 0.8447 | 0.6364 | 0.9387 | 0.813 | — |
| 10 | 493.7s | 89191.00 | 0.6850 | 0.8420 | 0.6352 | 0.9382 | 0.814 | — |

**阶段一累计（ep0–10）**：约 **91 min**（train+valid）；结束后 **Ctrl+C 触发 test**（best 仍为 ep8 权重）。

### 3.3 阶段一结束时的 test（best @ ep8）

来源：`movies_tv_seq_SASRec_full_20260604_003220.json`

| 集合 | HR@10 | NDCG@10 | MRR@10 | GAUC |
|------|-------|---------|--------|------|
| Valid（best ep8） | 0.8464 | 0.6870 | 0.6364 | 0.9391 |
| Test | 0.7986 | 0.6317 | 0.5789 | 0.9199 |

---

## 4. 阶段二训练过程（续训：降 lr + weight_decay）

### 4.1 动机与改动摘要

阶段一 valid 已平台化、test 低于 valid；续训时通过 `run_train.py` 覆盖：

- `RESUME_LR = 0.0005`（yaml 仍为 0.001，续训后写入 optimizer）  
- `RESUME_WEIGHT_DECAY = 1e-5`（阶段一为 0）  
- `EPOCHS = 25`，`stopping_step` 按 yaml **3**（阶段一日志为 5）

### 4.2 为何结束？

- **早停触发**：`Finished training, best eval result in epoch 12`  
- 自 ep12（NDCG **0.6874**）后，ep13–16 未再创新高，`stopping_step=3` 满足停止条件  
- 结束后自动加载 **best.pth（ep12）** 跑 test  

### 4.3 阶段二逐 Epoch（日志 epoch 9–16）

| Epoch | 训练耗时 | train loss | NDCG@10 | HR@10 | MRR@10 | GAUC | NDCG/HR | 存 best |
|------:|---------:|-----------:|--------:|------:|-------:|-----:|--------:|:-------:|
| 9 | 501.3s | 89354.39 | 0.6831 | 0.8390 | 0.6335 | 0.9350 | 0.814 | — |
| 10 | 504.9s | 90885.95 | 0.6855 | 0.8442 | 0.6351 | 0.9389 | 0.812 | — |
| 11 | 509.1s | 91429.22 | 0.6842 | 0.8426 | 0.6337 | 0.9402 | 0.812 | — |
| **12** | 509.0s | 91768.61 | **0.6874** | 0.8442 | **0.6375** | 0.9406 | **0.814** | **是** |
| 13 | 507.7s | 91978.11 | 0.6860 | 0.8420 | 0.6364 | 0.9405 | 0.815 | — |
| 14 | 506.2s | 92096.97 | 0.6854 | 0.8449 | 0.6347 | 0.9407 | 0.811 | — |
| 15 | 506.8s | 92137.30 | 0.6863 | 0.8440 | 0.6362 | 0.9414 | 0.814 | — |
| 16 | 521.6s | 92163.16 | 0.6854 | 0.8471 | 0.6342 | 0.9408 | 0.809 | — |

**阶段二累计（ep9–16）**：约 **68 min**（8 个 epoch）。

**说明**：续训后前几轮 **train loss 回升**（89354→91768）属正常——开启 `weight_decay` 后 CE 目标含 L2，与阶段一不可直接比绝对值；看 **valid/test** 即可。

---

## 5. 全训练汇总（阶段二结束后，全局 best @ epoch 12）

来源：`results/logs/movies_tv_seq_SASRec_full_20260604_125756.json`

| 集合 | HR@10 | NDCG@10 | MRR@10 | GAUC |
|------|-------|---------|--------|------|
| **Valid（全局 best，ep12）** | **0.8442** | **0.6874** | **0.6375** | 0.9406 |
| **Test** | **0.8038** | **0.6369** | **0.5842** | 0.9240 |
| Valid − Test Δ | −0.0404 | −0.0505 | −0.0533 | −0.0166 |

### 阶段一 best → 阶段二 best（test 对比）

| 指标 | 阶段一末（ep8）Test | 阶段二末（ep12）Test | Δ |
|------|---------------------|----------------------|---|
| NDCG@10 | 0.6317 | **0.6369** | **+0.0052** |
| HR@10 | 0.7986 | **0.8038** | **+0.0052** |
| MRR@10 | 0.5789 | **0.5842** | +0.0053 |
| GAUC | 0.9199 | **0.9240** | +0.0041 |

续训 **test 小幅提升**；valid NDCG 仅 +0.0004（0.6870→0.6874），收益有限但方向一致。

当前交付权重：`D:\recbole_checkpoints\movies_tv_seq\SASRec\best.pth`（**阶段二 ep12**，已覆盖阶段一 ep8 文件）。

---

## 6. 与其它模型对比（uni100，cap 20k）

| 模型 | Valid NDCG@10 | Test NDCG@10 | Valid HR@10 | Test HR@10 | 备注 |
|------|---------------|--------------|-------------|------------|------|
| BPR（best ep48） | 0.5846 | 0.5034 | 0.7822 | 0.7020 | 通用 BPR |
| LightGCN（best ep4） | 0.6159 | 0.5180 | 0.7688 | 0.6788 | 图 + BPR |
| SASRec 阶段一（ep8） | 0.6870 | 0.6317 | 0.8464 | 0.7986 | 序列 CE |
| **SASRec 全训练（ep12）** | **0.6874** | **0.6369** | **0.8442** | **0.8038** | **本报告主结果** |

---

## 7. 曲线解读与过拟合

### 阶段一

1. **train loss**：106761 → 89191，单调降。  
2. **Valid NDCG**：ep0–3 陡升，ep3–8 平台（**0.6870** @ ep8）。  
3. **HR/GAUC**：HR ep3 见顶；GAUC ep2 见顶 → **轻度过拟合**。  
4. ep9–10 valid 缓降 → Ctrl+C 合理。

### 阶段二

1. **Valid NDCG**：ep9 略低（0.6831）后回升，**ep12 全局最高 0.6874**；ep13–16 横盘略降 → 早停合理。  
2. **HR@10**：ep16 达 **0.8471** 但 NDCG 未涨 → 仍属「HR 与 NDCG 分叉」，勿单看 HR。  
3. **weight_decay**：未消除 valid–test gap（仍约 5 pt NDCG），但 **test 略优于阶段一**。  

**总结论**：两阶段合计约 **2.6 h** 训练；**交付以阶段二 ep12 为准**；不宜再在无改动下加长 epoch。

---

## 8. 资源与日志

| 项目 | 阶段一 | 阶段二 |
|------|--------|--------|
| GPU 显存 | ~2.33 G / 15.92 G | ~2.25 G / 15.92 G |
| 每 epoch batch | 10,683 | 10,683 |
| TensorBoard | `log_tensorboard/`（各 run 独立子目录） | 同左 |

---

## 9. 复现与补评

### 阶段二续训复现

```python
# run_train.py
MODEL = "sasrec"
RESUME_FROM = r"D:\recbole_checkpoints\movies_tv_seq\SASRec\best.pth"  # 需先有阶段一 ckpt；或从阶段一 ep8 备份再续
EPOCHS = 25
RESUME_LR = 0.0005
RESUME_WEIGHT_DECAY = 1e-5
EVAL_ONLY = False
```

### 仅评估（当前 best = 阶段二 ep12）

```python
MODEL = "sasrec"
EVAL_ONLY = True
RESUME_FROM = None
```

### 全库 HR（已完成，见 §11）

```bash
# scripts/eval_full_catalog_hr.py
MODEL = "sasrec"
FULL_CATALOG_USER_CAP = 0   # 本次为全量 test 用户
TOPK = [10, 20, 50, 100]
```

权重：本次补评使用 `D:\recbole_checkpoints\movies_tv\SASRec\best.pth`（由 `movies_tv_seq\SASRec\best.pth` 拷贝，与阶段二 ep12 一致）。

### 重建序列数据

```bash
python scripts/build_sequential_dataset.py --dataset movies_tv --max-len 50
```

---

## 11. 全库 HR@K（test，全量 657,203 用户）

**补评时间**：2026-06-04 16:22:57 — 16:24:12（约 **12 秒**）  
**日志**：`log/SASRec/SASRec-movies_tv-Jun-04-2026_16-22-57-6b07c4.log`  
**脚本**：`scripts/eval_full_catalog_hr.py`（`FULL_CATALOG_USER_CAP=0`，`TOPK=[10,20,50,100]`）

| 指标 | 全库 test | uni100 test（阶段二 ep12，`125756.json`） | Δ |
|------|----------:|------------------------------------------:|---:|
| **HR@10** | **0.0865** | 0.8038 | **−0.717** |
| HR@20 | 0.1201 | — | — |
| HR@50 | 0.1759 | — | — |
| HR@100 | 0.2282 | — | — |
| **NDCG@10** | **待重跑** | 0.6369 | — |
| **MRR@10** | **待重跑** | 0.5842 | — |
| **算术平均排名 meanrank** | **待重跑** | — | — |

> **全库指标说明**（均在 **~20 万 item** 真实排序后计算，与 uni100 不可直接比数值）：  
> - **HR@K**：真实 item 是否落在全库 Top-K（与 Hit@K 相同）。  
> - **NDCG@K**：看命中且在 Top-K 内的**位置**，越靠前越高（比 HR 更细）。  
> - **MRR@K**（Mean Reciprocal Rank）：每个用户取 Top-K 内**第一个**相关 item 的排名 \(r\)，贡献 \(1/r\)，再对用户平均；**越大越好**。未进 Top-K 则该用户贡献 0。  
> - **meanrank**：每个用户真实 item 的**绝对名次**（1-based）的算术平均；**越小越好**。  
> 本次补评（16:22）仅有 HR@K；请用更新后的 `eval_full_catalog_hr.py` **再跑一遍**，会同时输出 NDCG/MRR/meanrank。

**解读**：

1. 在 **~20 万 item** 全库上，SASRec 的 HR@10 从 uni100 的 **0.80 → 0.09**，说明 **uni100 严重高估** 全库排序能力；数据集 **并不简单**，只是小候选集里好刷分。  
2. 与 Pop uni100（HR@10≈0.56）对比：Pop 在 uni100 上「看起来不错」，但若也跑全库 HR，预期会 **进一步低于** 0.56（Pop 全库补评未跑完，见 Pop 报告）。  
3. 写论文时：**主表用 uni100（与 BPR/LightGCN 对齐）**；全库 HR 单独一小节作 **协议敏感性 / 难度说明**。

### 运行概况

```text
Full-catalog eval [test]: rows=657203, items=197944, HR@[10,20,50,100], users=全量用户
Evaluate: 321/321 [00:12<00:00, 26.52it/s, GPU RAM: 10.59 G/15.92 G]
```

### 跨模型对比：数量到底统不统一？

**统一的是「被算进 HR 的用户/样本数」，不统一的是「进度条上有多少步」**——后者不影响指标定义，只影响耗时。

| 项目 | SASRec（本次） | Pop（同脚本、同 cap=0） |
|------|----------------|------------------------|
| **评估样本数 `n_samples`** | **657,203** | **657,203**（日志 `rows=` 同一含义） |
| **全库 item 数** | 197,944 | 197,944 |
| **HR@K 含义** | 每用户全库排序后 Hit | 相同 |
| tqdm 总步数 `n_batches` | **321** | **657,203** |
| 原因 | 序列模型：约 `ceil(657203/2048)` 批 | 通用模型：`eval_batch_size < item_num` → RecBole **强制每批 1 用户** |

因此：**不是「SASRec 只评了 321 个用户」**——321 是 **batch 数**，每批约两千用户，合起来仍是 65.7 万人。  
若某模型全库 **没跑完**（Pop 中断）或 **cap 不一致**（一个 2 万、一个 65 万），那才真的「搞笑」、不能比；本次 SASRec 与 Pop **启动时**协议一致，Pop 只是未完成。

平台已改 `run_full_catalog_hr_eval` / `run_train` 打印：`评估样本数=... | batches=...（仅影响耗时）`，避免再误读 tqdm。

### 为什么 SASRec 全库只要 ~12 秒，Pop 却要几小时？

**不是 SASRec 少算了用户**，而是 **进度条单位与实现完全不同**：

| 对比项 | SASRec（本次） | Pop（同脚本、同 65.7 万用户） |
|--------|----------------|-------------------------------|
| 进度条总数 | **321** 步 | **657,203** 步 |
| 含义 | **321 个 batch**，每 batch 约 **~2048 用户** | 约 **每用户 1 步** |
| 每步计算 | Transformer **批量** `full_sort_predict`：一次为整批用户对 **全部 item** 打分（GPU 矩阵运算） | 对每个用户 **重复** 整条 `item_cnt`（~20 万维），`repeat_interleave` 展开，**步数 = 用户数** |
| 显存 | ~10.6 G / 15.9 G | 易顶满（日志曾 **16.2 G > 15.9 G**），越跑越慢 |
| 实测耗时 | **~12 s** 跑完 65.7 万用户 | 10+ 分钟仅 ~35%，ETA 数小时 |

因此：

- Pop 慢 **不是因为「Pop 算法更复杂」**，而是 RecBole 对 **Pop 的全库评估按用户细粒度迭代**，且 **显存压力大**。  
- SASRec 快 **不是因为任务更小**，而是 **按大 batch 做全库打分**，321 次迭代就覆盖全部 test 用户。  
- 两者 **评估的用户数相同（657,203）**；差异在 **batch 与实现**，不是 tqdm 算错比例。

若要让 Pop 也在合理时间内跑完全库：在脚本里设 `FULL_CATALOG_USER_CAP = 20000`（与 uni100 同批用户），仍会比 SASRec 慢，但可到 **十几分钟级** 而非数小时。

---

## 10. 备注

- 首次 SASRec 跑 flat `movies_tv` 会因缺少 `item_id_list` 报错；已用 `movies_tv_seq` 修复。  
- yaml 中 `checkpoint_dir: .../movies_tv/SASRec` 会被 `run_train.py` 重写为 `.../movies_tv_seq/SASRec`。  
- 首次失败日志：`SASRec-movies_tv-Jun-04-2026_00-26-48`。  
- 阶段一完整 RecBole 超参：`00-32-20-de2339.log` 第 1–110 行；阶段二：`12-57-56-217474.log` 第 1–110 行。  
- 根目录 `SASrec_meta_0.79.json` 为阶段一中断后 test 的 meta 快照（test HR≈0.7986）；**最终以 `125756.json` / ep12 为准**。
