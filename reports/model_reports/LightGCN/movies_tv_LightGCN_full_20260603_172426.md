# 训练报告：movies_tv · LightGCN（动态难负例，Ctrl+C 中断）

| 项目 | 内容 |
|------|------|
| 运行标签 | `movies_tv_LightGCN_full_20260603_172426` |
| 时间 | 2026-06-03 17:24:26 — 19:10+（epoch 4 约 11% 时中断） |
| 日志 | `log/LightGCN/LightGCN-movies_tv-Jun-03-2026_17-24-26-349d9e.log` |
| 配置 | `configs/lightgcn_movies_tv_full.yaml` |
| Epoch 完成 | **0 — 3**（epoch 4 训练中断，未完成 eval） |
| 结束原因 | **KeyboardInterrupt**（无 test、无 JSON） |
| Best 权重 | `D:\recbole_checkpoints\movies_tv\LightGCN\best.pth`（**epoch 2**） |

---

## 训练配置要点

| 参数 | 值 |
|------|-----|
| train 负例 | `sample_num=4`，`dynamic=True`，`candidate_num=100` |
| eval | uni100，valid/test 各 cap **10000** 用户 |
| embedding / layers | 64 / 3 |
| reg_weight | 1e-5，`weight_decay=0`（避免双重正则） |
| 每 epoch 训练 | ~**26 分钟**（5984 batch，约为旧配置 4×） |
| 目标 epochs | 50 |

---

## 训练为何结束？

- **非早停**：`stopping_step=5`，valid 在 ep2 后未连续 5 轮不涨。
- **实际**：手动 Ctrl+C；中断前 **ep3 valid NDCG 已略降**，曲线有早期平台迹象。
- **Test**：本次未跑；best 权重在磁盘上，可用 `EVAL_ONLY` 补评（见文末）。

---

## Best Valid（epoch 2，按 NDCG@10）

| 指标 | 值 | NDCG÷HR |
|------|-----|---------|
| HR@10 | 0.6996 | — |
| **NDCG@10** | **0.5610** | **0.802** |
| MRR@10 | 0.5171 | — |

---

## 逐 Epoch（0 — 3）

| Epoch | 训练耗时 | train loss | NDCG@10 | HR@10 | MRR@10 | NDCG/HR | 存 best |
|------:|---------:|-----------:|--------:|------:|-------:|--------:|:-------:|
| 0 | 1543.6s | 3786.75 | 0.5529 | 0.6976 | 0.5071 | 0.793 | 是 |
| 1 | 1533.7s | 3452.81 | 0.5608 | 0.7030 | 0.5156 | 0.798 | 是 |
| 2 | 1572.9s | 3184.13 | **0.5610** | 0.6996 | 0.5171 | **0.802** | **是** |
| 3 | 1574.0s | 2929.49 | 0.5588 | 0.7001 | 0.5142 | 0.798 | — |
| 4 | （中断） | — | — | — | — | — | — |

---

## 与历史运行对比

| 运行 | 配置 | Best Valid NDCG | Best Valid HR | 备注 |
|------|------|-----------------|---------------|------|
| **本轮** | 4 负 + 动态难负 | **0.5610** (ep2) | 0.6996 | 慢；ep3 回落 |
| 6/3 凌晨 | 1 负、无动态 | 0.5460 (ep7) | 0.7561 | 仅训 8 epoch 中断 |
| 6/3 凌晨 ep0 | 1 负 | 0.4823 | 0.6975 | 同 HR、低 NDCG |
| BPR 全训练 | BPR 50ep | **0.5846** | **0.7822** | 仍高于本轮 |

### 怎么看「情况不是很好」

1. **NDCG/HR 比值高**（~0.80）：正样本在 top-10 里更靠前，排序质量好。  
2. **HR@10 几乎不涨**（~0.70）：相对 BPR（0.78）和旧 LightGCN ep7（0.76）偏低。  
3. **ep2 后平台/回落**：ep3 NDCG 0.5588 < ep2 0.5610，继续训性价比存疑。  
4. **成本极高**：单 epoch ~26 分钟，50 epoch 理论 **20+h**，中断前已约 **1.5h** 仅 4 轮。  
5. **无 test**：无法和 BPR test NDCG 0.5034 直接比。

**结论**：动态难负例提升了早期 NDCG 与 NDCG/HR，但 **HR 停滞 + 第 3 轮回落 + 极慢**，作为正式跑法性价比一般；建议补 test 后再决定是否降 `dynamic`/`sample_num` 或改 lr。

---

## 评估协议（uni100 vs 全库 HR）

| 阶段 | 用户范围 | Item 范围 | 用途 |
|------|----------|-----------|------|
| 训练期 valid/test | `eval_*_user_cap: 10000`（本 run） | uni100：1 正 + 100 负 | 快速调参、选 best |
| 补评 uni100 test | 同上 1 万用户 | uni100 | 与 BPR/NeuMF 报告对齐 |
| **全库 HR@K** | `full_catalog_eval_user_cap: 0` → **657203** test 用户 | **全部 ~197944 item** | 真实排序难度，报告主指标 |

平台已修复（2026-06-03 晚）：

1. **topk 与 collector 不一致** → 全库阶段重建 `eval_collector`，`topk` 仅用 `[10,20,50]`，避免 `split_with_sizes 11 vs [50,1]`。  
2. **全库仍只有 1 万用户** → cap 前备份 `_benchmark_uncapped`，全库评估临时换回完整 test。  

相关日志：

| 日志 | 全库 test 用户数 | 状态 |
|------|------------------|------|
| `LightGCN-movies_tv-Jun-03-2026_22-34-21` | 10000 | 曾触发 topk 维度错误 |
| `LightGCN-movies_tv-Jun-03-2026_22-41-28` | **657203** | 补丁后已启动全量全库评估 |

全库 HR 结果（best.pth @ epoch 2）**待写入**（657k×20 万 item 极慢，跑完后把终端 `hit@*` / `hr@*` 填到下方）：

| 集合 | HR@10 | HR@20 | HR@50 | 备注 |
|------|-------|-------|-------|------|
| Test 全库 | — | — | — | `python scripts/eval_full_catalog_hr.py` |

yaml 关键项（`configs/lightgcn_movies_tv_full.yaml`）：

```yaml
eval_test_user_cap: 10000              # 仅 uni100
full_catalog_eval_user_cap: 0          # 全库 = 全部 test 用户
full_catalog_eval_topk: [10, 20, 50]
```

---

## 补跑 test（本次中断后）

**方式 A — 仅评估（推荐）**

```bash
cd recbole_platform
python scripts/eval_full_catalog_hr.py   # MODEL=lightgcn，全库 HR
```

**方式 B — `run_train.py`**

```python
MODEL = "lightgcn"
EVAL_ONLY = True
RESUME_FROM = None   # D:\recbole_checkpoints\movies_tv\LightGCN\best.pth
```

会跑 uni100 test（1 万用户）+ 全库 HR（657203 用户，耗时很长）。

**以后**：训练时 Ctrl+C 也会先跑 uni100 test，再尝试全库 HR（`run_train.py` 已支持）。

---

## 训练曲线：如何在日志里看出「loss 降、指标升」

### RecBole 默认会打什么

每轮训练结束一行（**本 run 可直接 grep**）：

```text
epoch k training [time: ..., train loss: XXXX]
valid result: hit@10 ... ndcg@10 ...
```

- **train loss**：当前实现里是 **整 epoch 各 batch loss 之和**（LightGCN 常 3000→2900），不是「平均每样本 0.x」；**只看单调下降与斜率**，不要和 BPR 的 ~485→10 数值横比。  
- **valid**：以 `valid_metric: NDCG@10` 选 best；日志里还有 `epoch k evaluating [time: ..., valid_score: 0.xxxx]`（与 NDCG@10 一致）。

本 run 曲线特征（epoch 0→3）：

| 信号 | 表现 | 解读 |
|------|------|------|
| train loss | 3787 → 3453 → 3184 → 2929 | 稳定下降，优化在进行 |
| NDCG@10 | 0.553 → 0.561 → **0.561** → 0.559 | ep2 平台，ep3 **回落** |
| HR@10 | ~0.70 横盘 | 召回侧几乎不涨 |
| NDCG/HR | 0.793 → **0.802** | 排序质量变好，但 HR 卡住 |

**「良好曲线」在本项目里指**：train loss 缓降 + **NDCG@10 连续 3–5 轮上升** + HR@10 同步抬升；本 run 属于 **loss 仍降、NDCG 早平台**，不宜再加 epoch。

### TensorBoard（RecBole 内置，无需改代码）

每次训练会在 `recbole_platform/log_tensorboard/` 下按日志名建目录，写入：

- `Loss/Train`（按 epoch）  
- `Vaild_score`（valid 主指标，即 NDCG@10）

查看：

```bash
tensorboard --logdir log_tensorboard
```

浏览器看 loss 与 valid 是否同向；若 loss 降而 valid 平/跌，即过拟合或难负例过强。

### 可选：W&B

yaml / config 设 `log_wandb: True` 并 `pip install wandb`，可在云端对比多次 run（需自行设 `wandb_project`）。

### 建议写入实验笔记的 4 列

与 BPR 报告一致，每 epoch 记：**train 耗时 | train loss | NDCG@10 | HR@10 | 是否存 best**。中断 run 也可事后从 `.log` 拼表（本报告「逐 Epoch」表即如此）。

---

## LightGCN 调优建议（针对本 run 现象）

| 问题 | 建议 | 理由 |
|------|------|------|
| 单 epoch ~26 min | `neg_sampling_dynamic: false`，`sample_num: 1` 或 `4` 固定负例 | 动态难负 + candidate_num=100 是主因；凌晨 1 负 run ~4 min/epoch |
| ep2 后 NDCG 回落 | `stopping_step: 3`，观察 ep1–3；或 **以 ep1/ep2 best 直接 EVAL_ONLY** | 已出现 valid 回落，继续训性价比低 |
| HR@10 低于 BPR | 试 `embedding_size: 128`（见 `19-28-32` run：ep1 NDCG **0.6036**） | 同动态负例下 NDCG 更高，但 ep2 仍跌，需关动态再比 |
| 与 BPR 差一截 | `n_layers: 2` 或 `3` 网格；`learning_rate: 0.001` → `5e-4` 续训 | 图卷积过深易过平滑；略降 lr 有时稳 valid |
| 指标「看起来不好」 | **分开看** uni100（调参）与全库 HR（交付） | uni100 HR~0.70 不等于全库；全库 HR 通常明显更低 |
| 正式对比 BPR | 固定 seed、`eval_sample_seed`；全库 HR@10/20/50 与 BPR test 全库同协议 | 避免只用 1 万用户 cap 得出「虚高/不可比」结论 |

**推荐下一组实验（二选一主线）**

1. **快迭代**：`dynamic=false`，`sample_num=4`，`embedding_size=128`，`epochs=20`，保留 cap=10000 → 看 NDCG/HR 是否连续上升。  
2. **追 BPR**：在 (1) 的 best 上 `EVAL_ONLY` + 全库 HR；若全库 HR@10 仍远低于 BPR，再调 `n_layers` / lr。

同日晚间另一次运行（对照，非本报告主 run）：

| 日志 | embedding | ep1 NDCG | 说明 |
|------|-----------|----------|------|
| `19-28-32` | 128 | **0.6036** | 动态负 candidate_num=50，ep2 起回落，Ctrl+C @ ep4 |
| **本报告 `17-24-26`** | 64 | 0.5608 | candidate_num=100，更慢，best @ ep2 |

---

## 备注

- 上一轮失败日志：`LightGCN-movies_tv-Jun-03-2026_17-13-03`（补丁类名错误，未开训）。  
- 全库评估：**657203 用户 × 197944 item**，请预留数小时至数天（视 GPU）；可先 `full_catalog_eval_user_cap: 50000` 做抽样冒烟。  
- 完整超参块见 `17-24-26-349d9e.log` 前 ~120 行；TensorBoard 事件与日志同名目录在 `log_tensorboard/`。
