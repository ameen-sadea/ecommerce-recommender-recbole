# 训练报告：movies_tv · Pop（非个性化基线）

| 项目 | 值 |
|------|-----|
| 运行标签 | `movies_tv_Pop_full_20260604_142802` |
| 时间 | 2026-06-04 14:28:02 — 14:28:40（约 **38 秒**） |
| 原始日志 | `log/Pop/Pop-movies_tv-Jun-04-2026_14-28-02-ca8047.log` |
| 指标 JSON | `results/logs/movies_tv_Pop_full_20260604_142802.json` |
| 权重 | `D:\recbole_checkpoints\movies_tv\Pop\best.pth` |
| 全库补评 | `movies_tv_Pop_full_catalog_20260605.json`（2026-06-05） |

配置文件：`configs/pop_movies_tv_full.yaml`

---

## 1. 为什么只有 1 个 epoch？（不是“没训够”）

Pop 在 RecBole 里属于 **ModelType.TRADITIONAL**：算法是「统计 train 里每个 item 被交互的次数，对所有用户推荐同一套热门排序」，**没有可学习的 embedding、也没有梯度更新**。

| 现象 | 含义 |
|------|------|
| `epochs: 1` | 只需 **遍历一遍 train** 把 `item_cnt` 数完；多跑 epoch 不会改变排序（除非 shuffle 导致计数顺序不同，Pop 实现为累加，结果相同） |
| `train loss: 0` | `calculate_loss` 为占位，不参与优化 |
| `Trainable parameters: 1` | 仅为占位参数，不是深度模型 |

因此 **1 epoch 是标准做法**，与 BPR/SASRec 的 20–50 epoch **不可直接对比“训练轮数”**；对比时应写清：**Pop = 统计基线，深度模型 = 迭代优化**。

---

## 2. 指标汇总（uni100，cap 2 万用户）

评估：`order=TO`，`uni100`（1 正 + 100 负），`eval_valid_user_cap` / `eval_test_user_cap` = **20000**，`seed=42`。

| 集合 | HR@10 | NDCG@10 | MRR@10 | GAUC |
|------|-------|---------|--------|------|
| **Valid** | 0.6005 | **0.4067** | 0.3467 | 0.8309 |
| **Test** | 0.5642 | **0.3808** | 0.3241 | 0.8093 |

Valid − Test（NDCG）：约 **0.026**，略优于多数深度模型在 test 上的落差，但 **绝对值仍明显低于** BPR / LightGCN / SASRec。

---

## 3. 与同数据集其它模型对比（Test，uni100）

| 模型 | 类型 | Test NDCG@10 | Test HR@10 | 相对 Pop (NDCG) |
|------|------|-------------:|-----------:|----------------:|
| **Pop** | 非个性化 | **0.3808** | 0.5642 | — |
| ItemKNN | 传统 CF | *见 ItemKNN 报告* | — | — |
| BPR | 浅层 MF | 0.5034 | 0.7020 | **+32%** |
| LightGCN | 图模型 | 0.5180 | 0.6790 | **+36%** |
| SASRec（阶段二） | 序列深度 | **0.6369** | **0.8038** | **+67%** |

**结论（写进论文/作业用）**：Pop 的 HR@10≈0.56 **看起来不低**，是因为：

1. **评估协议是 uni100**：候选只有 101 个，且含 1 个真实 next-item；Amazon Movies & TV 上 **头部 item 极热门**，Pop 在「小候选集」里很容易排进 Top-10。  
2. **GAUC 很高（≈0.81）**：在「正例 vs 99 个随机负例」下，热门 item 对随机负例区分度大，**不等于**全库排序能力强。  
3. **与个性化模型仍差一截**：SASRec Test NDCG 比 Pop 高约 **0.26**；说明数据 **并非**「Pop 就够、深度学习白做」。  
4. 全库已补评（§8）：Test HR@10 **0.017** vs uni100 **0.564**，可直接写「uni100 高估非个性化基线」。

---

## 4. 训练过程（Epoch 0）

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 数据加载 + cap | ~17 s | 657204 users，197944 items，稀疏度 99.995% |
| epoch 0 train | 6.83 s | 扫一遍 train 统计热度 |
| epoch 0 valid | 4.27 s | NDCG@10 = 0.4067 → 写入 `best.pth` |
| test | （在 `_finalize` 中） | 见 JSON |

---

## 5. 超参

| 参数 | 值 |
|------|-----|
| epochs | 1 |
| stopping_step | 1 |
| train_batch_size / eval_batch_size | 8192 |
| num_negatives_eval | 100 → uni100 |
| eval_*_user_cap | 20000 |
| full_catalog_eval | 默认关；全库用 `eval_full_catalog_hr.py` + **分块**（见下） |

---

## 6. 复现

```python
# run_train.py
MODEL = "pop"
DEBUG = False
RESUME_FROM = None
EVAL_ONLY = False
```

```powershell
cd recbole_platform
python run_train.py
```

仅补 test / 全库 HR：Pop 有 `best.pth`，可 `EVAL_ONLY = True`。

---

## 7. 报告撰写建议（回应「数据集是否太简单」）

- **不要把 Pop 与 SASRec 并列成「同一量级 SOTA」**；表格中单独一行标注 **Non-personalized baseline**。  
- 强调：**Movies_and_TV 规模大（65 万用户、20 万 item、极稀疏）**；Pop 分数来自 **评估设定 + 长尾流行度**，不是「任务 trivial」。  
- 全库 Pop：`eval_full_catalog_hr.py` 设 `MODEL=pop`、`EVAL_CHUNK_USERS=50000`（全量）、`COMPUTE_MEANRANK=False`；**默认 GPU**，无需强制 CPU。

### Pop 全库爆显存怎么办？

| 办法 | 说明 |
|------|------|
| **分块用户（推荐）** | `EVAL_CHUNK_USERS=50000`（脚本已默认）；块间释放 collector，指标加权合并 |
| **已内置：float32 + expand** | Pop `full_sort_predict` 补丁，减轻每步显存 |
| **collector 指标落 CPU** | 只把 hit/ndcg 等累计张量放 CPU，**模型仍在 GPU** |
| **关 meanrank** | `COMPUTE_MEANRANK=False`，避免每步对 20 万 item `sort` |
| **用户 cap** | `FULL_CATALOG_USER_CAP=20000` 与 uni100 对齐，写报告够用 |
| 勿用 | ~~全库强制 CPU~~（已移除；GPU 慢但可配合分块跑全量） |

---

## 8. 全库 Test 补评（2026-06-05）

与 §2 uni100 **不可比**；用于说明「小候选集下 Pop 虚高、全库极难」。

### 8.1 协议与入口

| 项 | 值 |
|----|-----|
| 脚本 | `scripts/eval_full_catalog_hr.py` |
| 权重 | 同上 `best.pth`（epoch 0 统计热度） |
| Test 用户 | **657,203**（全量，非 cap 2 万） |
| 候选 | **~197,944 item** 全排序 |
| 指标 | HR@10/50、NDCG@10/50、MRR@10/50；**未开** meanrank |
| 分块 | `EVAL_CHUNK_USERS=50000` → **14 块**（13×5 万 + 7,203） |
| 设备 | **GPU**（collector 指标落 CPU；模型打分在 CUDA） |
| 日志 | `log/Pop/Pop-movies_tv-Jun-05-2026_08-52-18-e5e1ad.log` |

```python
# eval_full_catalog_hr.py 当时设置
MODEL = "pop"
ON = "test"
FULL_CATALOG_USER_CAP = 0
EVAL_CHUNK_USERS = 50000
COMPUTE_MEANRANK = False
```

### 8.2 测试过程（分块轨迹）

启动：**2026-06-05 08:52**。此前修复 collector **cuda/cpu 混 cat** 后首次跑通全量。

```text
+------+----------+----------+---------------------------+
| 块号 | 用户数   | 步数     | 单块耗时（约）            |
+------+----------+----------+---------------------------+
| 1    | 50000    | 50000    | 56 s  (~883 it/s)         |
| 2    | 50000    | 50000    | 57 s                      |
| 3    | 50000    | 50000    | 56 s                      |
| 4    | 50000    | 50000    | 56 s                      |
| 5    | 50000    | 50000    | 56 s                      |
| 6    | 50000    | 50000    | 56 s                      |
| 7    | 50000    | 50000    | 56 s                      |
| 8    | 50000    | 50000    | 56 s                      |
| 9    | 50000    | 50000    | 57 s                      |
| 10   | 50000    | 50000    | 56 s                      |
| 11   | 50000    | 50000    | 56 s                      |
| 12   | 50000    | 50000    | 57 s                      |
| 13   | 50000    | 50000    | 56 s                      |
| 14   | 7203     | 7203     | 7 s   (~982 it/s)         |
+------+----------+----------+---------------------------+
| 合计 | 657203   | 657203   | **约 13 分钟**            |
```

- Pop 全库为 **每用户 1 batch**（非 SASRec 的数百批）；进度条总数 = 该块用户数。  
- 全程 **GPU RAM 峰值 ~0.02 G/15.92 G**（热度向量小 + 分块释放 collector）。  
- 指标按块 **用户数加权合并**，与一次跑完全量等价。

### 8.3 全库 Test 结果

```text
+----------------+-----------+-----------+-----------+-----------+
| 指标           | @10       | @50       | 备注      |
+----------------+-----------+-----------+-----------+-----------+
| HR@10 / HR@50  | 0.0171    | 0.0414    | 全库      |
| NDCG@10/50     | 0.0087    | 0.0138    | 全库      |
| MRR@10/50      | 0.0060    | 0.0070    | 全库      |
+----------------+-----------+-----------+-----------+-----------+
```

精确值见 `results/logs/movies_tv_Pop_full_catalog_20260605.json`。

### 8.4 uni100 vs 全库（同模型 Pop）

```text
+----------+-------------+-------------+---------------------------+
| 指标     | uni100 Test | 全库 Test   | 说明                      |
+----------+-------------+-------------+---------------------------+
| HR@10    | 0.5642      | 0.0171      | uni100 约 **33×** 全库 HR |
| NDCG@10  | 0.3808      | 0.0087      | uni100 约 **44×** 全库    |
| MRR@10   | 0.3241      | 0.0060      | 全库几乎随机级            |
| 用户数   | 20000       | 657203      | 全库已跑完                |
| 耗时     | 秒级        | ~13 min GPU | 14 块分块                 |
+----------+-------------+-------------+---------------------------+
```

**写作要点**：Pop 在 uni100 上 HR≈0.56 只因 **101 候选 + 头部热门**；全库 HR@10≈**0.017** 才反映「在 20 万 item 里找 next-item」的真实难度。与 SASRec 全库 HR@10≈0.087 对比，Pop 仍远低于个性化序列模型。
