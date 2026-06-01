# 生成式推荐技术演进路径分析报告

> 基于知乎话题讨论（126条回答）和学术论文分析
> 数据来源: zhihu-generative-rec-analysis.md + 知乎用户回答原始数据

---

## 一、技术演进总览

### 1.1 生成式推荐发展时间线

```
2023年         2024年2月        2025年         2025下半年-2026
  |               |               |                  |
Google          Meta           各厂商密集发布论文        |
TIGER           GR/HSTU        快手OneRec              业界态度
(语义ID)        (开创者)       字节RankMixer           从狂热转向
               验证Scaling     美团MTGR                 理性观望
                 Law           腾讯OnePiece
                              小红书GenRank
```

### 1.2 技术演进四大阶段

| 阶段 | 时间 | 核心特征 | 代表工作 |
|------|------|---------|---------|
| **探索期** | 2023-2024.2 | 学术研究，语义ID+生成式检索 | Google TIGER |
| **爆发期** | 2024.2-2025.6 | 工业落地，HSTU统一序列建模，Scaling Law验证 | Meta GR, 快手OneRec |
| **分歧期** | 2025.6-2026.2 | 技术路线分化，生成式 vs 判别式Scaling | 字节UniMixer, 百度GRAB |
| **深耕期** | 2026.2-至今 | Ultra-Long Sequence Scaling，新架构探索 | ULTRA-HSTU, TokenMixer-Large, UniMixer |

---

## 二、技术路径分类

### 2.1 路径一：生成式GR范式（Generative Retrieval）

**核心理念**: 将推荐问题建模为"序列生成"任务，模型直接生成下一个Item ID

**技术架构**:
```
用户行为序列 → [Encoder] → [Decoder: 自回归生成] → Semantic ID/Item ID
```

**代表工作**:

| 厂商 | 论文 | 年份 | 核心创新 |
|------|------|------|---------|
| Meta | GR/HSTU | 2024 | 统一序列建模, HSTU结构, Scaling Law验证 |
| 快手 | OneRec | 2025.02 | 端到端生成, Semantic ID, RL偏好对齐 |
| 美团 | MTGR | 2025.08 | 保留DLRM交叉特征, Group-LN |
| 腾讯 | OnePiece | 2025.12 | 统一Encoder-Decoder, 上下文工程 |
| Google | TIGER | 2023 | RQ-VAE语义ID, 生成式检索先驱 |
| 小红书 | GenRank | 2025 | 生成式重排 |

**实验效果对比**:

| 工作 | 数据集 | 主要指标 | 提升幅度 |
|------|--------|---------|---------|
| Meta GR | 工业数据集 | 生产端12.4%提升 | 显著 |
| 快手OneRec | 快手主站 | Watch Time **+1.6%** | 较小但真实 |
| 美团MTGR | 美团广告 | 多个业务指标提升 | 待明确 |
| 腾讯OnePiece | Shopee搜索 | 显著效果(未公开具体数字) | - |

**技术特点**:
- ✅ 统一建模: 召回+排序统一到单一模型
- ✅ 端到端: 消除级联误差传播
- ✅ Scaling Law: 模型规模增大效果可预测提升
- ❌ 延迟高: 自回归生成Decoder成本高
- ❌ Ground Truth模糊: Next Item定义不唯一
- ❌ 广告落地难: 出价/计费逻辑难以融合

---

### 2.2 路径二：判别式Scaling范式（Discriminative Scaling）

**核心理念**: 保留判别式架构，通过增大模型规模和数据实现Scaling Law

**技术架构**:
```
用户特征 + Item特征 → [Transformer/TokenMixer] → [判别式输出: CTR/CVR]
```

**代表工作**:

| 厂商 | 论文 | 年份 | 核心创新 |
|------|------|------|---------|
| 字节/抖音 | RankMixer | 2025.07 | Token Mixing, MoE稀疏激活 |
| 字节/抖音 | STCA | 2025.11 | stacked target-to-history cross attention, request-level batching |
| 字节/抖音 | LEMUR | 2025.11 | 多模态端到端训练, memory bank控制长序列编码成本 |
| 字节/抖音 | TRM | 2026.01 | "Farewell to Item IDs", semantic token破解scaling瓶颈 |
| 字节/抖音 | TokenMixer-Large | 2026.02 | 7B在线/15B离线, mixing-and-reverting, Sparse Per-token MoE |
| 字节/抖音 | MSN | 2026.02 | Memory-based sparse activation, 检索式注入个性化参数 |
| 字节/抖音 | MixFormer | 2026.02 | sequence/dense co-scaling, 统一主干协同扩展 |
| 字节/抖音 | UG-Separation | 2026.02 | user/item信息流显式拆分, dense模型复用能力 |
| 字节/抖音 | MERGE | 2026.01 | 动态层次化索引, 流式数据cluster动态生成/重置/合并 |

**字节技术链解读**:
- **主干扩展**: RankMixer → TokenMixer-Large → UG-Separation
- **长序列工业化**: STCA (低复杂度高复用) + LEMUR (多模态内生)
- **统一主干**: OneTrans → MixFormer → MDL (特征/场景/任务统一建模)
- **输入表征重写**: TRM (semantic token) + MERGE (动态索引)

**核心演进规律**: 从"单个大backbone如何扩展" → "统一建模下保持工业部署可行性" → "输入语言与索引层的系统性重写"

**实验效果对比**:

| 工作 | 数据集 | 主要指标 | 提升幅度 |
|------|--------|---------|---------|
| 字节RankMixer | 抖音主Feed | 多个业务场景 | 显著 |
| 快手UniMixer | 快手主站 | - | 取代OneRec |

**技术特点**:
- ✅ 延迟可控: 保留判别式打分机制
- ✅ 可控性强: 传统调参方法有效
- ✅ MFU高: 算子优化针对Transformer
- ❌ 范式未变: 仍是"匹配"非"生成"
- ❌ 依赖候选集: 无法超越候选池上限

---

### 2.3 路径三：语义ID生成式检索（Semantic ID based Retrieval）

**核心理念**: 通过量化将Item映射为离散的Semantic ID，实现类似自然语言的生成式检索

**技术架构**:
```
Item内容特征 → [RQ-VAE量化] → Semantic ID序列
用户行为 → [Encoder-Decoder] → 生成下一个Item的Semantic ID → 查表得到Item
```

**代表工作**:

| 厂商 | 论文 | 年份 | 核心创新 |
|------|------|------|---------|
| Google | TIGER | 2023 | RQ-VAE, 语义ID生成检索 |
| 快手 | OneRec | 2025 | 分层语义ID, Hierarchical |
| 腾讯 | OnePiece | 2025 | 协作语义ID, 统一tokenizer |

**技术特点**:
- ✅ 语义聚合: 相近物品共享前缀，解决冷启动
- ✅ 生成可控: ID空间有限，输出可枚举
- ❌ 碰撞风险: RQ-VAE量化存在冲突可能
- ❌ 精度损失: 连续embedding→离散ID有信息损失
- ❌ 依赖内容特征: 无文本描述的物品难以处理

---

### 2.4 路径四：LLM赋能推荐（LLM-augmented Recommendation）

**核心理念**: 不改变推荐范式，用LLM增强特征表示或用户理解

**代表工作**:

| 厂商 | 论文 | 年份 | 核心创新 |
|------|------|------|---------|
| 字节 | HLLM | 2024 | 分层LLM: Item LLM + User LLM |
| 阿里 | LUM | 2024-2025 | 长序列用户建模 |
| 京东 | xLLM | 2025.10 | 推理引擎优化 |
| - | LLM推荐Scaling Law | 2026.2 | 首个LLM推荐Scaling Law, 合成数据 |

**技术特点**:
- ✅ 工程可行: 不改变现有架构，渐进式引入
- ✅ 世界知识: LLM预训练知识注入推荐
- ✅ 冷启动改善: 文本语义缓解ID稀疏问题
- ❌ 延迟增加: LLM推理耗时长
- ❌ 收益有限: 作为特征增强而非范式改变
- ⭐ 新突破: 2026年首次验证LLM推荐的Scaling Law(arXiv:2602.07298)

---

## 三、各厂商技术演进路径分析

### 3.1 Meta（开创者）

**技术路线**: GR/HSTU → ULTRA-HSTU → 持续深耕

**演进路径**:
```
2024.2: GR论文发布，HSTU结构
  → 验证Scaling Law存在 (12.4%效果提升)
  → 成为工业界标杆工作
  → 后续多厂Follow HSTU架构

2026.2: ULTRA-HSTU发布
  → 18层16k序列处理能力
  → 5x训练+21x推理加速
  → 消费+4%, 互动+8%
  → AT/MoT动态拓扑 + 混合精度优化
  → 内存优化: 7GB→2.3GB/层
```

**核心贡献**:
1. 首次在工业级推荐系统验证Scaling Law
2. 提出HSTU替代标准Transformer
3. 统一序列建模框架
4. ULTRA-HSTU: 系统-模型协同设计，弯曲Scaling曲线

**后续影响**: 美团MTGR、小红书HSTU落地均基于此，VLM数据基础设施支撑超长序列

---

### 3.2 快手（激进派→务实派）

**技术路线**: OneRec(生成式) → UniMixer(判别式统一架构)

**演进路径**:
```
2025.2: OneRec端到端生成式
  → 语义ID + Session-wise生成
  → 观看时长+1.6%
  → 但遇到瓶颈: 参数量增大效果不增

2025-2026: 转向UniMixer判别式Scaling
  → 保留现有架构
  → 享受Scaling Law红利
  → 统一注意力/TokenMixer/FM三种Scaling块
  → 2026.4: UniMixer论文发布
```

**快手技术链完整视图**:
```
单场景生成      多场景扩展        平台级体系         广告/推理优化
   |               |                |                  |
OneRec      →    OneLoc         →  OneMall         →  GR4AD
(2025.2)    (本地生活geo-ID)   (电商+短视频+直播)   (LazyAR解码)
                                             →  GRank/PROMISE
                                             (process reward model)
```

**核心创新**:
1. QARM多模态对齐
2. Iterative Preference Alignment (IPA)
3. 端到端Session生成
4. geo-aware semantic ID (本地生活)
5. LazyAR解码结构 (广告推理预算优化)
6. UA-SID统一tokenization (广告内容+业务信号)
7. process reward model + inference-time scaling

**战略转变**: 从"颠覆式"到"渐进式"的代表，已形成连续演进的平台级生成体系

---

### 3.3 字节/抖音（务实派）

**技术路线**: HLLM → RankMixer → TokenMixer-Large → UG-Separation

**演进路径**:
```
2024: HLLM分层LLM推荐
  → Item LLM压缩Item为embedding
  → User LLM建模用户序列

2025.7: RankMixer判别式Scaling
  → Token Mixing替代Attention
  → MoE稀疏激活控制计算量
  → 抖音主Feed全量上线

2026.2: TokenMixer-Large
  → 70B/150B参数规模Scaling
  → mixing-and-reverting操作
  → Sparse Per-token MoE
  → 订单+1.66%, 广告ADSS+2.0%, 直播+1.4%

2026.2: UG-Separation
  → 用户组分离优化推理
  → W8A16量化
  → 推理延迟-20%
```

**核心创新**:
1. 混合参数化: 序列token共享 vs 非序列token独立
2. 跨请求KV Cache复用
3. 金字塔结构压缩序列
4. TokenMixer-Large: 工业级70B/150B参数Scaling
5. UG-Separation: 用户侧计算复用

**战略选择**: 判别式Scaling路线，低延迟高可控，持续优化推理效率

---

### 3.4 美团（深耕派）

**技术路线**: MTGR → SUAN → HoMer → EGA-V2(广告)

**演进路径**:
```
2025.5: MTGR 
  → 保留DLRM交叉特征(区别于Meta HSTU)
  → Group-LN统一异构特征空间
  → 动态掩码避免信息穿越
  
2025: 连续发布多篇论文
  → SUAN: 序列建模
  → HoMer: 层级记忆
  → EGA-V2: 端到端生成广告系统
```

**核心创新**:
1. Group-Layer Normalization
2. Sequence-Then-Sparse (STS) 训练策略
3. 动作感知多通道注意力(CamA)
4. 端到端广告生成框架

**特点**: 多篇论文密集发布，技术积累深厚

---

### 3.5 腾讯/Shopee（学术派）

**技术路线**: OnePiece + ColaRec

**演进路径**:
```
2025.9: OnePiece论文发布
  → 与人大高瓴学院合作
  → Shopee团队(腾讯投资)首次将LLM式上下文工程注入推荐
  → Shopee主搜索场景全量 (+2% GMV/UU, +2.90%广告收入)

2025: 算法大赛验证
  → 生成式推荐主题
  → 统一Scaling Law验证

2026: 广告one-model深融合
  → GPR: 端到端生成广告, 多级semantic ID + heterogeneous hierarchical decoder
  → OneRanker: generation与ranking架构级协同
```

**核心创新**:
1. 上下文工程框架(Prompt for Recommendation)
2. 隐式推理(Block-wise Reasoning)
3. 渐进式多目标训练
4. 协作语义ID统一Tokenizer
5. **GPR** — 广告端到端生成: 统一输入schema + 多级semantic ID + MTP/VAFT/HEPO训练链路
6. **OneRanker** — 处理三层深层矛盾:
   - 兴趣目标与商业价值目标的张力
   - 生成过程缺乏target awareness
   - 生成器与排序器之间的信息断裂
   → 引入task tokens、fake item tokens、ranking decoder、KV pass-through、Distribution Consistency loss

---

### 3.6 阿里（低调派）

**技术路线**: LUM系列 + SORT判别式ranking

**特点**:
- 公开工作较少
- 侧重用户长序列建模
- 内部探索多种路线

**核心创新（SORT）**:
1. request-centric sample organization — 请求级样本组织
2. local attention — 局部注意力替代全局attention
3. query pruning — 查询剪枝降低计算量
4. generative pre-training — 生成式预训练
5. 围绕电商ranking系统性改造Transformer

**意义**: SORT验证了判别式ranking backbone在生成式one-model升温背景下**依然存在明确且可观的进化空间**。与RankMixer"更适合推荐的Transformer变体"不同，SORT更接近"经过系统改造后仍可成立的Transformer本体"。

**定位**: 判别式大ranking backbone的重要对照工作

---

### 3.7 京东（工程派）

**技术路线**: xLLM推理优化 + xGR系统优化

**演进路径**:
```
2024: xLLM推理引擎开源
  → GitHub: jd-opensource/xllm
  → 工业级P99延迟控制
  
2024: xGR系统
  → xAttention: 共享KV优化
  → xBeam: Beam搜索优化
  → xSchedule: 调度优化
```

**核心创新**:
1. 推理引擎系统性优化
2. 生成式推荐工程落地实践

**特点**: 专注于推理系统优化，而非模型创新

---

### 3.8 百度（前沿派）

**技术路线**: GRAB

**核心创新**:
1. LLM启发的序列优先CTR预测
2. STS (Sequence-Then-Sparse) 训练策略
3. CamA多通道注意力机制

**特点**: 2026年最新工作，融合多个技术方向

---

### 3.9 Google（学术先驱）

**技术路线**: TIGER → 持续研究

**核心贡献**:
1. RQ-VAE语义ID生成
2. 生成式检索范式验证
3. 冷启动和多样性优势验证

**影响**: 后续快手OneRec、腾讯OnePiece均基于语义ID思路

---

### 3.10 其他厂商特色路线

**小红书 LASER** (arXiv:2602.11562, 2026):
- 长序列I/O访问与target-aware segmented attention结合
- 代表长序列全栈优化路线

**LinkedIn Feed-SR** (arXiv:2602.12354, 2026):
- Transformer-based sequential ranker工业落地
- RoPE、incremental training、recency weighting、late fusion
- 说明海外大厂在feed ranking中依然高度重视判别式sequential ranking

**Meta Foundation-Expert Paradigm** (arXiv:2508.02929, 2025):
- 平台级组织方式：中心foundation model + 轻量surface-specific experts
- 跨surface、跨模态、长期流数据的中心模型
- 代表"中心大模型 + 轻专家部署"的工业组织逻辑

**京东 OxygenREC** (arXiv:2512.22386, 2025):
- instruction-following引入生成式电商推荐
- "慢思考": 近线LLM生成Contextual Reasoning Instructions
- "快生成": 线上高吞吐encoder-decoder实时解码
- 将"慢思考"与"快生成"拆分

---

## 四、技术路径对比分析

### 4.1 生成式 vs 判别式核心区别

| 维度 | 生成式GR | 判别式Scaling |
|------|---------|--------------|
| **建模方式** | 自回归生成Next Item | 逐一打分排序 |
| **候选集** | 无需候选池，直接生成 | 必须有候选池 |
| **端到端** | 召回+排序统一 | 多阶段分离 |
| **延迟** | 高(Decoder自回归) | 低(并行打分) |
| **可控性** | 难调参，输出不稳定 | 易调参，可精确控制 |
| **Scaling Law** | 验证有效 | 字节RankMixer验证更稳定 |
| **广告落地** | 难(出价/计费) | 易(已有成熟体系) |
| **Ground Truth** | 模糊(Next Item不明确) | 清晰(点击/转化) |

### 4.2 各技术路径优缺点

#### 路径一: 生成式GR范式

**优点**:
- 端到端统一建模，消除级联误差
- 引入世界知识和推理能力
- 理论上可超越候选集限制

**缺点**:
- 延迟高，算力成本是判别式10倍+
- Ground Truth定义模糊，训练信号弱
- 广告场景落地困难(出价/计费难以融合)
- 效果不稳定，调参困难

**适用场景**: 搜索场景(低延迟要求)、长尾内容推荐

#### 路径二: 判别式Scaling范式

**优点**:
- 延迟可控，MFU高
- 工程成熟，与现有系统兼容
- 调参可控，效果可预测
- 广告落地无障碍

**缺点**:
- 仍是"匹配"非"生成"
- 受限于候选集大小
- 无法建模真正的用户意图推理

**适用场景**: 信息流推荐(高并发)、电商主场景

#### 路径三: 语义ID生成式检索

**优点**:
- 语义聚合改善冷启动
- ID空间有限，可控性强
- 生成过程可解释

**缺点**:
- RQ-VAE量化有信息损失
- 碰撞风险需处理
- 依赖物品内容特征

**适用场景**: 搜索召回、长尾物品推荐

#### 路径四: LLM赋能推荐

**优点**:
- 渐进式引入，不改变现有架构
- 世界知识注入改善泛化
- 冷启动场景优势明显

**缺点**:
- 延迟增加限制高并发场景
- 本质仍是辅助增强，非范式创新
- 收益有限，对大盘影响待验证

**适用场景**: 冷启动场景、多模态内容理解

---

## 五、核心挑战与关键问题

### 5.1 核心争议: 生成式 vs 判别式

**支持生成式**:
> "生成式推荐不是伪范式，而是未来，这是因为传统推荐范式无法具备scaling law" — 傅聪Cong

> "同等数据和算力规模下，谁通过生成式范式大幅提点了？生成式不是关键，Scaling Law才是" — BUGs

**质疑生成式**:
> "100%不看好生成式替代当前搜广推的召回-粗排-精排架构。生成式推荐最大的问题是没有ground truth" — 陈东文

> "字节全程没有明显押注，快手转UniMixer(判别式)，阿里无公开工作，腾讯换赛道" — 几野

### 5.2 关键问题分析

#### 问题1: Ground Truth缺失

**现象**: 推荐系统无法准确定义"正确的下一个Item"

**分析**:
- 用户行为有多种反馈(点击、收藏、购买、完播...)
- 每个Item可能满足不同维度的偏好
- 没有类似NLP的明确"正确句子"

**影响**: 
- 生成式模型训练信号弱
- 判别式模型有明确标签，训练更稳定

#### 问题2: 延迟与成本

**现象**: 生成式推荐Decoder延迟远高于判别式

**数据**:
- 传统CTR模型MFU: 4-5%
- 大模型训练MFU: 40-50%
- 生成式推理成本: 判别式10倍+

**解决方向**:
- 京东xLLM: 系统层优化
- 字节RankMixer/TokenMixer: 转向判别式
- 快手UniMixer: 统一架构Scaling
- Meta ULTRA-HSTU: 5x训练+21x推理加速
- 字节UG-Separation: 推理延迟-20%

#### 问题3: 广告落地

**现象**: 生成式推荐难以融入现有广告系统

**问题**:
- 广告需要出价/计费
- 生成式无法直接输出bid
- 多目标(点击+转化)难以融合

**尝试**: 美团EGA-V2端到端生成广告框架

#### 问题4: 可控性

**现象**: 生成式模型输出不稳定，调参困难

**对比**:
- 传统推荐: 调参→效果可预测
- 生成式: 调参→输出变化大

---

## 六、论文引用与实验对比分析

### 6.1 主要论文引用关系

```
Google TIGER (2023)
       ↓
   (启发)      快手OneRec (2025.02)
       ↓            ↓
       ↓      语义ID路线
       ↓            ↓
Meta GR/HSTU (2024.2) ← (参考)
       ↓
   (影响)      美团MTGR (2025.08)
       ↓            ↓
  工业Follow   保留交叉特征
       ↓
  字节HLLM (2024)
       ↓
字节RankMixer (2025.07) → 判别式Scaling
       ↓                      ↓
  (分化)              快手UniMixer (2025)
       ↓
  两条路线并行
```

### 6.2 关键实验结果对比

| 论文 | 评估指标 | baseline | 提升幅度 | 数据规模 |
|------|---------|---------|---------|---------|
| Meta GR | 生产端效果 | 传统DLRM | +12.4% | 亿级用户 |
| 快手OneRec | Watch Time | - | +1.6% | 主站全量 |
| 字节RankMixer | 业务指标 | 双塔模型 | 显著 | 抖音主Feed |
| 美团MTGR | 多个业务 | 已有GR | 持续提升 | 广告场景 |
| 腾讯OnePiece | 搜索效果 | 现有系统 | 显著 | Shopee主搜 |
| 京东xLLM | P99延迟 | 传统LLM serving | 大幅降低 | 工业环境 |

### 6.3 技术贡献总结

| 工作 | 核心贡献 | 创新程度 |
|------|---------|---------|
| Meta GR/HSTU | 验证Scaling Law，统一序列建模 | ★★★★★ |
| Google TIGER | 语义ID范式，冷启动改善 | ★★★★☆ |
| 快手OneRec | 端到端生成，RL偏好对齐 | ★★★★☆ |
| 字节RankMixer | 判别式Scaling，Token Mixing | ★★★★☆ |
| 美团MTGR | 交叉特征保留，Group-LN | ★★★☆☆ |
| 腾讯OnePiece | 上下文工程注入推荐 | ★★★☆☆ |
| 京东xLLM | 推理系统优化 | ★★★☆☆ |

---

## 七、未来展望与结论

### 7.1 技术路线判断

**短期(1-2年)**:
- 生成式作为"新组件"嵌入现有系统
- 判别式Scaling仍是主流
- 广告场景优先落地判别式

**中期(3-5年)**:
- 生成式可能在部分场景替代传统架构
- 技术收敛: 两条路线融合
- 端到端统一是方向但道路漫长

**长期**:
- 生成式是否成为主导范式: 不确定
- 取决于能否解决Ground Truth/延迟/广告问题

### 7.2 关键结论

1. **范式之争本质**: 不是"生成式 vs 判别式"，而是"是否准备好为新范式付出代价"

2. **技术路线分化**: 
   - 快手: 激进→务实 (OneRec→UniMixer)
   - 字节: 务实 (判别式Scaling优先)
   - 美团: 深耕 (多篇论文积累)

3. **核心挑战未解决**:
   - Ground Truth缺失
   - 延迟与成本
   - 广告落地
   - 可控性

4. **业界态度转变**:
   - 2024-2025.6: 狂热Follow
   - 2025.7-: 理性观望
   - 关注真实业务指标而非论文数字

---

## 八、五大行业明确趋势（2025-2026综合观察）

*来源: 知乎文章《从RankMixer到OneRanker》综合分析*

### 趋势一：判别式大ranking并未结束，而是进入成熟阶段

**核心判断**: 判别式ranker并未迅速失去演进空间，RankMixer、TokenMixer-Large、MSN、SORT、Feed-SR、MTFM等表明判别式大ranking依然存在明确且可持续的技术空间。

**真正发生的变化**: 判别式模型正在越来越像foundation model：
- 更统一的token接口
- 更深的一体化主干
- 更明确的scaling目标
- 更系统的serving优化

**工业竞争力**: 在强实时、高吞吐、严格时延约束的主链路场景中，判别式backbone仍然具备极强的工业竞争力。

---

### 趋势二：生成式one-model正在广告、电商与多场景推荐中加速落地

**核心判断**: 生成式推荐已经走过仅在retrieval试水的阶段，开始进入主排序、广告与电商推荐链路。

**标志性工作**: OneRec、MTGR、RankGPT、GPR、OneRanker、GR4AD、OxygenREC、OneMall

**广告领域推进更快的原因**:
- 广告天然面临多目标优化问题（点击+转化+价值）
- 页面级生成需求
- value alignment与全局收益优化
- 这些问题与生成式框架的建模方式天然契合

---

### 趋势三：semantic token正在从"表示技巧"上升为"系统级接口"

**核心判断**: item不再只是原子化ID，而正在演化为可组合、可迁移、可生成、可索引的token序列。

**被重新表述的传统问题**:
- 检索不再只依赖ANN索引
- 排序不再只面对固定item embedding
- 多场景推荐可共享更高层级的token空间
- 推理阶段的beam search、trie constraint、prefix constraint将逐渐成为推荐系统的一部分

**战略意义**: semantic token的意义已经远超"更有利于冷启动"这一局部优势，而正在成为推荐系统新的基础接口。

---

### 趋势四：serving、复用与inference-time scaling已成为一等公民

**核心判断**: 过去将线上部署放在最后一节作为工程实现说明；当前的情况已经发生变化。

**标志性工作**: UG-Separation、LASER、GR4AD、PROMISE、OxygenREC

**关键洞察**: 训练阶段的模型创新，如果不能转化为在线可控的计算图、存储访问模式与推理策略，就很难构成真正意义上的工业创新。

**新战场**: inference-time scaling不再是通用LLM独有的话题：
- beam设计
- path-level reward
- dynamic beam serving
- 用户侧复用
- 近线reasoning distillation

---

### 趋势五：统一化正在从"统一特征"走向"统一分布、统一目标、统一平台"

**核心判断**: 当前更激进的目标已经变成：
- 统一特征类型
- 统一场景分布
- 统一任务目标
- 统一训练与部署接口
- 统一推荐、搜索、广告之间的基础表达

**标志性工作**: OneTrans、MixFormer、MDL、MTFM、OneMall、OxygenREC

**推荐系统特有的foundation model逻辑**: tokenization、long context、multi-distribution、value alignment、serving co-design、one-model deployment

---

## 九、下阶段分水岭：三层"统一接口"

**核心判断**: 未来一到两年，真正拉开差距的因素大概率不会只是参数规模本身，而会是三类"统一接口"谁先成形。

### 第一层：统一的token接口
item、广告、内容、地理、场景、任务等对象，是否能够被映射到更稳定、更可扩展的语义空间中。

### 第二层：统一的backbone接口
无论是判别式统一backbone还是生成式one-model，核心竞争都在于：究竟由谁来承接sequence、feature、scenario、task与value这些异质信息。

### 第三层：统一的在线推理接口
训练、蒸馏、缓存、复用、beam search、近线reasoning之间能否形成闭环，将决定"论文中的大模型"能否转化为"线上可持续迭代的平台能力"。

### 核心结论
**推荐系统的大模型化，已经从"模型更大"演进到"接口重写"**。

---

## 附录: 主要论文列表

| 论文 | 作者/厂商 | arXiv ID | 年份 |
|------|----------|---------|------|
| GR (Generative Retrieval) | Meta | 2402.17152 | 2024.2 |
| HSTU (High-Speed Transformer for User) | Meta | 2402.17152 | 2024.2 |
| OneRec | 快手 | 2502.18965 | 2025.2 |
| RankMixer | 字节 | - | 2025.7 |
| MTGR | 美团 | 2505.18654 | 2025.8 |
| OneTrans | 字节 | 2510.26104 | 2025 |
| OnePiece | 腾讯/Shopee | 2509.18091 | 2025.9 |
| GRAB | 百度 | 2602.01865 | 2026.2 |
| TIGER | Google | 2305.05065 | 2023 |
| HLLM | 字节 | - | 2024 |
| xLLM | 京东 | 2510.14686 | 2025.10 |
| TokenMixer-Large | 字节 | 2602.06563 | 2026.2 |
| ULTRA-HSTU | Meta | 2602.16986 | 2026.2 |
| UG-Separation | 字节 | 2602.10455 | 2026.2 |
| UniMixer | 快手 | 2604.00590 | 2026.4 |
| VLM (Late Materialization) | - | 2604.24806 | 2026.4 |
| LLM推荐Scaling Law | - | 2602.07298 | 2026.2 |
| 生成式推荐综述 | 中科大等 | 2510.27157 | 2025.10 |

---

## 附录二：论文综合信息表

| 论文名称 | 作者/厂商 | arXiv 链接 | 发表年份 | 核心贡献点 | 技术路径类别 | 后续相关论文 |
|---------|----------|-----------|---------|-----------|------------|------------|
| **Recommender Systems with Generative Retrieval (TIGER)** | Google | [2305.05065](https://arxiv.org/abs/2305.05065) | 2023 | 1) RQ-VAE语义ID生成 2) 首个语义ID生成式检索模型 3) 验证冷启动和泛化能力 | 语义ID生成式检索 | 快手OneRec, 腾讯OnePiece, 多个生成式检索工作 |
| **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (GR/HSTU)** | Meta | [2402.17152](https://arxiv.org/abs/2402.17152) | 2024.2 | 1) HSTU架构替代标准Transformer 2) 工业级Scaling Law验证(+12.4%) 3) 统一序列建模框架(ICML 2024) | 生成式GR范式 | 美团MTGR, 小红书GenRank, 多个HSTU Follow工作 |
| **OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment** | 快手 | [2502.18965](https://arxiv.org/abs/2502.18965) | 2025.2 | 1) 首个端到端生成式推荐超过复杂级联系统 2) Session-wise生成替代逐点生成 3) DPO偏好对齐(+1.6% watch-time) | 生成式GR范式 | 后续UniMixer(快手转向判别式) |
| **MTGR: Industrial-Scale Generative Recommendation Framework in Meituan** | 美团 | [2505.18654](https://arxiv.org/abs/2505.18654) | 2025.8 | 1) 保留DLRM交叉特征(区别于Meta HSTU) 2) Group-Layer Normalization 3) 动态掩码避免信息泄露 4) 65x FLOPs提升 | 生成式GR范式 | SUAN, HoMer, EGA-V2(广告端到端) |
| **Towards Large-scale Generative Ranking (GenRank)** | 小红书 | [2505.04180](https://arxiv.org/abs/2505.04180) | 2025.5 | 1) 生成式重排架构 2) 验证生成架构主效益来源 3) 小红书 Explore Feed 大规模部署，A/B 显著提升用户满意度 | 生成式GR范式 | - |
| **OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer** | 字节 | [2510.26104](https://arxiv.org/abs/2510.26104) | 2025.10 | 1) 统一特征交互+序列建模 2) 混合参数化策略 3) 跨请求KV Cache 4) +5.68% GMV/UU (WWW 2026) | 生成式GR范式 | - |
| **OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System** | 腾讯/Shopee | [2509.18091](https://arxiv.org/abs/2509.18091) | 2025.9 | 1) LLM式上下文工程注入推荐 2) Block-wise隐式推理 3) 渐进式多任务训练 4) +2% GMV/UU, +2.90%广告收入 | 生成式GR范式 | - |
| **GRAB: An LLM-Inspired Sequence-First Click-Through Rate Prediction Modeling Paradigm** | 百度 | [2602.01865](https://arxiv.org/abs/2602.01865) | 2026.2 | 1) CamA多通道注意力机制 2) STS训练策略 3) +3.05% revenue, +3.49% CTR | 判别式Scaling | - |
| **xLLM Technical Report** | 京东 | [2510.14686](https://arxiv.org/abs/2510.14686) | 2025.10 | 1) 高性能LLM推理框架 2) PD/EPD解耦调度 3) 分布式KV Cache管理 4) 吞吐1.7x MindIE, 2.2x vLLM-Ascend | LLM赋能/推理优化 | xGR系统优化系列 |
| **HLLM: Hierarchical Large Language Model** | 字节 | - | 2024 | 1) Item LLM + User LLM分层架构 2) LLM压缩Item为embedding 3) 世界知识注入推荐 | LLM赋能推荐 | OneTrans统一架构 |
| **TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders** | 字节 | [2602.06563](https://arxiv.org/abs/2602.06563) | 2026.2 | 1) 70亿/150亿参数规模 2) mixing-and-reverting操作 3) 层间残差连接 4) Sparse Per-token MoE 5) 订单+1.66%, 广告ADSS+2.0%, 直播+1.4% | 判别式Scaling | UG-Separation优化推理 |
| **ULTRA-HSTU: Bending the Scaling Law Curve in Large-Scale Recommendation Systems** | Meta | [2602.16986](https://arxiv.org/abs/2602.16986) | 2026.2 | 1) 18层16k序列 2) 5x训练+21x推理加速 3) 消费+4%, 互动+8% 4) AT/MoT动态拓扑 5) 混合精度(BF16+FP8+INT4) 6) 7GB→2.3GB/层内存 | 生成式GR范式 | VLM(Versioned Late Materialization)基础设施 |
| **UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems** | 快手 | [2604.00590](https://arxiv.org/abs/2604.00590) | 2026.4 | 1) 统一注意力/TokenMixer/FM三种Scaling块 2) 参数化Token Mixing 3) 统一Scaling理论框架 4) UniMixing-Lite轻量版 | 判别式Scaling | 统一三大架构 |
| **Compute Only Once: UG-Separation for Efficient Large Recommendation Models** | 字节 | [2602.10455](https://arxiv.org/abs/2602.10455) | 2026.2 | 1) User-Group分离 2) TokenMixer用户侧计算复用 3) W8A16量化 4) 推理延迟-20% | 判别式Scaling优化 | TokenMixer-Large配套 |
| **Principled Synthetic Data Enables the First Scaling Laws for LLMs in Recommendation** | - | [2602.07298](https://arxiv.org/abs/2602.07298) | 2026.2 | 1) 首个LLM推荐Scaling Law 2) 合成数据消除噪声/偏置 3) recall@100提升+130% 4) 稳健幂律Scaling | LLM赋能推荐 | 指导LLM推荐预训练 |
| **A Survey on Generative Recommendation: Data, Model, and Tasks** | 中国科学技术大学等 | [2510.27157](https://arxiv.org/abs/2510.27157) | 2025.10 | 1) 生成式推荐全面综述 2) 数据/模型/任务三维度框架 3) 五大优势: 世界知识/NLU/推理/Scaling/创意生成 4) 挑战: 基准设计/鲁棒性/部署效率 | 综述 | 领域知识梳理 |
| **Versioned Late Materialization for Ultra-Long Sequence Training in Recommendation Systems at Scale** | - | [2604.24806](https://arxiv.org/abs/2604.24806) | 2026.4 | 1) 解决Fat Row范式存储/I/O瓶颈 2) UIH标准化存储+轻量版本指针 3) O2O一致性协议 4) 为ULTRA-HSTU提供数据基础设施 | 数据基础设施 | ULTRA-HSTU配套 |

---

## 附录三：技术演进脉络图

```
2023年
│
├── Google TIGER (arXiv:2305.05065)
│   └── 语义ID范式, RQ-VAE, 生成式检索先驱
│       │
│       └──→ 快手OneRec (2025.2) - 端到端生成式
│       └──→ 腾讯OnePiece (2025.9) - 协作语义ID
│
2024年2月
│
└── Meta GR/HSTU (arXiv:2402.17152) ─ ICML 2024
    │   开创者, 验证Scaling Law, +12.4%
    │
    ├──→ 美团MTGR (2025.8) - 保留交叉特征
    │    └──→ SUAN, HoMer, EGA-V2
    │
    ├──→ 小红书GenRank (2025.5) - 生成式重排
    │
    ├──→ 字节HLLM (2024) - 分层LLM
    │    └──→ 字节OneTrans (2025.10) - 统一Transformer
    │    └──→ 字节RankMixer/TokenMixer (2025-2026) - 判别式Scaling
    │         └──→ TokenMixer-Large (2026.2) - 70B/150B参数
    │              └──→ UG-Separation (2026.2) - 推理优化-20%
    │
    └──→ 京东xLLM (2025.10) - 推理引擎优化
         └──→ xGR系统优化

2026年2月: 新一轮技术突破
│
├── Meta ULTRA-HSTU (arXiv:2602.16986)
│    │  18层16k序列, 5x训练+21x推理加速, +4%消费+8%互动
│    │
│    └──→ VLM (arXiv:2604.24806) - 晚实例化数据基础设施
│
├── 字节TokenMixer-Large (arXiv:2602.06563)
│    │  70B/150B参数, mixing-and-reverting, Sparse MoE
│    │
│    └──→ UG-Separation (arXiv:2602.10455) - 用户组分离推理
│
├── 快手UniMixer (arXiv:2604.00590)
│    │  统一注意力/TokenMixer/FM三大架构
│    │
│    └──→ 统一Scaling理论框架
│
└── LLM推荐Scaling Law (arXiv:2602.07298)
     │  首个LLM推荐Scaling Law, 合成数据
     │
     └──→ 指导LLM推荐预训练

技术路线分化 (2025.6后)
│
├── 生成式路线: Meta ULTRA-HSTU持续深耕
│   └──→ 18层16k序列, 动态拓扑, 混合精度
│
└── 判别式Scaling路线: 字节TokenMixer-Large主导
     ├──→ 70B/150B参数Scaling
     └──→ UG-Separation推理优化

2025.10: 生成式推荐综述发布
│
└── A Survey on Generative Recommendation (arXiv:2510.27157)
     └── 数据/模型/任务三维度全面梳理
```

---

*报告生成时间: 2026-05-28 (2026年5月更新)*
*数据来源: 知乎话题回答 (126条) + 学术论文分析 + arXiv最新论文调研*
*更新内容: ULTRA-HSTU, TokenMixer-Large, UniMixer, UG-Separation, LLM推荐Scaling Law, 生成式推荐综述*
*说明: 表格中"后续相关论文"基于论文引用关系和技术脉络推断，非完整引用列表*