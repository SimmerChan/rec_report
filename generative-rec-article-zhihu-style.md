# 推荐技术路线深度分析——从生成式到Reasoning，2026年我们怎么看？

> 编辑于 2026年6月・北京
> 数据来源: 知乎话题（126条回答）+ arXiv最新论文 + 快手OneReason技术报告（arXiv:2606.06260）

***

> PS: 这是一篇综合分析，素材来自知乎上126条回答、arXiv最新论文，以及2026年6月9日快手刚刚发布的 OneReason 技术报告（arXiv:2606.06260）。写这篇文章的目的，是想系统梳理一下这两年推荐算法领域最热闹的方向——从生成式推荐到刚刚兴起的推荐 Reasoning——到底在发生什么。不保证全面，但保证有态度。

***

## 导语

最近这两年，如果你关注推荐系统领域的技术演进，有两个词一定绕不开：**生成式推荐**（Generative Recommendation）和**推荐 Reasoning**（Reasoning Recommendation）。

从2024年2月Meta发布GR/HSTU论文开始，工业界对生成式方向的热情一度非常高。各大厂纷纷跟进，快手、字节、美团、腾讯轮番发论文，仿佛一夜之间推荐系统就要被"重写"了。但到了2026年再看，业界的态度已经从狂热转向**理性观望**——有人坚定看好，有人坚决不看好，有人悄悄转向了另一条路。

而2026年6月9日快手发布的 OneReason（arXiv:2606.06260）值得**初步关注但不宜过度解读**——首次在工业级推荐场景中验证了"先思考再推荐"（Thinking-then-Recommending）的可行性，把 LLM 的 Reasoning 范式迁移到推荐基模。本地生活广告 A/B 实验曝光 +10.33%、收入 +8.234%，ROI > 5，年化数亿元商业增量。**但需注意：线上 A/B 是 Fast + Slow 两条范式合计收益，OneReason-8B 本体只做离线召回；"thinking 模式领先"是离线 Pass@4 指标，不是 A/B 收益。** 距离"推荐系统下一个分水岭"还有相当距离——本文将 OneReason 定位为"第五路线的早期探索"，而非"范式转移的宣言"。

说实话，这篇文章就是想把这件事说清楚：**从生成式到 Reasoning，推荐技术路线到底发展到哪了？五条技术路线各自的优劣是什么？未来1-2年真正靠谱的方向是什么？**

***

## 一、技术演进：两年时间，发生了什么

先把时间线理清楚。

2023年Google发布TIGER论文，提出用RQ-VAE做语义ID生成式检索，算是这个方向的学术先驱。但真正让工业界沸腾的，是2024年2月Meta的GR/HSTU论文——首次在工业级推荐系统上验证了Scaling Law存在，线上效果提升12.4%。这个数字太亮眼了，直接点燃了整个行业。

之后的事情就很快了：

- **2024年**：各厂商密集跟进，快手OneRec、字节HLLM相继发布
- **2025年**：技术路线开始分化——快手从生成式转向UniMixer判别式，字节主推RankMixer判别式Scaling
- **2026年2月**：Meta发布ULTRA-HSTU，18层/d=512/16K序列处理能力，5.3×训练+21.4×推理加速；同时字节TokenMixer-Large（7B/15B参数）成为判别式Scaling的标杆
- **2026年6月**：快手发布 OneReason（arXiv:2606.06260），开辟**推荐 Reasoning** 全新方向——578B token 四层预训练 + 推荐专属 CoT（归纳/溯因/演绎三段式）+ GRPO 改进 + Fast-Slow Thinking 部署架构。本地生活广告 10 天 A/B 曝光 +10.33%、收入 +8.234%，首次在工业级推荐场景让 "thinking" 模式稳定超过 "non-thinking"。

如果要用一句话概括这两年：**探索期→爆发期→分歧期→深耕期→Reasoning 期**，技术路线已经从"要不要跟进生成式"变成了"具体怎么落地"，又从"Scaling落地"延伸到了"Reasoning赋能"。

***

## 二、五条技术路线：生成式、判别式、语义ID、LLM增强、Reasoning

2026年6月之前，整个推荐系统的技术路线被划分为四条：生成式 GR、判别式 Scaling、语义 ID 生成式检索、LLM 增强推荐。但 OneReason（arXiv:2606.06260）的发布让这一框架需要扩展——**推荐 Reasoning** 成为第五条独立路线，也是 LLM 时代"Reasoning 浪潮"在推荐系统的标志性落地。

这是整个领域最核心的问题，也是知乎上争论最多的地方。

### 路线一：生成式GR范式

核心理念很简单：把推荐问题建模为"序列生成"任务，模型直接生成下一个Item ID，像ChatGPT生成下一个词一样。

代表工作：
- **Meta GR/HSTU (2024.2)**：开创者，HSTU结构替代标准Transformer，验证Scaling Law，+12.4%生产端提升
- **快手OneRec (2025.2)**：端到端生成，语义ID + RL偏好对齐，快手官方披露观看时长+1.6%
- **Meta ULTRA-HSTU (2026.2)**：18层/d=512/16K序列，5.3×训练+21.4×推理加速（相对vanilla HSTU），生产端消费+互动综合提升4%-8%、topline（DAU/访问量）+0.217%

技术架构上，是这样的：

```
用户行为序列 → [Encoder] → [Decoder: 自回归生成] → Semantic ID/Item ID
```

优点很明显：端到端统一建模，消除级联误差；理论上可以超越候选集限制；Scaling Law验证有效。

但问题同样明显：

| 维度 | 问题 |
|------|------|
| **延迟** | 自回归生成Decoder成本高，是判别式10倍+ |
| **Ground Truth** | "下一个物品"定义不唯一，点击/收藏/购买满足不同偏好 |
| **广告落地** | 出价/计费逻辑难以融合 |
| **可控性** | 调参→输出变化大，不如判别式稳定 |

### 路线二：判别式Scaling范式

这条路线的思路是：不改变判别式架构，通过增大模型规模和数据实现Scaling Law。

代表工作：
- **字节RankMixer (2025.7)**：Token Mixing替代Attention，MoE稀疏激活，抖音主Feed全量
- **字节TokenMixer-Large (2026.2)**：7B在线/15B离线参数，订单+1.66%、GMV+2.98%、广告ADSS+2.0%、直播+1.4%
- **字节UG-Separation (2026.2)**：用户组分离推理，推理延迟-20%
- **快手UniMixer (2026.4)**：统一注意力/TokenMixer/FM三种Scaling块
- **阿里SORT (2026.3)**：判别式ranking backbone，STS训练策略，电商场景订单+6.35%、买家+5.97%、GMV+5.47%、延迟-44.67%、吞吐+121.33%

技术架构是这样的：

```
用户特征 + Item特征 → [Transformer/TokenMixer] → [判别式输出: CTR/CVR]
```

优点：延迟可控（并行打分），MFU高（算子优化成熟），调参可控，广告落地无障碍。

缺点：本质上还是"匹配"非"生成"；受限于候选集大小；无法建模真正的用户意图推理。

### 路线三：语义ID生成式检索

还有一条路，不直接生成Item ID，而是先生成Semantic ID——通过量化将Item映射为离散的语义token序列，再做检索。

代表工作：
- **Google TIGER (2023)**：RQ-VAE语义ID，生成式检索先驱
- **快手OneRec (2025)**：Semantic Tokenization + Session-wise生成，KuaiShou Inc. 2025.02论文

这条路的优势在于：相近物品共享前缀token，可以解决冷启动；ID空间有限，输出可控。但问题也很明显：RQ-VAE量化存在碰撞风险，连续embedding转离散ID有信息损失。

### 路线四：LLM增强推荐

还有一条更务实的路：不用LLM替代推荐系统，而是用LLM作为特征增强模块，嵌入现有系统。

代表工作：
- **字节HLLM (2024)**：Item LLM + User LLM分层架构
- **阿里LUM (2024-2025)**：长序列用户建模
- **京东xLLM (2025.10)**：推理引擎优化，开源了GitHub仓库

这条路本质是"渐进式引入"，不改变现有架构，世界知识注入推荐，冷启动改善明显。但收益有限，作为特征增强而非范式创新。2026年2月还有一个重要突破：**首次验证了LLM推荐的Scaling Law**（arXiv:2602.07298, ICML 2026接收），通过分层合成数据消除原始用户行为数据的噪声/偏置/不完整性——使用合成数据训练的SasRec在**recall@100上比用真实数据训练高+130%**，并首次在LLM上观察到可预测的perplexity幂律下降。

### 路线五：推荐 Reasoning 探索（2026.6 起步）

2026年6月9日快手发布的 **OneReason**（arXiv:2606.06260）是这一路线的首次工业级探索，而非"开山之作"或"已确立的范式"。其核心理念是：把 LLM 时代"先思考再回答"的 Reasoning 范式（OpenAI o1、DeepSeek R1 范式）迁移到推荐系统，让推荐基模在生成推荐前先显式推理"为什么推这条"。

**客观定位**：OneReason 目前是**方法论验证 + 单点业务落地**阶段，远未到范式成熟。论文 §A.4.3 自承 "Fast-Slow 是工程折中，**cannot fully replace real-time retrieval nor fully exploit the model's capacity**"。本文对这条路线应保持"技术方向值得跟进、当前价值需客观评估"的态度，避免过早下"范式转移"等大结论。

**核心洞察**：推荐 Reasoning 不是数学推理那种"演绎式"，而是"溯因推理"（论文 §5.4 实际是归纳+溯因+演绎三段式，**不是单一溯因**）——用户兴趣不可见，模型需要从长期、嘈杂、跨域的行为序列中反推潜在兴趣，理解兴趣演化，并判断候选物料为何适合当前上下文。

代表工作：
- **快手 OneReason (2026.6)**：578B token 三阶段预训练（预热 110B + 全参 449B + 长序列 19B）+ 四层预训练数据架构（Token/Item/Relational/User 四种数据粒度）+ 推荐专属 CoT（Persona Abstraction → Interest Expansion → Transition Inference 三段式）+ GRPO 三方面改进（两阶段轨迹生成 + Set-wise奖励 + 优化稳定策略）+ "先专后合"训练链路 + Fast-Slow Thinking 双轨部署架构。本地生活广告 10 天 A/B（5% vs 5% 流量）曝光 +10.33%、收入 +8.234%。

**实际部署架构（论文 §9.2 + §A.4.3 客观描述）**：

```
┌────────────────────────┐
│ OneReason-8B（离线）    │ ── Slow Pipeline ──→ Redis 候选池
└────────────────────────┘                          │
                                                    ▼
用户请求 ──→ 候选集合融合(OneReason + OneRec) ──→ 既有 ranker ──→ 曝光
                                                    ▲
┌────────────────────────┐                          │
│ OneRec + Thinking Token│ ── Fast Pipeline（蒸馏版,实时）──┘
└────────────────────────┘
```

- **OneReason-8B 本体只做离线召回**（写入 Redis 候选池，不进主链路早期竞争）
- **线上排序是「蒸馏版 OneRec + 快手既有独立 ranker」**，大模型没有进入主链路
- **范式 A**（Slow Pipeline 直接召回）：曝光 +0.940%、收入 +4.528%
- **范式 B**（Fast Pipeline 蒸馏到 OneRec）：曝光 +6.831%、收入 +4.636%
- **Combined = A + B**：曝光 +10.332%、收入 +8.234%——**这是两条范式合计的线上收益，不全是 thinking 模式本身的贡献**
- **600 张旗舰 GPU 服务 4 亿用户，ROI > 5**

> **重要说明**：以上 10.33% / 8.234% 是 Fast Pipeline（蒸馏版 OneRec） + Slow Pipeline（OneReason 离线召回）**合计**的线上 A/B 收益。论文中"thinking 模式在 Pass@4 上领先 non-thinking +13.45%"是**离线 benchmark**上的指标，不应与线上 A/B 数字混用。

技术架构：

```
四层预训练数据（578B token） → 预训练基座 → 推荐专属CoT SFT → GRPO改进RL → Fast-Slow Thinking 部署
                                                          ↓
                  Slow Thinking（近线）：完整 OneReason 模型 → 离线召回+推理链生成
                  Fast Thinking（在线）：OneReason for OneRec（蒸馏/剪枝）→ 在线快思考服务
                                                          ↓
                                                    排序模型融合
```

**与前四条路线的客观差异**：

| 关键差异 | 说明（**需注意：部分是潜在优势，未全部实证**） |
|---------|------|
| **CoT 可读性** | 离线推理链显式可读，但**主链路已蒸馏成 0.8B embedding**，业务约束尚未直接写在推理层（论文 §A.4.3 自承 "current strategy still lacks consideration of business value"） |
| **归纳+溯因+演绎** | 论文 §5.4 实际是三段式（induction + abduction + deduction），不是单一溯因推理 |
| **多业务统一潜力** | 一个懂物料语义的基模 + 业务说明即可跨域出方案——这是**潜在优势**，本次只在本地生活广告单点验证 |
| **Agentic前置（论文未充分论证）** | "Agentic Recommender Harness" 是论文 §11 提的 **future work**，尚无实验或落地；"前置地基"是推断而非论文结论 |

**核心收益**（**严格区分线上 vs 离线指标**）：

- **线上 A/B（5% vs 5% × 10 天，本地生活广告场景）**：曝光 +10.332%、收入 +8.234%，ROI > 5，年化数亿元
  - **注意**：这是范式 A（Slow）+ 范式 B（Fast）**合计**的线上收益，**不全是 thinking 模式本身的贡献**
  - 范式 A 单独：曝光 +0.940%、收入 +4.528%
  - 范式 B 单独：曝光 +6.831%、收入 +4.636%
- **离线 benchmark（推荐准确度）**：Pass@4 上 thinking 平均领先 non-thinking +13.45%，首次在工业级推荐让"先思考"稳定成为正资产
- **通用能力**：MMLU-pro、GPQA-Diamond 基本保留 Qwen3-8B 原始水平，没有"训练推荐训坏通用能力"
- **CoT 能力内化**：引入 CoT 监督不仅提升 thinking，还能反哺 non-thinking 性能

**OneReason 的局限性**（论文 §A.4.1-§A.4.3 自承部分）：

- **Fast-Slow 是工程折中**：8B 推理无法实时在线，"**cannot fully replace real-time retrieval nor fully exploit the model's capacity**"
- **小 beam 解码**：覆盖率与效率的折中，长尾物料曝光仍是问题
- **增量训练代价**：continual pretraining 主要面向推荐目标，部分牺牲了指令遵循与通用能力
- **业务价值未直接进解码**："current strategy still lacks consideration of business value and resource constraints"（eCPM、ROI 尚未灌进 decoding objective）
- **0.8B 实时化是未来工作**：今天上 8B 实时仍有 RT 问题
- **单实验域**：仅本地生活广告 5% × 5% 验证，跨业务（短视频主 feed、电商、直播）迁移性还需再验
- **新物料端到端曝光周期 ≈ 1 个增量训练周期**：天/小时级，离线重推理

**关键技术细节**（注意区分"四层数据架构"和"三阶段训练策略"是不同概念）：

1. **四层预训练数据架构**（解决 item-text 语义割裂，"层"指数据粒度而非训练阶段）：
   - Token 粒度：子 Token 拆解组合、单 Token 释义、前缀语义预测
   - Item 粒度：物料描述粗粒化 + 多视角 Item QA
   - Relational 粒度："物品→兴趣说明文本→后续物品"链路数据
   - User 粒度：分域分组、全时序穿插、Item 替换为文本

2. **三阶段预训练策略**（"阶段"指训练进程，**与上面"四层数据"正交**）：
   - 预热（110B）：冻结主干，仅优化新增 item 嵌入
   - 全参训练（449B）：四层数据联合深度对齐
   - 长序列优化（19B）：上下文窗口放开至 32K

3. **R3 推荐 CoT 三段式**：
   - Persona Abstraction（用户抽象）：20 类用户偏好画像（家庭主妇、直播购物爱好者、共享设备用户等）
   - Interest Expansion（兴趣发散）：宽度 n=1/3/5 表现最佳（"少即是多"，n=10/20 衰减）
   - Transition Inference（兴趣推断）：证据强度+近期性+连贯性+画像匹配综合评估

4. **CoT 质量评估 5 维度**：
   - Safety（不剧透目标）/ Consistency（结论对齐）/ Logic（真正归纳非复读）/ Factuality（基于真实行为）/ Informativeness（提供洞察）

5. **RL 改进的 GRPO 三方面**：
   - 两阶段轨迹生成：先生成推理轨迹，再扩展多个候选推荐，缓解奖励稀疏
   - Set-wise 奖励：从 point-wise 抬升到 set-wise/list-wise，鼓励多方向兴趣覆盖
   - 优化稳定策略：推理文本 token 和推荐 itemic token 采用不同裁剪范围

6. **"先专后合"训练链路**：
   - Specialize：每领域（视频/电商/广告/直播）独立 RL
   - Unify：融合多领域专家，两种技术路线——RFT（Rejection Sampling FT）+ MOPD（Multi-Teacher On-Policy Distillation）

**重要问题：为什么 OneReason 之前 "thinking 反而 hurts"？**

之前 OneRec-Think、OpenOneRec 等公开尝试都观察到 thinking 模式劣于 non-thinking 模式。OneReason 通过分析找到根因：

1. **itemic token 与自然语言语义割裂**：item 只是离散标识符，不是可推理的语义单元
2. **缺乏推荐专属推理结构**：直接套用通用 LLM 的 CoT，期待通过泛化得到推荐思维链，但实际失败

**OneReason 的解法**：通过四层语义对齐预训练 + 推荐专属三段式 CoT + GRPO 专项改进，让 CoT 从"分散注意力"变成"真正辅助推荐决策"——SFT 后 Delta LL 全为负，RFT 后 Delta LL 全转正（0.63/1.27/0.57/1.10），这就是"RL 是解锁推理收益必备环节"的实证。

#### R0-R3 能力栈：OneReason 的诊断架构（**新增**）

OneReason 论文 §2 把推荐推理拆成 4 层渐进能力，**既是训练目标也是评测目标**，每一层有明确任务定义：

| 能力层 | 任务 | 评估什么 | 关键洞察 |
|--------|------|---------|---------|
| **R0 感知 Perception** | ①Item Understanding（item→描述）②Itemic Pattern Grounding（描述→item）③Item QA（多选题） | 模型能否把 itemic token 锚定到自然语言语义 | 论文案例：模型可能从短剧 token 幻觉出错误剧名，把都市情感短剧误判为医疗题材——**不是表达风格问题，是感知对齐问题** |
| **R1 推导 Derivation** | Item2Item：给定源 item 和候选集，选最匹配目标关系的 item | 从单个 item 语义推出 item-to-item 关系 | "不是泛化相似性，而是 source-to-follow-up bridge"；6 个 bridge variables 锁死推理跳点（source-side need / bridge type / abstract bridge / continuation direction / reason seed / confidence） |
| **R2 演变 Evolution** | ①Action Selection ②Topic Generation ③Direct Generation | 把同潜在兴趣的 item 推理为时间过程 | 三种任务变体"自主程度递增"；识别"触发→细化→收敛→替代"过程 |
| **R3 推荐 Recommendation** | ①Single-Domain ②Cross-Domain | 综合画像/历史/目标域，给出最终推荐 | 把 R0-R2 能力在最终的下一次交互预测设置中组合 |

**R0 的关键消融数据（0.8B 模型，30B token 预算）**：
- 加 Token 粒度：R1 Item2Item QA 从 0.00% → 20.57%
- 加 User 粒度：R1 达 29.72%，R2 Evolution Direct Gen 0.37%，Cross-Live 3.25% → 8.56%，Cross-Ad 8.58% → 10.84%

**重要判断**：**R0 不成立，后续所有推理都是建立在不可靠语义上**。这一观点借鉴 MLLM 文献的"读图/读表失败模式"——文本-视觉对齐不足时，模型"机械地读取表面文字"而非真正推理。

#### CoT 质量评估 5 维度（**新增**）

推荐推理痕迹沿 5 个互补维度评估质量：

1. **Safety** — CoT 是否直接暴露目标内容
2. **Consistency** — 痕迹结论是否与预期推荐目标对齐
3. **Logic** — 痕迹是否展示可迁移的推理而非表面拼接
4. **Factuality** — 痕迹是否准确反映观察到的用户历史
5. **Informativeness** — 推理是否提供有用的推荐指导

#### R3 推荐 CoT 的 5 大工业约束（**新增**）

Transfer Inference 阶段服从 5 大约束，保证 trace 的工业可用性：

1. **Evidence priority** — 仅引用输入上下文出现过的行为证据，禁止幻觉未出现历史
2. **Transition bridge** — 必须明示从压缩用户状态 C_u 到目标域 d 的逻辑跳点
3. **Granularity calibration** — 用跨 IP/品类粒度而非单个 IP 名称重复，支持跨 IP 泛化
4. **Conflict resolution** — 多兴趣冲突时，以近期轨迹强度为准仅选一
5. **Leakage control** — 关键约束："**reasoning must be written as if the answer is unknown**"——严禁 trace 出现 target 的标题/item ID/任何唯一标识

#### GRPO 推荐 RL 三大机制（**新增**）

GRPO（Group Relative Policy Optimization）从 DeepSeek-Math 移植到推荐场景时，OneReason 做了三处关键改造：

1. **两阶段 Rollout**（解决推荐 RL 奖励极稀疏问题）：先采样 N 个推理痕迹，再对每条 trace 并行生成 K 个 itemic token 序列 → N×K 个有效 rollout，但只生成 N 次长 CoT。论文 Figure 10 给出了完整图示。
2. **阶段感知裁剪（Stage-wise Clipping）**：推理 token 与物品 token 行为截然不同——推理痕迹长且主要支持探索，最终物品 token 短但直接决定奖励。**推理 token 用宽松裁剪 ε_CoT 鼓励探索，物品 token 用紧裁剪 ε_item 防止分布崩塌**。这个设计本质是把 *exploration* 与 *exploitation* 交给不同 token 类型分别管理。
3. **负样本降权（Negative-sample down-weighting）**：在 batch 内对命中真实值的 rollout 赋权重 1.0，对未命中赋 β < 1，缓解非命中样本主导梯度的问题。

#### 阶段感知裁剪的工程意义（**新增**）

**直觉解读**：为什么要"推理宽松、物品严格"？推理痕迹是"**怎么想**"，允许多样探索才能跳出局部最优——一旦裁剪过紧，模型会快速"背诵"出某一种推理路径而不再探索。物品 token 是"**选哪个**"，不能放任分布乱飘——一旦概率分布失控，就会生成不合法 token、出现 illegal item。

**RFT 与 MOPD 的分工**：
- **RFT（Rejection Sampling Fine-tuning）**：从各域教师中采样，**保留命中 ground-truth 且推理质量合格的轨迹**，聚合成高质量 CoT 数据继续训练。论文指出 RFT **更稳定保证 thinking > non-thinking**。
- **MOPD（Multi-Teacher On-Policy Distillation）**：学生按自身策略生成轨迹，按任务域路由到对应教师，通过 token-level **reverse-KL** 信号进行在线蒸馏。**MOPD 同步提升 thinking 和 non-thinking**，在 Product 和 Live 域表现更强。**reverse-KL 本质决定了其能力上限**：reverse-KL 是 mode-seeking 的，不完美复制教师的全分布——在 K ≤ 16 时学生可以完美追上甚至超过教师，但在 K > 16 时仍会保留 long-tail 差距。

#### CoT 似然增益（ΔLL）的实证意义（**新增**）

**核心实证**：SFT 模型在 4 域的 ΔLL 全为**负**（CoT 分散了对真实目标的注意力），RFT 模型在 4 域的 ΔLL 全为**正**（CoT 真正辅助了目标预测）。这是"RL 是解锁推理收益必备环节"的最强实证——SFT 学不会，RFT 才能学。

进一步发现：**RFT 模型的 CoT 似然进展在少量推理段后即饱和**（"reaching near-optimal likelihood after only a few reasoning segments"），这提示 **adaptive early-stopping 机制**（论文 §11 future work）可进一步压缩推理长度，节省 30-50% 长度。

**未来方向（Agentic Recommender）**：

OneRec 团队下一步计划：打造 **Agentic Recommender Harness**，让推荐基模具备规划与工具调用能力，逐步驱动推荐系统向 Agentic 架构演进。从"千人一面的固定流水线"走向"千人千策、能规划、能用工具、能多轮对话"。

#### 路线五的源流：R4ec → OneRec-Think → OpenOneRec → OneReason（**重要补充**）

OneReason 不是凭空出现的。它是快手 2025-2026 年**四步迭代的第四步**，每一步都解决上一步的遗留问题：

| 工作 | arXiv / 会议 | 时间 | 核心思路 | 线上结果 | 思考模式表现 |
|------|------|------|---------|---------|------------|
| **R4ec** | arXiv:2507.17249 / **RecSys 2025** | 2025.07 | Actor + Reflection 双 LLM 反复迭代反思改进 | **收入 +2.2%**（基线 20% / 实验 10% 流量 × 14 天） | 反思作为过滤器，**System-2 思维** |
| **OneRec-Think** | arXiv:**2510.11639** | 2025.10 | In-Text Reasoning + Itemic Alignment + Reasoning Activation + Reasoning Enhancement | **停留时长 +0.159%**（1.29% 流量） | **开了思考反而更差** |
| **OpenOneRec** | arXiv:**2512.24762** | 2026.01 | RecIF-Bench（8 任务 / 4 层能力）+ 1.7B/8B 开源 + 公开 9600 万条交互 | 公开学术版（SOTA on Amazon 10 域 Recall@10 +26.8%） | 同上，**未稳定超过 non-thinking** |
| **OneReason** | arXiv:2606.06260 | 2026.06 | R0-R3 能力栈 + 四层语义对齐预训练 + 专精后统一 RL | **曝光 +10.33% / 收入 +8.234%**（5%×5%×10 天） | **首次稳定超过 non-thinking**（Pass@4 / Recall@4 4 域全正） |

**关键时间洞察**：

1. **R4ec（2025.07）是真正的"前序思想工作"**——在 OneRec 体系之外，由快手 + 中科院团队独立发表。核心思想是 *actor π_θ* 推理 + *reflection π_ψ* 挑错 + LoRA 微调，**System-2 思维**进入推荐。收入 +2.2% 的同时**对长尾数据效果更明显**。这是 LLM 推理范式在推荐系统的早期工业落地。

2. **OneRec-Think（2025.10）首次"硬加 CoT"但失败**——停留时长仅 +0.159%（1.29% 流量），更关键的是论文自己披露"thinking mode 相对 non-thinking mode 并未稳定显示优势"。这与 OpenOneRec 的现象一致。

3. **OpenOneRec（2026.01）开放基础模型**——首次开源 OneRec 全套训练框架 + 1.7B/8B 模型 + 9600 万条数据，配套提出 RecIF-Bench 评测。但同样观察到 thinking 模式相对 non-thinking **未稳定超越**。

4. **OneReason（2026.06）首次"thinking > non-thinking"稳定成立**——在四域跨域推荐基准上（Cross-Video / Cross-Product / Cross-Ad / Cross-Live），Pass@64 / Recall@64 全部正收益（+0.15 到 +0.33 个百分点）。这是知乎讨论"推荐 CoT 是不是伪命题"问题的关键转折点。

**为什么 OneReason 这次成了？OneReason 论文归因到两个根因**：

1. **感知（Perception）问题**：itemic token 缺少自然语言语义锚定，CoT 在"没有真语义"的 token 上推理，相当于"让看不懂题目的学生展示解题过程"——步骤越详细越错。R0 任务（Item Understanding、Itemic Pattern Grounding、Item QA）就是为了诊断这个根因。
2. **认知（Cognition）问题**：通用 LLM 的 CoT 模板（reasoning scaffold）没考虑推荐场景"多合理答案 + 不可观测意图"的特殊性。R1-R3 任务栈把"物品间一跳关联（R1）/ 时间兴趣演变（R2）/ 综合推荐决策（R3）"拆成可监督的能力层级。

**相关独立验证工作（同期业界对推荐 Reasoning 的多路探索）**：

- **ReaRec（人大 x 阿里，2025.04）**：推理时计算框架，K 步隐式推理 + 推理位置嵌入 RPE。在仅增加 3.51% 平均延迟（固定 2 步）下，**推荐表现 +7.49%**；最优推理步数事后分析显示 **30-50% 性能提升空间**（Yelp 数据集）。这是**业界第一个公开证明推荐系统存在推理时 Scaling Law 潜力**的工作。但论文自承未实现理想"推理步数增加性能上升"的 inference-time scaling law 曲线，列出 4 个未来方向：自适应推理深度 / inference-time scaling / 编推解耦 / 高效推理。
- **OnePiece（傅聪 x 人大，Shopee Search，2025.09）**：上下文工程（锚点物品序列作为专家示例）+ 隐式推理 + 多目标训练，**Shopee Search 主场景全流量生效**。登顶 HuggingFace Daily Paper 前三。意义：**搜索侧（区别于推荐侧）已经全量落地生成式 + 推理范式**，比 OneReason 早 9 个月验证"工业级全量"。
- **阿里 AMEN（Autoregressive Modeling of Evolving Next-interest）**：Next Interest Flow 概念——用自回归语言模型预测用户未来兴趣的稠密向量轨迹，注入传统 CTR 模型。淘宝信息流线上 **GMV +11.24%**（信息流 GMV 能稳定 +1% 即值得全量，11.24% 是异常高数字）。与 OneReason 思路的关键差异：**阿里的"Reasoning"是预测未来，不显式生成 CoT 文本**；**快手的"Reasoning"是显式 CoT + RL**。
- **CQ-SID（阿里/天猫 App 搜索，arXiv:2605.14434，2026.05）**：把生成式召回当作"补充通道"而非端到端替换——Category-Query constrained SID + EG-GRPO。意义：阿里在搜索场景的 GR 答卷与 OneReason 思路平行但不同（无显式 CoT，但有 GRPO 增强）。
- **UniVA（腾讯/微信视频号广告，arXiv:2605.05803，2026.05）**：Unified Value Alignment for Generative Recommendation，**eCPM-aware RL 让 bid 方差下降 + GMV 提升同时拿到**。意义：腾讯在广告场景的 GR 答卷，与 OneReason 的"思考"思路分叉——直接对齐商业价值，不做 CoT。

**知乎"推荐 Reasoning" 讨论的三个核心立场**（基于 2026.06 知乎相关话题调研）：

1. **乐观派（与快手官方一致）**：浅梦、Du 论文精读博主等认为 OneReason 是首个 thinking 稳定超过 non-thinking 的工业基础模型（论文评分 8.4/10），是 LLM Reasoning 浪潮在推荐系统的标志性落地。
2. **理性派（与本文立场一致）**：知乎专栏"推荐系统到底存不存在'思维链'"（猫的薛定谔，101 票）提出更冷静的视角——**"推荐系统大概率并不存在 LLM 意义上的'思维链'——它的内核从头到尾还是协同过滤，所谓'推理链/CoT'，本质上是借了一个时髦概念，给协同过滤加了一层正则化约束"**。这一观点的解释力很强：OneRec-Think 用 LLM 蒸馏 CoT（注入 LLM 知识）、OneReason 用 R0-R3（注入结构化监督）、CaLIR 用商品类目层级（注入树先验）——**所有"有效的"CoT 都在注入原本不在模型手里的外部信息**，否则 CoT 只是个空草稿区。
3. **批判派（"为什么思考会伤害"的诊断派）**：Zhang et al. 2026 的工作 "Why Thinking Hurts"（arXiv:2506.00007）正式测量了 OpenOneRec 上的非思考/思考差距，**归因到"来自通用子空间的文本惯性"**——即模型在没有真正理解 itemic 语义的情况下，硬套自然语言 CoT 模式，反而把生成的内容拖离推荐目标的真实分布。OneReason §10 Related Works 显式引用这一工作，说明快手团队是承认这一根因的，并把它作为方法论设计的靶子。

> **本文的判断**：OneReason 既是真正的工程突破（首次让 thinking 稳定生效），也是范畴争议的延续（"思维链"在推荐里到底是不是真"思考"）。**目前来看，把它定位为"推荐 Reasoning 路线的首次工业级验证"是恰当的；定位为"范式转移"还为时尚早**。

***

## 三、五条路线的核心差异

| 维度 | 生成式GR | 判别式Scaling | 语义ID检索 | LLM增强 | Reasoning推荐 |
|------|---------|--------------|-----------|---------|---------------|
| **建模方式** | 自回归生成Next Item | 逐一打分排序 | Semantic ID量化+生成 | LLM特征增强 | 先思考再推荐（CoT+RL，**仅离线召回+蒸馏，主链路仍是既有 ranker**） |
| **编码空间** | 原子化Item ID，vocab百亿级 | token/embedding混合 | RQ-VAE/RQ-Kmeans量化语义ID | embedding/语义空间 | 四层对齐预训练（Token/Item/Relational/User） |
| **候选集** | 无需候选池，直接生成 | 必须有候选池 | 无需ANN索引 | 依赖候选池 | **需候选池（OneReason 召回 → Redis → 与 OneRec 融合）** |
| **延迟** | 高（Decoder自回归） | 低（并行打分） | 中等 | 中等 | **离线 8B 召回 + 在线 0.8B 蒸馏，主链路延迟不受影响** |
| **可控性** | 难调参，输出不稳定 | 易调参，可精确控制 | ID空间有限，可控 | 易调参 | **CoT 可读但业务约束尚未直接写进推理层（论文 §A.4.3 自承）** |
| **广告落地** | 难 | 易（已有成熟体系） | 较难 | 易 | **单点验证（本地生活广告），跨业务迁移性未验证** |
| **冷启动** | 差 | 差 | 优（语义聚合） | 优（世界知识） | 强（语义对齐预训练） |
| **Ground Truth** | 模糊（Next Item不明确） | 清晰（点击/转化） | 模糊 | 清晰 | 多目标RL奖励（Set-wise奖励） |
| **可解释性** | 黑盒 | 黑盒 | 黑盒 | 中等 | **离线 CoT 可读（仅 Slow Pipeline 输出），在线已蒸馏** |

这里有个关键问题知乎上讨论很多：**Ground Truth缺失**。

判别式模型训练信号很清晰——用户点了就是1，没点就是0。但生成式模型的问题在于，你没法说"下一个物品应该是什么"是唯一正确答案。用户行为有多种反馈，每个Item可能满足不同维度的偏好，没有类似NLP的"正确句子"作为标准答案。

这个问题让生成式模型的训练信号天然弱于判别式。

***

## 四、各厂商的战略选择：有人激进，有人务实

这个领域最有意思的地方在于，各家厂商的技术路线选择差异非常大，而且**战略转向**非常明确。

### 快手：从"激进到务实"，OneReason 开辟新维度（**但需要客观评估**）

这是 2026 年 6 月 OneReason 发布后最值得重新审视的章节。之前普遍认为快手是"从生成式转向判别式"——OneRec → UniMixer 的转向叙事。OneReason 让这个判断变得不完整：**快手实际上在同时跑三条独立的技术路线**，而非简单的"转向"。

但需要**客观评估**：OneReason 目前是**单点业务验证**（本地生活广告 10 天 A/B），距离与 OneRec/UniMixer 真正"并行"还有相当距离。

**路线 A：OneRec 系列（生成式）**

快手是生成式推荐的激进推动者，2025年2月发布OneRec，端到端生成+RL偏好对齐，观看时长+1.6%。2025年8月发布 OneRec-V2，Lazy Decoder 架构让 MFU 达到 62%，编码资源从 97.66% 降至 2.34%，计算量下降 94%，训练资源下降 90%，成功扩展至 8B 参数。

**路线 B：UniMixer（判别式 Scaling）**

2026年4月发布的 UniMixer——统一注意力/TokenMixer/FM三种Scaling块的判别式架构。这个转向非常有代表性：从"颠覆式"到"渐进式"。

**路线 C：OneReason（推荐 Reasoning 探索，2026.6 新增）**

OneReason 与 OneRec/UniMixer 的关系需要准确理解：
- **不是 UniMixer 的替代品**——OneReason 走的是 Reasoning 基模路线，UniMixer 走的是判别式 Scaling 路线，二者属于不同范式
- **与生成式 OneRec 系列相互延伸**——OneRec 是"端到端生成"，OneReason 是"先思考再生成"，**但 OneReason-8B 实际只做离线召回，主链路仍由蒸馏版 OneRec + 既有 ranker 承担**
- **是 Agentic Recommender 的潜在前置**——但 Agentic Recommender Harness 是论文 §11 的 **future work**，尚无实验或落地

快手内部目前形成的完整格局：
- **生成式**：OneRec → OneRec-V2（端到端生成 + Lazy Decoder 工程优化，**已在大规模主站全量**）
- **判别式**：UniMixer（统一注意力/TokenMixer/FM 三种 Scaling 块）
- **Reasoning 探索**：OneReason（推荐专属 CoT + GRPO 改进 + Fast-Slow Thinking 双轨部署，**单点业务验证**）
- **平台化**：OneRec → OneLoc（本地生活）→ OneMall（电商+短视频+直播）→ GR4AD（广告推理优化）

**避免"过度归因"的客观描述**：
- OneRec 在快手主站**已全量**（停留时长 +1.6% 是大规模主流量数据）
- UniMixer 是**架构层探索**（论文 arXiv:2604.00590，线上效果数据未公开）
- OneReason **只在本地生活广告 5% × 5% 验证**（曝光 +10.33% 是单实验域、小流量数据）
- 三者**业务权重和成熟度差异很大**，不宜简单并列说"三线并行"

### 字节：全程务实派

字节很有意思，从一开始就没有明显押注生成式路线，而是走了判别式Scaling：

- **HLLM (2024)**：分层LLM推荐，Item LLM + User LLM，7B参数
- **OneTrans (2025.10)**：统一Transformer，per-user GMV +5.68%，WWW 2026接收
- **RankMixer (2025.7)**：Token Mixing + MoE稀疏激活，模型MFU从4.5%提升到45%
- **TokenMixer-Large (2026.2)**：7B在线/15B离线参数，订单+1.66%、GMV+2.98%、广告ADSS+2.0%、直播+1.4%
- **UG-Separation (2026.2)**：用户组分离推理，推理延迟-20%（抖音/头条/穿山甲/千川）

字节的逻辑很清晰：**延迟可控 + 可控性强 + 广告落地无障碍 = 工业落地优先**。TokenMixer-Large的数字也非常实在——订单+1.66%、GMV+2.98%、广告ADSS+2.0%、直播+1.4%，都是可以直接说出口的业务指标。

### Meta：开创者继续深耕

Meta走的是生成式路线，但很有意思的是，2024年2月的GR/HSTU验证了Scaling Law之后，工业界follow最多的其实是**HSTU架构本身**，而不是生成式范式。

2026年2月Meta发布ULTRA-HSTU：18层/d=512/16K序列，5.3×训练+21.4×推理加速（vs vanilla HSTU），生产端消费+互动综合提升4%-8%、topline（DAU/访问量）+0.217%。技术亮点包括：
- AT/MoT动态拓扑
- 混合精度优化（BF16+FP8+INT4）
- 内存优化：7GB→2.3GB/层

Meta的判断是：**生成式的天花板更高，但需要系统-模型协同设计才能弯曲Scaling曲线**。

### 美团：深耕积累

美团的技术路线相对独特，在判别式CTR方向持续深耕：MTGR保留DLRM交叉特征（区别于Meta HSTU），Group-LN统一异构特征空间，动态掩码避免信息穿越。之后又发布了EGA-V2（广告端到端生成），技术积累比较深厚。

**Notice**: MTGR名字带"Generative"，但论文用BCE loss训练、AUC评估，本质上是**判别式CTR精排**，不是生成式。

### 阿里：低调深耕

阿里公开工作不多，但SORT工作（arXiv:2603.03988, Alibaba International Digital Commercial Group）是2026年的重磅炸弹——判别式ranking backbone的系统性改造，围绕电商ranking改造Transformer，提出了request-centric sample organization、local attention、query pruning和生成式预训练。线上A/B测试拿到电商场景**订单+6.35%、买家+5.97%、GMV+5.47%**，同时**延迟-44.67%、吞吐+121.33%**。SORT的意义在于：验证了判别式ranking backbone在生成式one-model升温背景下**依然存在明确且可观的进化空间**。

### 京东：工程派

京东的思路比较务实——不自研大模型，而是专注于推理系统优化。xLLM推理引擎已开源（GitHub: jd-opensource/xllm），xGR系统包括xAttention（共享KV优化）、xBeam（Beam搜索优化）、xSchedule（调度优化）。

### 百度：前沿探索

百度2026年发布GRAB（"Generative Ranking for Ads at Baidu"），提出LLM启发的序列优先CTR预测范式，核心创新是**CamA多通道注意力机制**（Causal Action-aware Multi-channel Attention），CTR+3.49%、收入+3.05%，属于融合多个技术方向的前沿工作。

### Google：学术先驱

Google的贡献主要在学术层面——2023年TIGER论文提出RQ-VAE语义ID，是生成式检索的学术先驱，验证了冷启动和泛化能力。后续快手OneRec等均基于语义ID思路发展。

### 小红书、LinkedIn：各有特色

小红书发布LASER（arXiv:2602.11562, Xiaohongshu/RedNote），全栈长序列优化框架：SeqVault混合DRAM-SSD索引将检索延迟-50%、CPU使用-75%；Segmented Target Attention (STA) + Global Stacked Target Attention (GSTA) 算法。线上A/B测试**ADVV +2.36%、收入 +2.08%**（1亿+DAU），代表长序列全栈优化路线。

LinkedIn发布Feed-SR（arXiv:2602.12354），Transformer-based sequential ranker替代DCNv2，部署服务12亿会员，线上**time spent +2.10%、互动（赞/评论/转发）+3.52%**。论文特别披露他们**评估过多种替代sequential架构和LLM-based排序架构**，最终Feed SR在online metrics和production efficiency上提供了最佳组合——这说明海外大厂在feed ranking中依然高度重视判别式sequential ranking，没有盲目跟进生成式。

***

## 五、为什么业界态度从狂热转向理性观望？

知乎上有个问题被反复讨论：**生成式推荐到底是不是伪范式？**

支持者的核心论点是：

> "生成式推荐不是伪范式，而是未来，这是因为传统推荐范式无法具备scaling law" —— 傅聪Cong

> "同等数据和算力规模下，谁通过生成式范式大幅提点了？生成式不是关键，Scaling Law才是" —— BUGs

质疑者的核心论点是：

> "100%不看好生成式替代当前搜广推的召回-粗排-精排架构。生成式推荐最大的问题是没有ground truth" —— 陈东文

> "字节全程没有明显押注，快手转UniMixer(判别式)，阿里无公开工作，腾讯换赛道" —— 几野

说实话，我觉得两边的观点都有道理。**问题的关键可能不在于"生成式 vs 判别式"，而在于"是否准备好为新范式付出代价"。**

生成式的代价是什么？

1. **延迟成本**：自回归生成Decoder的计算量是判别式的10倍以上
2. **可控性成本**：调参逻辑完全不同，效果不稳定
3. **广告落地成本**：出价/计费逻辑难以融入现有系统
4. **Ground Truth成本**：训练信号弱，需要更多数据和算力弥补

判别式Scaling的代价是什么？

1. **候选集限制**：无法超越候选池上限
2. **意图建模限制**：无法建模真正的用户意图推理
3. **范式天花板**：理论上存在上限

各厂商的选择其实都是在权衡这些代价。

***

## 六、2026年真正靠谱的方向是什么？

综合知乎讨论和论文分析，我认为未来1-2年真正靠谱的方向有以下几个：

### 趋势一：判别式大ranking进入成熟阶段，但正在"Foundation Model化"

这不是在说判别式会消失——恰恰相反，判别式ranker并未迅速失去演进空间。RankMixer、TokenMixer-Large、MSN、SORT、Feed-SR等工作表明，判别式大ranking依然存在明确且可持续的技术空间。

真正发生的变化是：**判别式模型正在越来越像foundation model**：
- 更统一的token接口
- 更深的一体化主干
- 更明确的scaling目标
- 更系统的serving优化

在强实时、高吞吐、严格时延约束的主链路场景中，判别式backbone仍然具备极强的工业竞争力。

### 趋势二：生成式在广告场景落地更快

生成式推荐已经走过仅在retrieval试水的阶段，开始进入主排序、广告与电商推荐链路。标志性工作包括：OneRec、GPR、OneRanker、GR4AD、OxygenREC。

广告天然面临多目标优化问题（点击+转化+价值），页面级生成需求，value alignment与全局收益优化——这些问题与生成式框架的建模方式天然契合。

### 趋势三：Semantic Token正在成为系统级接口

Item不再只是原子化ID，而正在演化为可组合、可迁移、可生成、可索引的token序列。

- 检索不再只依赖ANN索引
- 排序不再只面对固定item embedding
- 多场景推荐可共享更高层级的token空间
- 推理阶段的beam search、trie constraint、prefix constraint将逐渐成为推荐系统的一部分

这个意义已经远超"更有利于冷启动"这一局部优势，而正在成为推荐系统新的基础接口。

### 趋势四：Serving和Inference-time Scaling已成一等公民

过去将线上部署放在最后一节作为工程实现说明；当前的情况已经发生变化。标志性工作包括：UG-Separation、LASER、GR4AD、PROMISE、OxygenREC。

关键洞察是：**训练阶段的模型创新，如果不能转化为在线可控的计算图、存储访问模式与推理策略，就很难构成真正意义上的工业创新**。

Inference-time scaling不再是通用LLM独有的话题——beam设计、path-level reward、dynamic beam serving、用户侧复用、近线reasoning distillation都在成为推荐系统的新战场。

### 趋势五：统一化正在从"统一特征"走向"统一分布、统一目标、统一平台"

当前更激进的目标已经变成：
- 统一特征类型
- 统一场景分布
- 统一任务目标
- 统一训练与部署接口
- 统一推荐、搜索、广告之间的基础表达

标志性工作包括：OneTrans、MixFormer、MDL、MTFM、OneMall、OxygenREC。

### 趋势六：推荐 Reasoning 进入早期探索期（2026.6 起步，**勿过早下"分水岭"结论**）

2026年6月9日快手 OneReason（arXiv:2606.06260）的发布，是 LLM 时代"Reasoning 浪潮"在推荐系统的**首次工业级落地**。这是 LLM 主线"Scaling → Reasoning → Agentic"的第二步迁移——之前 OpenAI o1/DeepSeek R1 已经把"先思考再回答"做成共识范式，让模型在数学、代码、Agent 任务上获得跨代的能力跃迁。OneReason 验证了"推荐场景也能用 CoT + RL 拿到收益"，但**距离"分水岭"还远**。

**为什么说"早期探索期"而不是"分水岭"？**

1. **架构上是工程折中**：OneReason-8B 实际只做**离线召回**（Slow Pipeline，写入 Redis 候选池），主链路仍是「蒸馏版 OneRec + 既有 ranker」，大模型没有进入主排序。论文 §A.4.3 自承 "Fast-Slow 是工程折中，**cannot fully replace real-time retrieval**"。
2. **业务是单点验证**：仅在快手**本地生活广告 5% × 5% × 10 天**验证（曝光 +10.33% / 收入 +8.234%），未在主 feed、电商主站、直播等场景验证，跨业务迁移性未证。
3. **"Thinking 优势"是离线指标**：Pass@4 +13.45% 是**离线 benchmark**，不是线上 A/B 收益。线上 +10.33% 是 Fast + Slow 两条范式**合计**的收益。
4. **"Agentic Recommender Harness"是 future work**：论文 §11 提的未来方向，没有实验或落地。
5. **增量训练代价**：continual pretraining 部分牺牲了指令遵循与通用能力，多业务混部时需警惕。

**潜在价值（需后续验证）**：
- **方法论价值**：四层语义对齐预训练 + R3 CoT + GRPO 改进是系统化方法论
- **业务杠杆真实**：曝光 +10.33% / 收入 +8.234% / ROI > 5 是真实线上收益
- **边际成本可控**：600 旗舰 GPU / 天 / 4 亿用户 ≈ 百微秒级 GPU 时间
- **范式 A/B 价值**：可以单独上线其中一种（只 Slow 或只 Fast），降低初次落地风险

**未来方向（论文 §11 提的 future work，未充分论证）**：
- 打造 **Agentic Recommender Harness**——规划、工具调用、长程对话推荐
- 0.8B 级别小模型做真正的实时召回（替代当前的离线 8B）
- 业务价值（eCPM、ROI）直接灌进 decoding objective

***

## 七、结语：未来1-2年的分水岭在哪？

如果让我预测，未来1-2年真正拉开差距的因素，大概率不会只是参数规模本身，而会是三类"统一接口"谁先成形：

**第一层：统一的token接口**
item、广告、内容、地理、场景、任务等对象，是否能够被映射到更稳定、更可扩展的语义空间中。

**第二层：统一的backbone接口**
无论是判别式统一backbone还是生成式one-model，核心竞争都在于：究竟由谁来承接sequence、feature、scenario、task与value这些异质信息。

**第三层：统一的在线推理接口**
训练、蒸馏、缓存、复用、beam search、近线reasoning之间能否形成闭环，将决定"论文中的大模型"能否转化为"线上可持续迭代的平台能力"。

**总的来说，推荐系统的大模型化，已经从"模型更大"演进到"接口重写"。**

**2026 年 6 月新增的"探索性第四层"**：推荐基模的下一个分水岭**可能**在于：模型能否在生成推荐之前进行显式的、可解释的、可干预的推理？业务约束能否直接写在推理层？策略迭代节奏能否从周级降到天级？这些是**OneReason 想要回答但目前只是初步验证**的问题——`本地生活广告 +10.33%` 是单点数据，距离真正的"统一 Reasoning 接口"还有相当距离。本文不应在 2026 年 6 月就把"分水岭"作为定论。

两条技术路线最终可能会走向融合，而不是非此即彼。生成式负责"创意生成和全局优化"，判别式负责"实时推理和精确排序"——这个分工可能比"谁取代谁"更符合工业实际。

最后说一句：知乎上的讨论确实很有价值，但论文数字和线上效果之间往往存在巨大鸿沟。看到"12.4%"和"+1.6%"，要理解它们的scale完全不同。HSTU的12.4%是**线上A/B生产测试**的端到端提升（同一篇论文在公共数据集上还有NDCG +65.8%的提升），快手的+1.6%是快手主站全量的watch-time提升——两个数字的可信度都比较高，但它们的量级和意义完全不同。

判断一个技术方向是否靠谱，最终还是要看**它在你的具体业务场景下是否可行**，而不是论文上的数字多漂亮。

***

## 技术演进脉络图

```
2023年
│
├── Google TIGER
│   └── 语义ID范式, RQ-VAE, 生成式检索先驱
│       │
│       └──→ 快手OneRec (2025.2) - 端到端生成式
│
2024年2月
│
└── Meta GR/HSTU ─ ICML 2024
    │   开创者, 验证Scaling Law, +12.4%
    │
    ├──→ 美团MTGR (2025.8) - 保留交叉特征判别式CTR
    │    └──→ EGA-V2 - 广告端到端生成
    │
    ├──→ 小红书GenRank (2025.5) - 生成式重排
    │
    ├──→ 字节HLLM (2024) - 分层LLM
    │    └──→ 字节OneTrans (2025.10) - 统一Transformer
    │    └──→ 字节RankMixer/TokenMixer (2025-2026) - 判别式Scaling
    │         └──→ TokenMixer-Large (2026.2) - 7B/15B参数
    │              └──→ UG-Separation (2026.2) - 推理优化-20%
    │
    └──→ 京东xLLM (2025.10) - 推理引擎优化
         └──→ xGR系统优化

2026年2月: 新一轮技术突破
│
├── Meta ULTRA-HSTU
│    │  18层/d=512/16K, 5.3×训练+21.4×推理, 消费+互动+4%-8%
│    │
│    └──→ VLM - 晚实例化数据基础设施
│
├── 字节TokenMixer-Large
│    │  7B/15B参数, mixing-and-reverting, Sparse MoE
│    │
│    └──→ UG-Separation - 用户组分离推理
│
├── 快手UniMixer
│    │  统一注意力/TokenMixer/FM三大架构
│    │
│    └──→ 统一Scaling理论框架
│
└── LLM推荐Scaling Law
     │  首个LLM推荐Scaling Law, 合成数据
     │
     └──→ 指导LLM推荐预训练

技术路线分化 (2025.6后)
│
├── 生成式路线: Meta ULTRA-HSTU持续深耕
│   └──→ 18层/d=512/16K, 动态拓扑, 混合精度
│
└── 判别式Scaling路线: 字节TokenMixer-Large主导
     ├──→ 7B/15B参数Scaling
     └──→ UG-Separation推理优化

2026.6 推荐 Reasoning 路线（**早期探索，单点业务验证**）
│
└── 快手OneReason (arXiv:2606.06260)
     │  578B token 四层预训练, 推荐专属CoT, GRPO改进, Fast-Slow Thinking
     │  **注：8B本体只做离线召回，主链路仍是蒸馏版 OneRec + 既有 ranker**
     │
     ├──→ 0.8B参数基座 (Qwen3-8B基础上, 即将开源)
     ├──→ R3 CoT三段式 (Persona Abstraction → Interest Expansion → Transition Inference)
     ├──→ "先专后合"训练链路 (Specialize-then-Unify)
     ├──→ Fast-Slow Thinking部署架构
     │     ├── Slow Thinking (近线): 完整 OneReason 8B → Redis 候选池
     │     └── Fast Thinking (在线): OneRec + Thinking Token (蒸馏版0.8B)
     ├──→ 本地生活广告 5%×5%×10天 A/B: 曝光+10.33% / 收入+8.234% (Fast+Slow合计)
     └──→ Agentic Recommender Harness (论文future work, 尚未落地)

下一代演进目标 (2026 H2+, **论文 §11 提的 future work**)
│
└── Agentic Recommender
     │  规划 + 工具调用 + 长程对话推荐
     │  千人千策 替代 千人一面的固定流水线
     │
     └──→ OneReason 是潜在前置地基（**论文未充分论证，仅为推断**）
```

***

## 主要论文列表

| 论文/模型名称 | arXiv ID / 链接 | 厂商 | 发表年份 |
|-------------|----------------|------|---------|
| TIGER | [2305.05065](https://arxiv.org/abs/2305.05065) | Google | 2023 |
| GR/HSTU | [2402.17152](https://arxiv.org/abs/2402.17152) | Meta | 2024 |
| HLLM | [2409.12740](https://arxiv.org/abs/2409.12740) | 字节 | 2024 |
| OneRec | [2502.18965](https://arxiv.org/abs/2502.18965) | 快手 | 2025 |
| GenRank | [2505.04180](https://arxiv.org/abs/2505.04180) | 小红书 | 2025 |
| MTGR | [2505.18654](https://arxiv.org/abs/2505.18654) | 美团 | 2025 |
| EGA-V2 | [2505.17549](https://arxiv.org/abs/2505.17549) | 美团 | 2025 |
| xLLM | [2510.14686](https://arxiv.org/abs/2510.14686) | 京东 | 2025 |
| OneTrans | [2510.26104](https://arxiv.org/abs/2510.26104) | 字节 | 2025 |
| OxygenREC | [2512.22386](https://arxiv.org/abs/2512.22386) | 京东 | 2025 |
| GRAB | [2602.01865](https://arxiv.org/abs/2602.01865) | 百度 | 2026 |
| LLM推荐Scaling Law | [2602.07298](https://arxiv.org/abs/2602.07298) | - | 2026 |
| TokenMixer-Large | [2602.06563](https://arxiv.org/abs/2602.06563) | 字节 | 2026 |
| UG-Separation | [2602.10455](https://arxiv.org/abs/2602.10455) | 字节 | 2026 |
| LASER | [2602.11562](https://arxiv.org/abs/2602.11562) | 小红书 | 2026 |
| Feed-SR | [2602.12354](https://arxiv.org/abs/2602.12354) | LinkedIn | 2026 |
| ULTRA-HSTU | [2602.16986](https://arxiv.org/abs/2602.16986) | Meta | 2026 |
| VLM (Late Materialization) | [2604.24806](https://arxiv.org/abs/2604.24806) | Meta | 2026 |
| UniMixer | [2604.00590](https://arxiv.org/abs/2604.00590) | 快手 | 2026 |
| SORT | [2603.03988](https://arxiv.org/abs/2603.03988) | 阿里 | 2026 |
| RankMixer | [2507.15551](https://arxiv.org/abs/2507.15551) | 字节 | 2025 |
| **OneRec-Think** | **[2510.11639](https://arxiv.org/abs/2510.11639)** | **快手** | **2025**（arXiv编号已修正） |
| **R4ec**（Actor+Reflection 反思机制） | **[2507.17249](https://arxiv.org/abs/2507.17249)** / **RecSys 2025** | **快手+中科院** | **2025.07** |
| **OpenOneRec** | **[2512.24762](https://arxiv.org/abs/2512.24762)** | **快手** | **2026**（arXiv编号已修正） |
| **OneReason** | **[2606.06260](https://arxiv.org/abs/2606.06260)** | **快手** | **2026** <span class="badge badge-new">NEW</span> |
| **ReaRec** | 待补充（人大 x 阿里） | 人大 + 阿里 | 2025.04 |
| **OnePiece** | 待补充（傅聪 x 人大，Shopee Search 全流量） | 人大 + Shopee | 2025.09 |
| **Why Thinking Hurts** | **[2506.00007](https://arxiv.org/abs/2506.00007)** | 独立 | 2026 |
| **CQ-SID** | **[2605.14434](https://arxiv.org/abs/2605.14434)** | 阿里/天猫 | 2026.05 |
| **UniVA** | **[2605.05803](https://arxiv.org/abs/2605.05803)** | 腾讯/微信视频号广告 | 2026.05 |

**说明**：
- OneTrans 已录用为 **WWW 2026** 论文（per-user GMV +5.68%）
- LLM推荐Scaling Law 已录用为 **ICML 2026** 论文（合成数据训练SasRec recall@100 +130% vs 真实数据）
- GR/HSTU 发表为 **ICML 2024** 论文（A/B +12.4%, NDCG +65.8%）
- MTGR 发表为 **CIKM 2025** 论文（DOI: 10.1145/3746252.3761565）
- **R4ec 发表为 RecSys 2025 论文**：Actor+Reflection 双 LLM 迭代反思，**收入 +2.2%**（10% 流量 × 14 天），对长尾数据效果更明显
- **OneRec-Think（In-Text Reasoning for Generative Recommendation）**：停留时长 +0.159%（1.29% 流量），代码：github.com/wangshy31/OneRec-Think
- **OpenOneRec（OneRec-Foundation）**：1.7B/8B 双版本 + RecIF-Bench 评测 + 公开 9600 万条交互；HuggingFace: OpenOneRec；GitHub: Kuaishou-OneRec/OpenOneRec；Amazon 10 域 Recall@10 +26.8%
- **OneReason 是快手 2026年6月9日发布的推荐 Reasoning 探索工作**：578B token 四层预训练 + 推荐专属 CoT + GRPO 改进 + Fast-Slow Thinking 双轨部署。**线上 A/B 是 Fast + Slow 两条范式合计收益**（曝光 +10.33% / 收入 +8.234% / ROI > 5），仅本地生活广告 5%×5%×10天单点验证。**OneReason-8B 本体只做离线召回，主链路仍是蒸馏版 OneRec + 既有 ranker**。"Thinking 模式领先 +13.45%" 是离线 Pass@4 指标，不是 A/B 收益。HuggingFace: OpenOneRec/OneReason-0.8B-pretrain
- **ReaRec（人大 x 阿里，2025.04）**：固定 2 步隐式推理下 +3.51% 平均延迟 +7.49% 推荐表现；Yelp 数据集最优推理步数分析显示 30-50% 性能提升空间
- **OnePiece（傅聪 x 人大，2025.09）**：Shopee Search 主场景全流量生效，登顶 HuggingFace Daily Paper 前三
- **CQ-SID（阿里/天猫 App 搜索，2026.05）**：Category-Query constrained SID + EG-GRPO，把生成式召回当作"补充通道"而非端到端替换
- **UniVA（腾讯/微信视频号广告，2026.05）**：Unified Value Alignment for Generative Recommendation，eCPM-aware RL
- **Why Thinking Hurts（2026）**：诊断"思考模式反而 hurts" 根因——**归因到"来自通用子空间的文本惯性"**。OneReason §10 Related Works 显式引用此工作
- 其余均为 arXiv 预印本

***

### 参考资料

[1] [知乎话题：生成式推荐是未来还是伪范式](https://www.zhihu.com/question/1966168036277977152)

[2] [知乎话题：生成式推荐会成为下一代推荐系统的范式吗](https://www.zhihu.com/question/1916207872066430537)

[3] [知乎专栏：OneReason:生成式推荐学会了「先想再推」](https://zhuanlan.zhihu.com/p/2047245741110867954)（浅梦，2026.06）

[4] [知乎专栏：推荐系统到底存不存在"思维链"?](https://zhuanlan.zhihu.com/p/2047425388184445133)（猫的薛定谔，101 票，2026.06）

[5] [知乎专栏：Du论文-2026-06-快手-OneReason](https://zhuanlan.zhihu.com/p/2048412422847870910)（最详细中文译解版，74KB）

[6] [知乎专栏：快手向左，阿里向右｜2025大模型融入推荐系统的两条路线深度解读](https://zhuanlan.zhihu.com/p/2032858501400221432)（GuoXun，128 票）

[7] [知乎专栏：快手[R4ec]：通过迭代反思与改进机制](https://zhuanlan.zhihu.com/p/1938044297418109393)（40 票）

[8] [知乎专栏：阿里向搜索，腾讯向广告｜2026年5月生成式SID范式的两条工业答卷](https://zhuanlan.zhihu.com/p/2039425526247510743)（GuoXun，59 票）

[9] [知乎专栏：「人大 x 阿里」Think Before Recommend: 释放序列推荐模型隐藏的推理能力](https://zhuanlan.zhihu.com/p/1890431761223222035)（ReaRec，41 票）

[10] [知乎专栏：OnePiece：首个全面落地推理能力的工业级生成式搜索框架！](https://zhuanlan.zhihu.com/p/1990378803617371940)（傅聪Cong，18 票）

[11] [知乎专栏：「快手」ONEREC-THINK: In-Text Reasoning for Generative Recommendation](https://zhuanlan.zhihu.com/p/1961917655096201392)（猫的薛定谔，52 票）

[12] [知乎专栏：(开源)细读快手OneRec-Think技术分享](https://zhuanlan.zhihu.com/p/1963536264767738349)（州懂，116 票）

[13] [知乎专栏：(快手,生成式推荐,GR) OpenOneRec论文解读](https://zhuanlan.zhihu.com/p/1990434758312863235)（鹦鹉丛中笑，55 票）

> PS: 这篇文章基于知乎 126 条回答 + 13 篇知乎深度专栏 + arXiv 最新论文综合分析写成。OneReason / 推荐 Reasoning 部分已基于 2026.06 知乎社区的 13 篇相关深度文章做了二次扩展，覆盖：
> - **R0-R3 能力栈**（Perception / Derivation / Evolution / Recommendation 4 层诊断架构）
> - **路线五的源流**（R4ec → OneRec-Think → OpenOneRec → OneReason 四步迭代）
> - **同期业界对推荐 Reasoning 的多路探索**（ReaRec / OnePiece / AMEN / CQ-SID / UniVA）
> - **"为什么思考会伤害"的根因诊断**（Why Thinking Hurts 论文 + 知乎三派立场）
> - **GRPO 推荐 RL 三大机制**（两阶段 Rollout / 阶段感知裁剪 / 负样本降权）
> - **RFT vs MOPD 的能力分工**（reverse-KL mode-seeking 性质）
> - **CoT 似然增益（ΔLL）的实证**（SFT 全负 → RFT 全正）
> - **arXiv 编号修正**（OneRec-Think: 2510.11639, OpenOneRec: 2512.24762）
>
> 如果你也是推荐系统从业者，欢迎在评论区分享你的看法。