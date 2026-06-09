# 推荐技术路线深度分析——从生成式到Reasoning，2026年我们怎么看？

> 编辑于 2026年6月・北京
> 数据来源: 知乎话题（126条回答）+ arXiv最新论文 + 快手OneReason技术报告（arXiv:2606.06260）

***

> PS: 这是一篇综合分析，素材来自知乎上126条回答、arXiv最新论文，以及2026年6月9日快手刚刚发布的 OneReason 技术报告（arXiv:2606.06260）。写这篇文章的目的，是想系统梳理一下这两年推荐算法领域最热闹的方向——从生成式推荐到刚刚兴起的推荐 Reasoning——到底在发生什么。不保证全面，但保证有态度。

***

## 导语

最近这两年，如果你关注推荐系统领域的技术演进，有两个词一定绕不开：**生成式推荐**（Generative Recommendation）和**推荐 Reasoning**（Reasoning Recommendation）。

从2024年2月Meta发布GR/HSTU论文开始，工业界对生成式方向的热情一度非常高。各大厂纷纷跟进，快手、字节、美团、腾讯轮番发论文，仿佛一夜之间推荐系统就要被"重写"了。但到了2026年再看，业界的态度已经从狂热转向**理性观望**——有人坚定看好，有人坚决不看好，有人悄悄转向了另一条路。

而2026年6月9日快手发布的 OneReason（arXiv:2606.06260）打破了这一沉寂——首次在工业级推荐场景中验证了"先思考再推荐"（Thinking-then-Recommending）的可行性，把 LLM 的 Reasoning 范式成功迁移到推荐基模。本地生活广告 A/B 实验曝光 +10.33%、收入 +8.234%，ROI > 5，年化数亿元商业增量。这意味着推荐系统的下一个分水岭已经从"Scaling"延伸到"Reasoning"。

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

### 路线五：推荐 Reasoning（2026.6 新增）

2026年6月9日快手发布的 **OneReason**（arXiv:2606.06260）是这一路线的开山之作。其核心理念是：把 LLM 时代"先思考再回答"的 Reasoning 范式（OpenAI o1、DeepSeek R1 范式）迁移到推荐系统，让推荐基模在生成推荐前先显式推理"为什么推这条"。

**核心洞察**：推荐 Reasoning 不是数学推理那种"演绎式"，而是"溯因推理"——用户兴趣不可见，模型需要从长期、嘈杂、跨域的行为序列中反推潜在兴趣，理解兴趣演化，并判断候选物料为何适合当前上下文。

代表工作：
- **快手 OneReason (2026.6)**：578B token 四层预训练 + 推荐专属 CoT（Persona Abstraction → Interest Expansion → Transition Inference 三段式）+ GRPO 三方面改进（两阶段轨迹生成 + Set-wise奖励 + 优化稳定策略）+ "先专后合"训练链路 + Fast-Slow Thinking 部署架构。本地生活广告 10 天 A/B 曝光 +10.33%、收入 +8.234%

技术架构：

```
四层预训练数据（578B token） → 预训练基座 → 推荐专属CoT SFT → GRPO改进RL → Fast-Slow Thinking 部署
                                                          ↓
                  Slow Thinking（近线）：完整 OneReason 模型 → 离线召回+推理链生成
                  Fast Thinking（在线）：OneReason for OneRec（蒸馏/剪枝）→ 在线快思考服务
                                                          ↓
                                                    排序模型融合
```

**与前四条路线的本质差异**：

| 关键差异 | 说明 |
|---------|------|
| **可解释性** | CoT 推理链显式可读，业务约束直接写在推理层，策略迭代节奏从周级降到天级 |
| **溯因 vs 演绎** | 推荐 Reasoning 是溯因推理（从行为反推兴趣），不是数学推理的演绎推理 |
| **多业务统一** | 一个人懂物料语义的基模 + 一段业务说明即可跨域出方案，无需为每个域从零搭召回排序栈 |
| **Agentic前置** | 是 Agentic RecSys 的前置地基，规划、工具调用、长程对话推荐都需要 Reasoning 基模在底层托住 |

**核心收益**：

- 业务侧：本地生活广告 10 天 A/B（5% vs 5%），曝光 +10.332%、收入 +8.234%，ROI > 5，年化数亿元
- 推理模式：Pass@4 上 thinking 平均领先 non-thinking +13.45%，首次在工业级推荐让"先思考"稳定成为正资产
- 通用能力：MMLU-pro、GPQA-Diamond 基本保留 Qwen3-8B 原始水平，没有"训练推荐训坏通用能力"
- CoT 能力内化：引入 CoT 监督不仅提升 thinking，还能反哺 non-thinking 性能

**关键技术细节**：

1. **四层预训练数据架构**（解决 item-text 语义割裂）：
   - Token 粒度：子 Token 拆解组合、单 Token 释义、前缀语义预测
   - Item 粒度：物料描述粗粒化 + 多视角 Item QA
   - Relational 粒度："物品→兴趣说明文本→后续物品"链路数据
   - User 粒度：分域分组、全时序穿插、Item 替换为文本

2. **三阶段预训练策略**：
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

**未来方向（Agentic Recommender）**：

OneRec 团队下一步计划：打造 **Agentic Recommender Harness**，让推荐基模具备规划与工具调用能力，逐步驱动推荐系统向 Agentic 架构演进。从"千人一面的固定流水线"走向"千人千策、能规划、能用工具、能多轮对话"。

***

## 三、五条路线的核心差异

| 维度 | 生成式GR | 判别式Scaling | 语义ID检索 | LLM增强 | Reasoning推荐 |
|------|---------|--------------|-----------|---------|---------------|
| **建模方式** | 自回归生成Next Item | 逐一打分排序 | Semantic ID量化+生成 | LLM特征增强 | 先思考再推荐（CoT+RL） |
| **编码空间** | 原子化Item ID，vocab百亿级 | token/embedding混合 | RQ-VAE/RQ-Kmeans量化语义ID | embedding/语义空间 | 四层对齐预训练（Token/Item/Relational/User） |
| **候选集** | 无需候选池，直接生成 | 必须有候选池 | 无需ANN索引 | 依赖候选池 | 溯因推理生成，无需候选池 |
| **延迟** | 高（Decoder自回归） | 低（并行打分） | 中等 | 中等 | Fast-Slow双轨（近线+在线） |
| **可控性** | 难调参，输出不稳定 | 易调参，可精确控制 | ID空间有限，可控 | 易调参 | 可解释CoT，业务约束可写在推理层 |
| **广告落地** | 难 | 易（已有成熟体系） | 较难 | 易 | 强（OneReason本地生活广告+8.234%收入） |
| **冷启动** | 差 | 差 | 优（语义聚合） | 优（世界知识） | 强（语义对齐预训练） |
| **Ground Truth** | 模糊（Next Item不明确） | 清晰（点击/转化） | 模糊 | 清晰 | 多目标RL奖励（Set-wise奖励） |
| **可解释性** | 黑盒 | 黑盒 | 黑盒 | 中等 | 强（CoT推理链显式可读） |

这里有个关键问题知乎上讨论很多：**Ground Truth缺失**。

判别式模型训练信号很清晰——用户点了就是1，没点就是0。但生成式模型的问题在于，你没法说"下一个物品应该是什么"是唯一正确答案。用户行为有多种反馈，每个Item可能满足不同维度的偏好，没有类似NLP的"正确句子"作为标准答案。

这个问题让生成式模型的训练信号天然弱于判别式。

***

## 四、各厂商的战略选择：有人激进，有人务实

这个领域最有意思的地方在于，各家厂商的技术路线选择差异非常大，而且**战略转向**非常明确。

### 快手：三线并行，不止是从生成式到判别式

这是 2026 年 6 月 OneReason 发布后最值得重新审视的章节。之前普遍认为快手是"从生成式转向判别式"——OneRec → UniMixer 的转向叙事。但 OneReason 让这个判断变得不完整：**快手实际上在同时跑三条独立的技术路线**，而非简单的"转向"。

**路线 A：OneRec 系列（生成式）**

快手是生成式推荐的激进推动者，2025年2月发布OneRec，端到端生成+RL偏好对齐，观看时长+1.6%。2025年8月发布 OneRec-V2，Lazy Decoder 架构让 MFU 达到 62%，编码资源从 97.66% 降至 2.34%，计算量下降 94%，训练资源下降 90%，成功扩展至 8B 参数。

**路线 B：UniMixer（判别式 Scaling）**

2026年4月发布的 UniMixer——统一注意力/TokenMixer/FM三种Scaling块的判别式架构。这个转向非常有代表性：从"颠覆式"到"渐进式"。

**路线 C：OneReason（推荐 Reasoning，2026.6 全新开辟）**

这是 OneReason 的关键意义——**它不是 UniMixer 的替代，而是与 OneRec/UniMixer 并行的第三条战线**。OneRec 团队在生成式 OneRec 系列之外，开辟了 Reasoning 新维度，与判别式 UniMixer 路线相互独立。

**OneReason 的特殊定位**：
- 与判别式 UniMixer 并不冲突——OneReason 走的是 Reasoning 基模路线，UniMixer 走的是判别式 Scaling 路线
- 与生成式 OneRec 系列相互延伸——OneRec 是"端到端生成"，OneReason 是"先思考再生成"，是生成式路线的升级而非替代
- 是 Agentic Recommender 的前置地基——规划、工具调用、长程对话推荐都需要 Reasoning 基模在底层托住

快手内部目前形成的完整格局：
- **生成式**：OneRec → OneRec-V2（端到端生成 + Lazy Decoder 工程优化）
- **判别式**：UniMixer（统一注意力/TokenMixer/FM 三种 Scaling 块）
- **Reasoning**：OneReason（推荐专属 CoT + GRPO 改进 + Fast-Slow Thinking 部署）
- **平台化**：OneRec → OneLoc（本地生活）→ OneMall（电商+短视频+直播）→ GR4AD（广告推理优化）

这个"三线并行"的格局，让快手成为推荐系统领域最完整的技术布局者。**简单的"从激进到务实"叙事，已经无法描述 2026 年 6 月之后的快手战略——更准确的描述是"激进-务实-Reasoning 三轨推进"**。

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

### 趋势六：推荐 Reasoning 成为下一个分水岭（2026.6 新增）

2026年6月9日快手 OneReason（arXiv:2606.06260）的发布，标志着 LLM 时代"Reasoning 浪潮"在推荐系统的正式落地。这是 LLM 主线"Scaling → Reasoning → Agentic"的第二步迁移——之前 OpenAI o1/DeepSeek R1 已经把"先思考再回答"做成共识范式，让模型在数学、代码、Agent 任务上获得跨代的能力跃迁，OneReason 让推荐系统也具备同等能力。

**为什么 Reasoning 是下一个分水岭？**

推荐系统过去十年靠 Scaling 拿到的红利，下一阶段同样需要 Reasoning 来释放新的增长曲线。核心判断有三点：

1. **推荐天然是"溯因推理"**：用户行为是"果"，潜在意图是"因"——从嘈杂、跨域、跨时段的行为序列里反推某个具体物料"为什么适合此刻"，本质是溯因推理。统计模型记得住"看过A的人也看B"，但回答不了用户多跳因果链；尤其在冷启用户、新物料、长尾品类、跨域迁移这些行为信号天然稀疏的场景里。
2. **从"黑盒打分"到"可解释、可干预的认知过程"**：业务杠杆显著放大——推理链直接读出"为什么推这条"，业务约束直接写在推理层，策略迭代节奏从周级降到天级；新业务接入也不再需要为每个域从零搭召回排序栈。
3. **Agentic RecSys 的前置地基**：推荐的下一站，是从"千人一面的固定流水线"走向"千人千策、能规划、能用工具、能多轮对话"的 Agentic 推荐系统。规划、工具调用、长程对话推荐这些能力的前提，都需要一个懂物料语义、有推理能力、能稳定指令遵循的基模在底层托住。

**关键技术判断**：

- **推荐 Reasoning ≠ LLM Reasoning 简单复制**：itemic token 与自然语言的语义对齐、推荐专属的归纳/溯因/演绎 CoT 结构、GRPO 在稀疏奖励下的稳定优化——每一层都需要重新设计
- **RL 是解锁推理收益的必备环节**：SFT 后 Delta LL 全为负（CoT 分散注意力），RFT 后 Delta LL 全转正（CoT 真正辅助决策）——只做 SFT 反而会"过度思考损害性能"
- **Fast-Slow Thinking 双轨部署**：近线 Slow Thinking（完整 OneReason）+ 在线 Fast Thinking（蒸馏/剪枝版 OneReason for OneRec）+ 排序融合——兼顾推理深度和实时性
- **跨域统一**：一个人懂物料语义的基模 + 一段业务说明即可跨域出方案，不再为每个域搭独立栈

**业务验证**：快手本地生活广告 10 天 A/B（5% vs 5%），曝光 +10.332%、收入 +8.234%，ROI > 5，年化数亿元商业增量。这说明 OneReason 不只是离线 benchmark 上的探索，也具备在严格延迟与成本约束下进入工业系统的可行性。

**未来方向**：打造 Agentic Recommender Harness，让推荐基模具备规划与工具调用能力，逐步驱动推荐系统向 Agentic 架构演进——从"千人一面的固定流水线"走向"千人千策、能规划、能用工具、能多轮对话"。

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

**2026 年 6 月新增的第四层：统一的 Reasoning 接口**。OneReason 的发布证明，推荐基模的下一个分水岭不在于"参数规模"或"推理速度"，而在于：模型能否在生成推荐之前进行显式的、可解释的、可干预的推理？业务约束能否直接写在推理层？策略迭代节奏能否从周级降到天级？这是 LLM 时代 Reasoning 浪潮在推荐系统的具象化——也是从"匹配"到"生成"到"Reasoning"三阶段进化的最后一公里。

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

2026.6 推荐 Reasoning 路线
│
└── 快手OneReason (arXiv:2606.06260)
     │  578B token 四层预训练, 推荐专属CoT, GRPO改进, Fast-Slow Thinking
     │
     ├──→ 0.8B参数基座 (Qwen3-8B基础上)
     ├──→ R3 CoT三段式 (Persona Abstraction → Interest Expansion → Transition Inference)
     ├──→ "先专后合"训练链路 (Specialize-then-Unify)
     ├──→ Fast-Slow Thinking部署架构
     │     ├── Slow Thinking (近线): 完整 OneReason
     │     └── Fast Thinking (在线): OneReason for OneRec (蒸馏版本)
     └──→ Agentic Recommender Harness (下一站演进方向)

下一代演进目标 (2026 H2)
│
└── Agentic Recommender
     │  规划 + 工具调用 + 长程对话推荐
     │  千人千策 替代 千人一面的固定流水线
     │
     └──→ OneReason 是前置地基
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
| **OneReason** | **[2606.06260](https://arxiv.org/abs/2606.06260)** | **快手** | **2026** <span class="badge badge-new">NEW</span> |
| OpenOneRec | 待补充 | 快手 | 2026 |
| OneRec-Think | 待补充 | 快手 | 2026 |

**说明**：
- OneTrans 已录用为 **WWW 2026** 论文（per-user GMV +5.68%）
- LLM推荐Scaling Law 已录用为 **ICML 2026** 论文（合成数据训练SasRec recall@100 +130% vs 真实数据）
- GR/HSTU 发表为 **ICML 2024** 论文（A/B +12.4%, NDCG +65.8%）
- MTGR 发表为 **CIKM 2025** 论文（DOI: 10.1145/3746252.3761565）
- **OneReason 是快手 2026年6月9日发布的推荐 Reasoning 工作**：578B token 四层预训练 + 推荐专属 CoT + GRPO 改进 + Fast-Slow Thinking 部署架构。本地生活广告 10 天 A/B 曝光 +10.33%、收入 +8.234%。HuggingFace: OpenOneRec/OneReason-0.8B-pretrain
- 其余均为 arXiv 预印本

***

### 参考资料

[1] [知乎话题：生成式推荐是未来还是伪范式](https://www.zhihu.com/question/1966168036277977152)

[2] [知乎话题：生成式推荐会成为下一代推荐系统的范式吗](https://www.zhihu.com/question/1916207872066430537)

> PS: 这篇文章基于知乎126条回答和arXiv最新论文综合分析写成。如果你也是推荐系统从业者，欢迎在评论区分享你的看法。