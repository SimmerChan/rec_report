# 推荐算法演进与生成式推荐 - 团队汇报演讲稿

> **演讲时长**: 约30分钟 | **建议语速**: 中等偏慢，每分钟约180字

---

## 演讲时间分配

| 章节 | 内容 | 时长 |
|------|------|------|
| 开场 | 汇报背景与Agenda | 1分钟 |
| 第一部分 | 为什么推荐系统正在经历范式转变 | 4分钟 |
| 第二部分 | 推荐算法演进全景图 | 4分钟 |
| 第三部分 | 生成式推荐核心技术解析 | 7分钟 |
| 第四部分 | 大厂技术路线对比 | 5分钟 |
| 第五部分 | 工业落地挑战与业界共识 | 4分钟 |
| 第六部分 | 昇腾推荐团队行动建议 | 5分钟 |
| **合计** | | **30分钟** |

---

## 开场 (1分钟)

各位同事好，今天给大家汇报推荐算法领域正在发生的重要技术演进。

这次调研的背景是：从2024年开始，以Meta GR为代表的生成式推荐工作在工业界引起广泛关注。我们需要搞清楚两个问题：第一，这一波技术浪潮到底在发生什么，有哪些关键技术节点；第二，对于昇腾的推荐业务来说，我们应该如何参与和布局。

今天的汇报分为六个部分：首先看为什么推荐系统正在经历范式转变，然后是算法演进全景图，接着深入解析生成式推荐的核心技术，再看各大厂的技术路线对比，然后是工业落地挑战和业界共识，最后是昇腾团队的行动建议。

---

## 第一部分：为什么推荐系统正在经历范式转变 (4分钟)

**三大痛点与LLM破局**

推荐系统当前面临三大核心痛点。

**第一，优化边际效益递减。** 从DIN到DIEN到SIM，模型越复杂效果提升越小。DIN引入Attention，DIEN加入兴趣演化网络，MIMN用Memory突破超长序列，SIM在阿里生产环境实现CTR加7.1%。但这是公开论文中可查的最大单次提升，之后再无显著突破。这说明单靠模型结构升级，边际收益已接近天花板。

**第二，级联架构的误差传播。** 传统推荐系统是候选池一亿级到召回万级到粗排千级到精排百级到重排十级。各模块通信和缓存代价超过50%，误差逐级放大，召回漏掉的item精排再强也无法挽回。

**第三，GPU资源严重浪费。** 传统CTR模型大量算子是memory-bound的Embedding Lookup，H100实际MFU只有约5%。Kunlun论文优化到37%，OneRec-V2的Lazy Decoder架构达到了62%。

**LLM带来了破局契机。** 四个关键突破同时到来：一是Scaling Law，模型规模扩大可预测提升效果；二是长序列建模能力，自注意力天然适合处理用户行为序列；三是世界知识注入，预训练语料蕴含跨领域多模态知识；四是RL对齐，链式推理能力涌现。

标志性事件是2024年Meta GR论文首次提出生成式推荐范式，Facebook主APP实现CTR加12.4%；2025年快手OneRec全量上线，停留时长提升1.6%，核心指标100%正向。

---

## 第二部分：推荐算法演进全景图 (4分钟)

**四代算法分类与三条演进主线**

推荐算法发展经历了四个时代。

**2003到2015是协同过滤时代，** ItemCF、UserCF、SVD、ALS、Swing，核心问题是数据稀疏、冷启动难解，技术本质是统计共现频率，ID为王，无语义信息。

**2015到2019是深度学习时代，** YouTube DNN、Wide&Deep、DeepFM、DCN引入Embedding加MLP架构，自动特征交叉，端到端学习。

**2019到2023是序列建模时代，** DIN、DIEN、SIM、MIMN、BST、BERT4Rec用Attention建模用户行为序列依赖，用户兴趣提取显式化。

**2024年到现在是生成式时代，** Meta GR、OneRec、RankMixer、Ultra-HSTU用NTP Loss替代CTR Loss，Semantic ID实现端到端生成，Scaling Law驱动范式转变。

**三条演进主线相互交织。**

第一条是从特征工程到端到端学习：手工特征FTRL、FM到自动交叉DeepFM、DCN，到端到端序列DIN、SIM，再到LLM语义Embedding。

第二条是序列建模升级：从RNN到Transformer，从短期到长期，从固定窗口到超长序列，再到Mamba线性注意力。

第三条是从判别式到生成式：从ID Embedding查找到Semantic ID量化到Token-Mixing建模，再到端到端LLM生成推荐。

**关键演进节点记住三个时间点。** 2023年Google TIGER首创生成式召回，2024年Meta GR开山证明Scaling Law存在，2025到2026年工业级落地，OneRec、RankMixer、TokenMixer-Large先后上线。

---

## 第三部分：生成式推荐核心技术解析 (7分钟)

**判别式与生成式的本质区别**

先搞清楚判别式和生成式的本质区别。

判别式建模P(click|user, item)，在封闭候选集中逐一打分排序，本质是从已知候选中挑最优。

生成式建模Generate(user_history)，给出用户历史序列直接生成推荐列表，本质是自回归预测下一个token，可以生成全新item，享受Scaling Law红利。

**生成式推荐有三条技术路线。**

路线A是Semantic ID生成式，用RQ-Kmeans将item编码为三层语义token，解码在受限空间进行，语义性强，冷启动友好。代表是Google TIGER和快手OneRec。

路线B是纯生成式Transformer，用原子ID作token，vocab百亿级，解码需近似搜索，语义弱，冷启动困难。代表是Meta HSTU和Ultra-HSTU。

路线C是Transformer加判别式排序，保留sparse embedding table，用CTR loss训练，AUC评估，不是真正生成式。代表是字节RankMixer、TokenMixer-Large、美团MTGR、百度GRAB。

**Semantic ID解决核心问题。** 传统ID没有语义，比如video_1024892和article_882745可能讲同一话题但ID毫无关联。Semantic ID用LLM编码加RQ-Kmeans量化，相近内容共享前缀token，获得泛化能力。为什么用RQ-Kmeans而非RQ-VAE？因为RQ-VAE存在codebook collapse，大量token从不被使用。

**超长序列建模是另一核心技术。** 传统模型截断或压缩用户历史，生成式模型可接收完整序列。Meta的Ultra-HSTU处理18层16K序列，用半局部注意力突破O(n²)瓶颈，实现5倍训练加速和21倍推理加速。

**四大代表工作要记住。**

Meta GR是生成式开山之作，1.5万亿参数，CTR提升12.4%，NDCG提升65.8%，首次验证工业推荐系统Scaling Law。

快手OneRec是端到端颠覆路线，RQ-Kmeans加Session-wise加IPA对齐，停留时长提升1.6%，核心指标100%正向。OneRec-V2用Lazy Decoder，MFU达62%，计算量下降94%。

Meta Ultra-HSTU是稀疏注意力极致优化，2026年2月发布，半局部注意力SLA加动态拓扑，生产环境消费提升4%、互动提升8%。

字节TokenMixer-Large是判别式Scaling代表，7亿到15亿参数，Mixing-and-Reverting操作解决梯度消失，电商订单提升1.66%、广告ADSS提升2.0%。

---

## 第四部分：大厂技术路线对比 (5分钟)

**各厂商路线分化与核心数据**

2025到2026年各大厂技术路线明显分化。

**Meta是生成式GR的坚定推进者，** GR、HSTU到ULTRA-HSTU再到VLM，持续深耕，系统模型协同设计。

**快手是从激进到务实的典型代表，** OneRec端到端生成式验证可行性后，2026年转向UniMixer判别式统一架构，保留现有架构享受Scaling红利。

**字节是判别式Scaling的最完整布局，** HLLM到OneTrans到RankMixer到TokenMixer-Large再到UG-Separation，全程判别式无激进生成式押注，低延迟高可控。

**美团是深耕GR的多篇论文积累者，** MTGR保留交叉特征，65倍FLOPs提升。

**腾讯是从上下文工程到广告深融合，** OnePiece到GPR到OneRanker，Shopee主搜索GMV提升2%、广告提升2.9%。

**阿里是判别式另一条路，** SSR突破隐式稀疏天花板，SORT系统性改造Transformer。

**核心数据要记住。** Meta GR CTR加12.4%，快手OneRec停留时长加1.6%，字节TokenMixer-Large订单加1.66%、广告加2.0%，美团MTGR FLOPs加65倍，百度GRAB CTR加3.49%、收入加3.05%。

---

## 第五部分：工业落地挑战与业界共识 (4分钟)

**知乎讨论精华与五大共识**

知乎讨论分为三派。

**质疑派**认为：腾讯马进说推荐大模型可能比纯生成式更work；陈东文说没有ground truth是最大问题；几野观察到字节判别式为主、快手转UniMixer、阿里无公开工作、腾讯换赛道。

**看好派**认为：OnePiece发明者傅聪说生成式不是伪范式是未来；九河之间说字节是极致进化非范式革命，判别式仍是基石；王喆说生成式价值在于突破信息茧房，世界知识注入是核心优势。

**讨论焦点**围绕两个问题：是否具备Scaling Law决定天花板高度，ROI是否为正决定工程落地价值。

**五大业界共识。**

一是语义token升级为系统级接口，Item从原子ID演化为可组合、可迁移、可生成、可索引的token序列。

二是统一化从特征走向平台，OneTrans、MixFormer、MDL实现统一特征类型、场景分布、任务目标。

三是推理serving和复用成一等公民，UG-Separation、LASER、GR4AD证明不能转化为在线计算图的论文创新不构成真正工业创新。

四是生成式one-model在广告电商加速落地，美团EGA-V2、快手GR4AD、腾讯GPR端到端生成已在广告场景率先突破。

五是判别式大ranking backbone进入成熟期，RankMixer、TokenMixer-Large更像foundation model，更统一token接口、更深主干、更明确scaling目标。

---

## 第六部分：昇腾推荐团队行动建议 (5分钟)

**三大行动方向与优先级矩阵**

昇腾推荐团队的核心定位是支撑互联网客户从GPU迁移到昇腾平台，或在昇腾平台原生开发新一代推荐模型。生成式推荐技术浪潮为昇腾提供了差异化窗口期。

**行动方向一：迁移支撑是根基，优先级P0。**

PyTorch API兼容度提升至95%以上，FBGEMM和HKV算子性能对标NVIDIA，迁移工具链建设。客户代码零改动迁移，立即启动。

**行动方向二：原生开发是差异化，优先级P1。**

TorchRec-V2生成式框架适配，Recsys-GR完整适配Meta GR和HSTU，训推一体方案。建议时间是2026年第三季度到第四季度。

**行动方向三：生态跟进与时俱进，优先级P2。**

PyTorch版本同步，arXiv新论文一个月内完成算子需求分析，Benchmark建设持续扩充模型至60以上。

**优先级矩阵。**

P0立即行动：PyTorch API兼容度提升至95%以上，消除迁移阻断项；FBGEMM和HKV算子性能对标NVIDIA，推荐模型最大瓶颈是昇腾差异化竞争点。

P1第三季度：Recsys-GR完整适配，3个月内完成HSTU融合算子实现；Ascend-RecBench独立代码仓建设，客户决策支撑和销售武器。

P2第四季度：Ascend-RecServe推理框架整合，整合vLLM-ascend加AOTI加动态Batch，推理延迟目标降低20%以上。

---

## 结语 (1分钟)

总结三个核心要点。

第一，推荐系统正在经历从判别式到生成式的范式转变，背后是Scaling Law、长序列建模、世界知识注入和RL对齐四重突破的同时到来。

第二，生成式推荐有三条技术路线，字节判别式Scaling和Meta生成式GR是两条主线，工业落地正在加速，但成本、延迟、可控性三座大山仍需攻克。

第三，昇腾的机会在于迁移支撑和原生开发两端发力，让客户在不修改代码的情况下切换到昇腾平台，同时快速跟进前沿模型适配。

谢谢大家，欢迎提问和讨论。

---

> **备注**: 本演讲稿基于slides-v2.html、generative-rec-tech-evolution-analysis.md和zhihu-generative-rec-analysis.md生成。
