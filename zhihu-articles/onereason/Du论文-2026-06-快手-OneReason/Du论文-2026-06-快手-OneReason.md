# Du论文-2026-06-快手-OneReason
> 发布时间: 编辑于 2026-06-11 14:46・浙江
> 原文链接: https://zhuanlan.zhihu.com/p/2048412422847870910

---

​

目录

```text
# 当推荐系统遇上 o1：快手 OneReason 是如何让"思考"真正涨指标的？
```

> OneReason Technical Report · 第一个在工业推荐基准上让 thinking mode 持续打过 non-thinking 的生成式推荐基础模型

**作者**：OneRec Team（Biao Yang, Boyang Ding, Chenglong Chu, Dunju Zang, Fei Pan, Han Li, Hao Jiang, … 等数十位核心贡献者与贡献者）

**机构**：快手（Kuaishou）

**发表**：arXiv 2026-06（arXiv:2606.06260v1）· 已在快手 App 本地服务广告场景上线

**链接**：[arXiv:2606.06260](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2606.06260) · [PDF](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2606.06260)

**关键词**：[Generative Recommendation](https://zhida.zhihu.com/search?content_id=276703921&content_type=Article&match_order=1&q=Generative+Recommendation&zhida_source=entity) · Reasoning · Chain-of-Thought · [Reinforcement Learning](https://zhida.zhihu.com/search?content_id=276703921&content_type=Article&match_order=1&q=Reinforcement+Learning&zhida_source=entity) · Foundation Model · Itemic Token · [Industrial Deployment](https://zhida.zhihu.com/search?content_id=276703921&content_type=Article&match_order=1&q=Industrial+Deployment&zhida_source=entity)

* * *

### 目录

[摘要（Abstract）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#%E6%91%98%E8%A6%81abstract) [1\. 引言（Introduction）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#1-%E5%BC%95%E8%A8%80introduction) [2\. 推理设计哲学（Reasoning Design Philosophy）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#2-%E6%8E%A8%E7%90%86%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A6reasoning-design-philosophy) [3\. 推理推荐基准设计（Benchmark Design）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#3-%E6%8E%A8%E7%90%86%E6%8E%A8%E8%8D%90%E5%9F%BA%E5%87%86%E8%AE%BE%E8%AE%A1benchmark-design) [4\. 预训练流程（Pre-Training Pipeline）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#4-%E9%A2%84%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8Bpre-training-pipeline) [5\. SFT 流程（SFT Pipeline）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#5-sft-%E6%B5%81%E7%A8%8Bsft-pipeline) [6\. RL 流程（RL Pipeline）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#6-rl-%E6%B5%81%E7%A8%8Brl-pipeline) [7\. CoT 分析指标（CoT Analysis Indicators）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#7-cot-%E5%88%86%E6%9E%90%E6%8C%87%E6%A0%87cot-analysis-indicators) [8\. 实验（Experiments）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#8-%E5%AE%9E%E9%AA%8Cexperiments) [9\. 部署（Deployment）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#9-%E9%83%A8%E7%BD%B2deployment) [10\. 相关工作（Related Works）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#10-%E7%9B%B8%E5%85%B3%E5%B7%A5%E4%BD%9Crelated-works) [11\. 结论与未来工作（Conclusion & Future Work）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#11-%E7%BB%93%E8%AE%BA%E4%B8%8E%E6%9C%AA%E6%9D%A5%E5%B7%A5%E4%BD%9Cconclusion--future-work) [术语对照表（Glossary）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#%E6%9C%AF%E8%AF%AD%E5%AF%B9%E7%85%A7%E8%A1%A8glossary) [参考文献（References）](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#%E5%8F%82%E8%80%83%E6%96%87%E7%8C%AEreferences) [AI 解读：OneReason 的工业意义](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ai-%E8%A7%A3%E8%AF%BBonereason-%E7%9A%84%E5%B7%A5%E4%B8%9A%E6%84%8F%E4%B9%89)

* * *

### 摘要（Abstract）

OneRec 系列的**生成式推荐模型**（Generative Recommendation Models）已在快手的多个真实业务场景中广泛部署，包括**短视频**（Short-Video）、**直播**（Live-Streaming）、**广告**（Advertising）和**电商**（E-Commerce）。然而，这些生成式模型只能从**扩展性优势**（Scaling Advantage）中获益，其**推理能力**（Reasoning Ability）难以被激活——因为我们无法仅用物品 token（Itemic Tokens）构造出有意义的**思维链**（Chain-of-Thought, CoT）序列。

> 受 LLM 领域”先思考再回答”（Think Before Answer）范式成功的启发，我们进行了初步研究（即 OneRec-Think、OpenOneRec）来探索生成式推荐中的推理能力。 _— “Inspired by the success of the reasoning-style ‘think before answer’ paradigm in the LLM field, we conduct preliminary studies (i.e., OneRec-Think, OpenOneRec) to explore reasoning capability in generative recommendation.”_

然而，我们注意到一个**出乎意料的现象**：**思考模式（Thinking Mode）在推荐基准上相比非思考模式（Non-Thinking Mode）并未展现出优势**。

借鉴多模态语言模型（MLLM）中关于 CoT 鲁棒性的最新发现，我们认为推荐中有效推理依赖两个因素：（1）**感知**（Perception）——将物品 token 锚定在其底层语言语义中的能力；（2）**认知**（Cognition）——将用户行为序列重组为连贯的**潜在兴趣点**（Latent Interest Points）的能力。

因此，我们提出 **OneReason**，包含三大核心技术：

**（一）** 预训练中的**强物品 token 感知**（Strong Itemic Token Perception）

**（二）** SFT 中面向推荐任务的**三层认知增强 CoT 格式**（Three-Level Cognition-Enhanced CoT Format）

**（三）** RL 中的**先专精后统一训练策略**（Specialize-then-Unify Training Recipe）

> 借助上述技术，OneReason 首次成功实现了在快手多个真实业务基准上，**思考模式持续优于非思考模式**。 _— “Equipped with the above techniques, OneReason successfully achieves a thinking mode that outperforms the non-thinking mode across multiple real-world business benchmarks at Kuaishou.”_

此外，我们还观察到一个有趣的现象：用 CoT 监督数据替换无 CoT 推荐数据，可以在多个业务域中**提升非思考推理的性能**。这表明 CoT 监督的某些收益可以迁移到直接解码中。

最后，我们系统地提出了一系列推理基准和设计洞见，并将开源 **OneReason-8B** 和 **OneReason-0.8B** 模型以促进生成式推荐模型的研究。

> **图 1：性能概览**（Figure 1: Performance overview）— _截图占位_

* * *

### 1\. 引言（Introduction）

近年来，基于 [Transformer](https://zhida.zhihu.com/search?content_id=276703921&content_type=Article&match_order=1&q=Transformer&zhida_source=entity) 的**大语言模型**（Large Language Models, LLMs）已将机器智能推至前所未有的高度。在大规模计算和先进训练方法论的驱动下，LLM 展现出两个激动人心的特性：

**特性一：预训练中的扩展性**（Scaling Property in Pre-training）——当模型规模、训练数据和计算资源被扩大时，收敛损失遵循一个规律性的、可外推的衰减趋势 \[[Kaplan et al., 2020](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-kaplan2020scaling); [Hoffmann et al., 2022](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-hoffmann2022)\]。随着扩展的推进，多个基准上的评估分数同步提升，展示了模型卓越的语义理解和泛化能力。

**特性二：后训练中的推理性**（Reasoning Property in Post-training）——在预训练之后，后训练阶段通过高质量的**监督微调**（[Supervised Fine-Tuning](https://zhida.zhihu.com/search?content_id=276703921&content_type=Article&match_order=1&q=Supervised+Fine-Tuning&zhida_source=entity), SFT）和**强化学习**（Reinforcement Learning, RL）进一步释放和细化模型的高级能力 \[[Ouyang et al., 2022](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-ouyang2022); [Shao et al., 2024](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-shao2024deepseekmath)\]。在这些后训练技术的驱动下，LLM 学会遵循 CoT 风格的程序 \[[Wei et al., 2023](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-wei2023cot); [Kojima et al., 2023](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-kojima2023)\]，形成”先思考再回答”（Think-Before-Answer）的推理范式。这一范式显著增强了其解决复杂、深度问题的能力，如 OpenAI o1 \[[Jaech et al., 2024](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-jaech2024openai)\] 和 DeepSeek-R1 \[[Guo et al., 2025](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-guo2025deepseek)\] 系列所示。

受 LLM 的**扩展性**和**推理性**的启发，我们深感好奇：**这两个特性是否能被进一步利用来推动工业推荐系统的能力？**

> 对于**扩展性**，我们的 OneRec V1/V2 系列生成式模型已经验证了**工业推荐场景中的扩展性**，在快手贡献了显著的业务收益。 _— “For the Scaling property, our OneRec V1/V2 series of generative models have validated the scaling property in industrial recommendation scenarios, contributing significant business gains at Kuaishou.”_

对于**推理性**，我们承认纯推荐数据驱动的 OneRec 模型无法自行解锁推理能力。由于 OneRec 模型完全在纯物品序列数据（Pure Itemic Sequential Data）上训练，它只学到了扁平的转移模式，没有任何底层的逻辑思维痕迹。

为了引出推理能力，我们最近的模型 OneRec-Think \[[Liu et al., 2025](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-liu2025onerecthink)\] 和 OpenOneRec \[[Zhou et al., 2026](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-zhou2026openonerec)\] 探索了引入交错的物品-文本推荐数据和通用领域推理数据。在这两个模型中，我们成功观察到：

> **“先思考再回答”模式可以推广到推荐任务**，允许我们检查模型的底层逻辑并理解用户的兴趣以及下一个物品预测的理由。 _— “the ‘think-before-answer’ pattern can be generalized to recommendation tasks, allowing us to inspect the underlying logic of the model and to understand the user’s interests as well as the rationale behind the next-item prediction.”_

然而，我们也注意到一个出乎意料的现象：**思考模式在推荐基准上相比非思考模式没有表现出显著优势**。

为了探索这一现象的根本原因，我们转向了**多模态 LLM（MLLM）**文献，其中观察到了惊人相似的推理模式脆弱性。\[[Sun et al., 2026](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-sun2026reading)\] 揭示，当文本和视觉模态对齐不足时，模型倾向于机械地”读取”表面视觉文本，而非真正推理底层视觉语义——这有力地表明**深度跨模态对齐是真正思考的先决条件**。\[[Zhou et al., 2025](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-zhou2025perceptioncognition)\] 系统综述了 MLLM 中的感知到认知路径，并得出结论：**没有先建立模态间稳健的感知对齐，高级交互推理就无法涌现**。在此对齐基础上，\[[Jiang et al., 2025](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-jiang2025corvid)\] 进一步证明，即使模态已对齐，推理鲁棒性仍然脆弱，除非 CoT 痕迹本身被精心结构化——特别是采用**逻辑连贯的粗到细推理格式**来防止幻觉和过度思考。

> **为什么“思考”反而让推荐变差？一个反直觉但又很合理的现象**
> 想象你让一个连股票代码都读不顺的人写 _投资分析报告_——他写得越多，判断错得越严重。推荐模型面临同样问题：在深度语义对齐到位之前，“思考”是饱含错误环节的镲型炭火；错误越多，生成的 CoT 越是“以讯传讯”的噪声源，反而会护望估计的后验分布分散。这就是为什么 OneRec-Think 的 thinking mode 在推荐指标上反而走不赢 non-thinking——不是“思考”不好，是“该思考什么”本身还没打下基础。于是 OneReason 的修法就是两道酶：**先学看（Perception）、再学想（Cognition）**，并且两道酶都必须“可考核”。
> 这些工作汇聚于两个本质且互补的支柱：**(i) 感知中的模态对齐**，为感知提供基础并防止模型在空洞的表面信号上运作；**(ii) 认知中的 CoT 质量**，提供连贯的粗到细推理轨迹，在对齐到位后实现忠实的”先思考再回答”过程。 _— “these works converge on two essential and complementary pillars for unlocking genuine reasoning capability: (i) modality alignment in perception, which grounds perception and prevents the model from operating on hollow surface signals; and (ii) CoT quality in cognition, which provides coherent coarse-to-fine reasoning trajectories and enables a faithful think-before-answer process once alignment is in place.”_
> **图 2：OneReason 的预训练、SFT、RL 和基准测试流程**（Figure 2: The pre-training, SFT, RL, and benchmarking pipeline of OneReason）— _截图占位_
> **图 3：MLLM 思维链推理的三种失败/成功模式**（Figure 3: Three failure/success modes of MLLM chain-of-thought reasoning）— _截图占位_

对于生成式推荐，这些洞见意味着：让模型真正”思考”用户兴趣需要同时具备两个条件——**感知中的深度对齐**，即在物品模态和自然语言模态之间建立深度对齐，使物品 token 成为可引用的、可组合的语义单元，而非不透明的标识符；以及**认知中的推荐特定且逻辑流畅的 CoT 结构**，即设计用于在此对齐基础上对用户历史和物品属性进行有意推理的脚手架。

鉴于上述分析，我们提出 **OneReason**，一个具有真正推理能力的推荐基础模型。具体而言：

**感知层面**——为了获得更好的物品感知能力，我们收集了复杂多样的粗到细对齐预训练语料，将新插入的物品 token 与文本 token 在同一语义空间中对齐。

**认知层面**——为了获得更好的推理认知能力，我们精心设计了标准化的思考结构以确保 CoT 质量，并构建了 SFT 语料。

**强化学习层面**——在预训练和 SFT 之后的模型基础上，我们发现在多域混合 RL 下思考模式仍然落后于非思考模式，但在 RL 限制在单一域时始终超越非思考模式。这一对比促使我们采用**“先专精后统一”RL 策略**——首先在每个域内进行聚焦 RL 以充分释放思考模式的优势，然后应用**拒绝采样微调**（[Rejection Sampling Fine-Tuning](https://zhida.zhihu.com/search?content_id=276703921&content_type=Article&match_order=1&q=Rejection+Sampling+Fine-Tuning&zhida_source=entity), RFT）或**多教师在线策略蒸馏**（[Multi-Teacher On-Policy Distillation](https://zhida.zhihu.com/search?content_id=276703921&content_type=Article&match_order=1&q=Multi-Teacher+On-Policy+Distillation&zhida_source=entity), MOPD）进行跨域平衡和精炼。

> 同时，我们还有一个有趣的观察：在相同训练 token 数量下，添加推荐 CoT 监督可以在多个域中**改善非思考模式的性能**——相比于纯无 CoT 数据训练。我们将此视为行为证据：CoT 痕迹中的某些监督信号可以迁移到直接解码，而非证明 CoT 是普遍必要的。 _— “we also make an interesting observation: under an equal number of training tokens, adding recommendation CoT supervision can improve non-thinking-mode performance compared with training on pure unCoT data in several domains.”_

为了评估模型在推荐场景中的推理能力，我们引入 **OneReason-Bench**，一个精心策划的推理导向推荐基准集合。该基准沿多个轴组织，旨在以全面和结构化的方式评估推理能力。

* * *

### 2\. 推理设计哲学（Reasoning Design Philosophy）

在详细描述架构和训练方案之前，我们首先概述推荐推理的设计原则：**推荐中什么是好的 CoT 痕迹？**

在涉及数学、代码或符号逻辑的经典 LLM 推理任务中，通常假设存在保持任务约束并逐步应用相关世界知识来逻辑回答问题的中间步骤。

> 与拥有单一正确答案的经典推理任务不同，**推荐同时允许多个合理的物品**，且底层的用户意图从不可直接观测——它必须从一长串历史物品 token 序列中推断。 _— “Unlike classical reasoning tasks with a single correct answer, recommendation admits multiple plausible items at once, and the underlying user intent is never directly observable; it must be inferred from a long sequence of historical itemic tokens.”_

因此，**推荐推理是溯因的（Abductive）而非演绎的（Deductive）**：它需要假设潜在的兴趣点，建模它们如何随时间演变，并用它们来解释为什么特定候选物品适合当前上下文。

换言之，推荐 CoT 应该：选择相关行为作为假设的兴趣点，将它们压缩为可解释的偏好，建模这些兴趣之间的时间转移，最终将推断的偏好状态与推荐物品关联起来。

这一区分驱动了我们的训练和评估设计。我们将推荐推理分解为沿两个维度的**四种诊断能力**（R0→R1→R2→R3 递进层次）：

**R0: 感知（Perception）**　通过显式语义内容解读物品模式是基础能力；没有它，用户行为不可解释。

**R1: 推导（Derivation）**　从单个物品语义推理到基于常识或知识的物品到物品关系；没有它，潜在兴趣无法从嘈杂的交互历史中提取。

**R2: 演变（Evolution）**　将同一潜在兴趣的物品推理为时间过程；没有它，长期、短期和周期性偏好无法建模。

**R3: 推荐（Recommendation）**　连贯推理以在所有服务域（如短视频、直播）上产生高质量决策。

> **直觉解读**：为什么要拆成 R0→R3 四层？
> 数学题只需要 _演细推理_（有唯一正确答案），但推荐是 _溯因推理_（Abductive）——你看到一串行为，需要反推用户的潜在意图。这意味着：如果模型连 **R0 读懂物品**都做不到，后面 R1/R2/R3 都是空中楼阁；如果能读懂但不会 **R1 一跳关联**，就无法从“看了 A”推出“可能喜欢 B”；如果会关联但不会 **R2 时间演变**，就只能推荐“同品类重复”而无法抱住兴趣转移。**这个递进层次同时是诊断架构也是训练架构**——训练从 R0 加到 R3，评测从 R0 查到 R3，缺什么补什么。

* * *

### 3\. 推理推荐基准设计（Benchmark Design）

将基准放在模型描述之前是刻意的：**四粒度预训练**、**压缩-推理 SFT** 和 **先专精后统一 RL** 各自由 OneReason-Bench 的 R0–R3 层揭示的特定差距所驱动。基准因此不是排行榜，而是一个**测量协议**，用于在每一步中证明、监控和最终验证每个下游设计选择。

### 3.1 任务分类体系（Task Taxonomy）

OneReason-Bench 将推理导向的基准任务组织成一个渐进式层次：**感知（R0）→ 推导（R1）→ 演变（R2）→ 推荐（R3）**。

**统一任务形式化**：所有任务被表述为序列生成 **Y = F(X)**，其中 **X** 组合了任务指令 **I** 和上下文 **C**（如物品模式、用户画像或交互历史）。目标 **Y** 可以是物品模式、答案选项、自然语言响应或结构化的演变链。

> **表 1：OneReason-Bench 的任务分类体系**（Table 1: Task taxonomy organized across four reasoning layers · 关键数据：R0 含 3 个子任务, R1 含 1 个子任务, R2 含 3 个子任务, R3 含 2 个子任务）— _截图占位_
> **图 4：任务分类体系图示**（Figure 4: Illustration of the task taxonomy）— _截图占位_

### R0: 感知（Perception）

**感知**（Perception）将物品模式锚定在显式语义内容中，为更高层次的推理和推荐提供语义基础。我们用三种任务类型实例化此层：

**物品理解**（Item Understanding）　给定一个物品 **i**，模型生成其自然语言描述，从物品模式中解码物品语义。

**物品模式锚定**（Itemic Pattern Grounding）　给定自然语言描述，模型将其锚定到对应的物品。

**物品问答**（Item QA）　给定物品 **i** 和基于内容的多选选项 **O^a**，模型选择正确选项。

### R1: 推导（Derivation）

在感知基础上，推导评估模型是否能从单个物品语义推理到基于常识或知识关联的物品到物品关系。

**Item2Item**　给定源物品 **i** 和候选物品集 **O^item**，模型选择最匹配目标物品关系的物品。

### R2: 演变（Evolution）

**演变**评估模型是否能将用户兴趣推理为时间过程，而非静态偏好画像。我们设计了三种自主程度递增的任务变体：

**演变动作选择**（Evolution Action Selection）　给定历史 **H** 和指定兴趣主题 **t**，模型识别与该主题逻辑相关的历史动作 **A**。

**演变主题生成**（Evolution Topic Generation）　给定历史 **H** 和指定兴趣主题 **t**，模型生成描述该兴趣如何发展的演变链 **E\_t**。

**演变直接生成**（Evolution Direct Generation）　仅给定历史 **H**，模型自主生成有效的兴趣演变链 {**E**}，无需主题引导。

### R3: 推荐（Recommendation）

**单域推荐**（Single-Domain Recommendation）　给定用户画像 **P** 和目标域内的交互历史 **H**ᵈ，预测用户下一步将参与的物品。

**跨域推荐**（Cross-Domain Recommendation）　给定用户画像 **P**、多域交互历史 **H** 和指定目标域 **d**，预测该域中的下一个物品。

### 3.2 评估协议（Evaluation Protocols）

**文本生成指标**　物品理解任务使用 **LLM-as-a-Judge** 协议。

**选择指标**　单答案选择任务报告 **Accuracy**；多答案选择任务报告 **F1**。

**演变生成指标**　演变主题生成和直接生成使用 **Action-Logic Score** 系列。

**推荐指标**　推荐和物品模式锚定任务报告 **Pass@K** 和 **Recall@K**。

### 3.3 通用智能检查（General-Intelligence Sanity Check）

遵循 OpenOneRec，我们保留了通用智能检查套件，涵盖 MMLU-Pro 等通用任务，以确保推荐专业化不以牺牲通用推理或指令遵循能力为代价。

* * *

### 4\. 预训练流程（Pre-Training Pipeline）

### 4.1 物品分词器（Itemic Tokenizer）

为了获得紧凑且语义锚定的物品嵌入（Item Embeddings），我们在物品理解任务上联合训练一个多模态编码器和一个解码器 LLM。编码器集成了 **ViT**（Vision Transformer）\[[Radford et al., 2021](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-radford2021learning)\]、从开源 Qwen3-VL 初始化的 LLM \[[Yang et al., 2025](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-yang2025qwen3)\]，以及音频编码器 \[[Vyas et al., 2026](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-vyas2026pushing)\]。它将丰富的内容信号——包括封面图片、视频帧、文本描述和音频——提炼为紧凑的稠密嵌入。

我们通过 **RQ-KMeans** \[[Luo et al., 2025](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-luo2025qarm)\] 对这些嵌入进行量化，使用三层码本（Three-Layer Codebook），每层含 8192 个码。每个物品因此表示为一个物品模式（Itemic Pattern），由一个域感知的起始 token 和三个子 token 组成：

 \\text{<|domain\\\_begin|><a\\\_5028><b\\\_6733><c\\\_2559>} \\tag{1}

其中 `domain` 是 `video`（视频）、`prod`（商品）、`ad`（广告）或 `living`（直播）之一。与 OpenOneRec 不同，我们**去掉了尾部结束 token**，以减少物品模式消耗的总体上下文长度，为推理痕迹留出更多容量。

### 4.2 预训练数据（Pre-Training Data）

先前的生成式推荐工作按任务类型组织预训练数据——物品描述、用户行为序列和画像-文本交错——并混合联合训练。虽然这建立了物品模式和自然语言之间的基本映射，但留下了三个结构性缺口：语义表达同质化、缺乏显式的物品内细粒度语义层次和跨物品关系逻辑、以及用户行为建模的窄条件范式。

> 为弥补这些缺口，我们将预训练语料重构为**四个渐进粒度**，从微观 token 语义到宏观用户行为。 _— “To address these gaps, we restructure the pre-training corpora into four progressive granularities, moving from microscopic token semantics to macroscopic user behavior.”_
> **图 5：四粒度预训练推荐语料概览**（Figure 5: Overview of our four-granularity pre-training recommendation corpora）— _截图占位_

### 推荐语料（Recommendation Corpora）

**Token 粒度（Token Granularity）**：关注单个或组合的子物品 token 语义，在最细粒度实现模态对齐。包含：

**①组合前缀语义预测**（Compositional Prefix Semantic Prediction）　要求模型预测子 token 对 `<a_xxxx><b_xxxx>` 的组合语义含义。

**②部分到整体语义预测**（Part-to-Whole Semantic Prediction）　结构化数据以引出两步语义预测过程——首先预测每个子 token 的细粒度语义，然后合成为物品的整体描述。

**Item 粒度（Item Granularity）**：将物品模式与文本模态对齐，推动跨模态融合到完整物品级别。关键增强：

**①容量感知的描述粗粒化**（Capacity-Aware Caption Coarse-Graining）　由于每个物品仅用三个子 token 表示，过于详细的描述会迫使模型产生幻觉。因此我们在对齐前对描述进行粗粒化：移除物品编码无法恢复的实例特定噪声（如 OCR 文本、ASR 歌词、日期等），将细粒度连续属性映射到粗略区间。

**②多角度物品问答**（Multi-Perspective Item QA）　从多个角度查询模型，包括目标受众偏好、核心产品吸引力、视觉风格和负反馈理由。

**关系粒度（Relational Granularity）**：通过自然语言显式解释物品之间的连接。构建物品模式和自然语言解释的交错序列：

 \\text{Itemic\\\_Pattern}\_0 \\rightarrow \\text{Textual\\\_Explanation}\_0 \\rightarrow \\text{Itemic\\\_Pattern}\_1 \\rightarrow \\cdots \\rightarrow \\text{Itemic\\\_Pattern}\_n \\tag{2}

包括**直接物品到物品关联**（n=1）和**多物品兴趣流**（n>1）。**三个互补数据源**：

**①search-after-play**　用户看完某视频后随即发起的搜索词，捕获”看了什么 → 进一步想了解什么”的显式意图跳转。

**②TagNext CF**　基于协同过滤的物品到下一物品转移，过滤后要求两个物品在标签空间中能被一句话逻辑桥接。

**③多域共现窗口**　同一用户同一时间窗口内跨域（视频 / 直播 / 商品 / 广告）交互的物品对，强制跨域语义交互。

实证上，正负样本在粗粒度 embedding 上存在明显 margin（论文 Figure pos-neg-delta-embed），证明三个源提供了差异化且可被判别的语义信号。

**用户粒度（User Granularity）**：扩展跨模态对齐到用户交互的完整时序演变。包含两种数据格式：

**①域分组行为序列**（Domain-Grouped Behavior Sequences）　通过多轮问答对话动态结构化域分组子序列。

**②按时间交错行为序列**（Chronologically Interleaved Behavior Sequence）　严格按时间戳重构行为序列，并概率性地将部分物品模式替换为对应的文本描述，创建混合时间线以强制深度语义整合。为跨越单个 24h 窗口的子链（sub-chain），从原始序列 \[Item₀, Item₁, …\] 中按间隔 Δ 采样 \[Item₀, Item_Δ, Item_{2Δ}, …\]，并附加相似度过滤。

### 通用领域语料（General-Domain Corpora）

**①文本语料**　涵盖数学推理、代码、科学文献、医学文本和通用指令遵循语料。

**②多模态语料**　将图像 token 化为物品模式，**四个来源**——图文理解对（image-text understanding pairs）、文本-图像交错文档（text-image interleaved documents）、图像编辑三元组（image-editing triplets）、文本到图像生成数据（text-to-image generation）。为与推荐语料保持统一 token 协议而引入专用 `<|sid_begin|>` 域。

### 数据混合策略（Data Composition Strategy）

**①Loss-masked context**　推荐任务与通用领域数据混合训练时，为平衡配方比例而自动注入的通用领域上下文 token **不参与 loss 计算**，只有推荐答案 token 贡献监督信号，避免填充数据冲淡推荐梯度。

**②QA-style：free-form = 1：1**　每个任务同时以结构化问答型和自由生成型两种格式提供，按 1:1 采样，防止模型过拟合任一表面格式。

### 4.3 预训练语料消融实验（Ablation Study）

在 0.8B 模型上以固定 30B token 预算进行消融研究，定义五组配置（Exp1-Exp5），逐步引入各粒度数据。

> **表 2：消融实验结果**（Table 2: Each experiment builds on the previous one by adding a new granularity · 关键数据：Exp1→Exp2 Token 粒度使 R0 Item Understanding\_ad 从 16.37% 提升到 37.86%；Exp4→Exp5 用户粒度使 Cross-Live 从 3.25% 提升到 8.56%）— _截图占位_

**关键洞见**：

**①** Token 和关系粒度数据通过**增强**扩展模型技能集，而 Item 和用户粒度数据通过**替换**精炼现有能力。

**②** 训练中出现**能力权衡**——关系推理的增益可能暂时降低静态检索能力，但后续阶段会恢复。

**③ 用户粒度是最终整合器**，将所有先前知识置于时间行为的上下文中，在多样推荐场景中实现峰值性能。

> **表 3：物品理解案例对比**（Table 3: A representative Item Understanding case in the video domain · 展示 Baseline、+Token、+Item 三种配置下的生成效果差异）— _截图占位_

### 4.4 训练方案（Training Recipe）

训练方案分为**三个阶段**：

> **图 6：三阶段预训练方案**（Figure 6: The three-stage pre-training recipe）— _截图占位_
> **表 4：训练阶段摘要**（Table 4: Summary of training stages · 关键数据：Stage 1 = 110B tokens，Stage 2 = 449B tokens，Stage 3 = 19B tokens，总计约 578B tokens）— _截图占位_

**Stage 1**（110B tokens）　仅训练新引入的嵌入和 LM 头权重，其余骨干冻结，让物品模式嵌入稳定进入语义空间而不扰动预训练权重。单样本长度上限 4K，学习率 2×10⁻⁴ → 1×10⁻⁴。

**Stage 2**（449B tokens）　解冻所有参数进行全联合训练，使模型从四粒度数据混合中吸收推荐知识。单样本长度仍为 4K，学习率 1×10⁻⁴ → 1×10⁻⁵。

**Stage 3**（19B tokens）　将单样本长度上限**从 4K 提升至 32K tokens**，允许模型在长程行为依赖的完整用户历史上训练。学习率 1×10⁻⁵ → 1×10⁻⁶。

所有阶段使用**样本打包**（Sample Packing）以提高序列利用率，打包序列长度为 48K tokens。消融实验中的 Exp1 baseline 并非从零起点，而是在 OpenOneRec dataset 上训练后的模型上进行逐步加入四粒度数据（Exp2→Exp5）的增量对比，避免了初始状态差异干扰。

* * *

### 5\. SFT 流程（SFT Pipeline）

物品-文本对齐预训练为模型提供了必要的感知基础。SFT 在此基础上培养**推荐认知**——即在指令格式下操作锚定物品 token 的能力。我们围绕 R0–R3 层次组织此阶段，但设置了两个互补的训练轴：

**①压缩轴**（Compression Axis）　教模型将冗长嘈杂的用户历史缩减为类型化的画像状态和紧凑的兴趣演变主题。

**②推理轴**（Reasoning Axis）　教动态操作——R1 推导一跳物品到物品桥接、R2 追踪时间兴趣演变、R3 执行转移判断以进行下一次交互预测。

### 5.1 R0: 感知（Perception）

R0 提供 SFT 阶段的感知层。其作用是在要求模型执行更高层次推荐推理之前，使物品 token 变得**可读、可控、语义可用**。

描述数据涵盖四大内容域：短视频、电商商品、直播和广告。问答数据范围较窄：仅覆盖短视频和电商商品域。R0 因此包含两个互补的任务族：

**① 物品 token 与自然语言描述之间的双向映射**

**② 平台内容上的问答（QA）**

> **表 5：R0 案例对比——预训练物品描述映射 vs SFT 指令式感知监督**（Table 5: Representative R0 case comparing item-caption pre-training with instruction-style SFT perception supervision · 展示预训练的直接映射 vs SFT 的 CoT 化物品 token 语义解释）— _截图占位_

关键区别不在于目标描述本身，而在于**监督的粗到细物品 token 语义如何支持该描述的显式化**。描述数据包含思考和非思考变体。在思考格式中，模型首先解释每个物品 token 如何贡献最终物品含义，然后生成最终描述。

总计，R0 包含约 **941K** 样本（约 682K 描述样本和 259K 问答样本）。

### 5.2 R1: 推导（Derivation）

R1 目标是**跨物品关系推导**。它将 R0 从理解单个物品扩展到解释为什么一个物品可能自然导向另一个。

R1 从与关系预训练相同的原始候选池构建（TagNext CF 关系和播放后搜索数据），但监督标准不同。关系预训练暴露大规模协同转移，而 R1 要求每个保留的配对支持**显式的、一跳的、基于证据的推理**。

> R1 定义的不是泛化的“相似性”，而是**source-to-follow-up bridge**：学生模型训练时**只看到 source**，target 仅作为 teacher-side filtering 依据，可避免信息泄露。

**Teacher-side 显式相关性三分类**：仅保留 explicitly related 一类，严格丢弃 explicitly unrelated 和 uncertain。

**① Explicitly related**　source 到 target 存在可被一句话表述的逻辑桥接。

**② Explicitly unrelated**　source 和 target 在语义上明显不能被联系。

**③ Uncertain**　可能相关但需要上下文信息才能判断。

**6 个 Bridge Variables 锁死推理跳点**：source-side need（源端表面需求）、bridge type（关系型号：互补 / 升级 / 跨品类联动 等）、abstract bridge（抽象桥接表述）、continuation direction（延续方向）、reason seed（推理种子句）、confidence（置信度）。

> **表 6：R1 案例对比——预训练消息式关系数据 vs 一跳 SFT 推导监督**（Table 6: Representative R1 case comparing message-style relational pre-training with one-hop SFT derivation supervision）— _截图占位_

此过程将原始候选池提炼为约 358K TagNext CF 对和 388K 播放后搜索对，最终采样约 **400K** R1 训练样本。

### 5.3 R2: 演变（Evolution）

R2 目标是**用户兴趣演变建模**。它将推荐认知从静态偏好理解扩展到对行为的**时间结构化解释**。

我们从全域用户行为时间线构建 R2，其中异构动作按时间顺序组织。一个强 LLM 首先识别指示用户意图有意义转变、细化或关闭的关键演变节点。候选链经由独立的 LLM-as-a-Judge 质量评估阶段过滤。

基于验证的演变链，我们构建了三个 SFT 任务族：**演变动作选择**、**演变主题生成**和**演变直接生成**。

最终 R2 数据集包含约 **130K** 训练实例。

### 5.4 R3: 推荐（Recommendation）

R3 是 SFT 混合物的推荐级组合层。R0–R2 分别训练专用能力，R3 将它们在最终的下一次交互预测设置中组合在一起。

### 推荐推理痕迹构造

每个 R3 合成样本由三个成分构建：用户画像 **P\_u**、按时间排列的交互时间线 **H\_u**，以及真实的后续交互 **y\_u⁺**。目标交互仅用作监督答案，**被显式禁止出现在推理痕迹中**。

核心设计原则是将推荐 CoT 视为**两轴压缩加转移判断**过程。

**第一轴**——画像类型压缩（Persona-Type Compression）　总结稳定的偏好先验。

**第二轴**——兴趣演变类型压缩（Interest-Evolution-Type Compression）　识别近期轨迹主题。

具体而言，我们采用**三阶段协议**：

**阶段一：画像抽象（Persona Abstraction）**

 \\mathcal{C}\_u = \\mathrm{Abstract}(\\mathcal{P}\_u, \\mathcal{H}\_u) \\tag{3}

目标是将稀疏嘈杂的行为证据压缩为紧凑的、可解释的先验。

> **表 7：画像抽象代表性案例**（Table 7: Representative persona-abstraction cases for recommendation reasoning · 展示中年家庭主妇、直播购物爱好者、多人共用设备等画像类型）— _截图占位_

**阶段二：兴趣展开（Interest Expansion）**

 \\mathcal{Z}\_u = \\mathrm{Expand}(\\mathcal{C}\_u, \\mathcal{H}\_u) \\tag{4}

> **图 7：兴趣展开宽度的效果**（Figure 7: Effect of interest expansion width **n** · 关键数据：**n ∈ {1,3,5}** 一致优于更宽的如 10 或 20）— _截图占位_

**阶段三：转移推断（Transition Inference）**

 z\_u^{\\star} = \\mathrm{Infer}\_{\\mathrm{trans}}(\\mathcal{Z}\_u \\mid \\mathcal{C}\_u, \\mathcal{H}\_u, d) = \\arg\\max\_{z \\in \\mathcal{Z}\_u} s(z \\mid \\mathcal{C}\_u, \\mathcal{H}\_u, d) \\tag{5}

其中 **C\_u** 表示压缩的用户状态，**Z\_u** 表示展开的候选兴趣假设集，**d** 表示目标域。

转移推断需服从**5 大约束**，从而让 trace 具备工业可用性：

**① Evidence priority（证据优先）**　仅能引用输入上下文中出现过的行为证据，不允许幻觉未出现的历史。

**② Transition bridge（转移桥接）**　必须明示从压缩用户状态 **C\_u** 到预测目标域 **d** 的逻辑跳点。

**③ Granularity calibration（粒度校准）**　使用跨 IP / 品类粒度而非单个 IP 名称重复，以支持跨 IP 泛化。

**④ Conflict resolution（冲突解决）**　当多个兴趣假设冲突时，需以近期轨迹强度为准仅选一。

**⑤ Leakage control（泄露控制）** **_reasoning must be written as if the answer is unknown_**——严禁 trace 中出现 target 的标题 / 物品 ID / 任何唯一标识。

> **表 8：R3 推荐 CoT 案例——展示兴趣展开和转移推断**（Table 8: R3 recommendation CoT case showing Interest Expansion and Transition Inference）— _截图占位_

### 推理痕迹质量评估

推荐推理痕迹的质量沿**五个互补维度**评估：

**安全性**（Safety）　CoT 是否直接暴露目标内容。

**一致性**（Consistency）　痕迹结论是否与预期推荐目标对齐。

**逻辑性**（Logic）　痕迹是否展示可迁移的推理而非表面拼接。

**事实性**（Factuality）　痕迹是否准确反映观察到的用户历史。

**信息性**（Informativeness）　推理是否提供有用的推荐指导。

> **图 8：推理痕迹质量的分数分布和均值画像**（Figure 8: Score distribution and mean-score profile for reasoning-trace quality）— _截图占位_

### 5.5 物品指令数据（Itemic Instruction Data）

物品指令数据改善了物品 token 显式出现在输入上下文中时的指令遵循行为。论文明确划分为 **6 大任务组**：

**① Conversion and Editing（转换与编辑）**　物品 ↔ 描述双向转换、描述改写。

**② Retrieval and Selection（检索与选择）**　从候选集中选出最匹配某描述的物品。

**③ Matching and Classification（匹配与分类）**　判断两个物品是否为同一类别。

**④ Comparison and Aggregation（比较与聚合）**　两个物品间差异点提取、多物品列表主题抽取。

**⑤ Generation and Verification（生成与验证）**　基于主题生成物品列表、验证给定物品是否符合描述。

**⑥ Instruction Control（指令控制）**　严格遵循输出格式 / 个数 / 顺序等限制性指令。

最终数据集包含约 **103.0K** 样本。

### 5.6 通用领域数据（General-Domain Data）

纳入 StepFun 发布的 **Step-3.5-Flash-SFT** 数据集作为通用领域数据源，**预处理**后包含约 **1.5M** 高质量 SFT 样本。为使其与推荐任务输入格式对齐，预处理包含：移除工具调用迹象（tool-use traces）、仅保留 system + user + assistant 三角色对话格式。

* * *

### 6\. RL 流程（RL Pipeline）

SFT 阶段后，模型已获得通用语义理解、指令遵循能力和初步的推荐推理能力。然而，SFT 主要使模型模仿教师生成的推理轨迹，性能仍受限于教师模型的能力。

> 我们采用**“先专精后统一”策略**：首先在每个推荐域上分别通过 RL 优化模型，得到四个域专精教师模型；然后通过 RFT 或 MOPD 将域特定能力整合到一个统一模型中。 _— “we adopt a ‘specialize-then-unify’ strategy. We first optimize the model separately on each recommendation domain through RL, resulting in four teacher models that specialize in domain-specific recommendation reasoning.”_
> **图 9：”先专精后统一”流程概览**（Figure 9: Overview of the “specialize-then-unify” pipeline）— _截图占位_

### 6.1 面向推荐的强化学习（Recommendation-oriented RL）

我们首先使用 **GRPO**（Group Relative Policy Optimization）\[[Shao et al., 2024](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-shao2024deepseekmath)\] 在每个推荐域上分别优化模型。

**面向推荐的 GRPO**：对于用户 **u**，模型将其上下文 **q** 作为输入，采样一组推荐 rollout：

 \\mathcal{G}\_{u} = \\{(\\mathrm{CoT}\_{u,i}, \\mathbf{c}\_{u,i})\\}\_{i=1}^{G} \\tag{6}

归一化组内奖励以获得相对优势：

 \\hat{A}\_{u,i} = \\frac{R\_{u,i} - \\mathrm{mean}(\\{R\_{u,k}\\}\_{k=1}^{G})}{\\mathrm{std}(\\{R\_{u,k}\\}\_{k=1}^{G}) + \\delta} \\tag{7}

**两阶段 Rollout 设计**：为了分摊推理成本，我们首先采样 **N** 个推理痕迹，然后在每个痕迹条件下并行生成 **K** 个物品 token 序列：

 \\mathbf{c}\_{u,i,j} = \[\\langle\\texttt{|domain\\\_begin|}\\rangle, c^{(1)}\_{u,i,j}, c^{(2)}\_{u,i,j}, c^{(3)}\_{u,i,j}\] \\tag{8}

产生 **N × K** 个有效推荐 rollout，但只生成 **N** 个推理痕迹。

> **图 10：面向推荐的 GRPO 概览**（Figure 10: Overview of recommendation-oriented GRPO）— _截图占位_

**奖励设计**：每个 rollout 的奖励为：

 R\_{u,i,j} = R\_{\\mathrm{rule}}(\\mathbf{c}\_{u,i,j}) \\cdot R\_{\\mathrm{div}}(\\mathrm{CoT}\_{u,i}) \\tag{9}

其中基于规则的奖励：

 R\_{\\mathrm{rule}}(\\mathbf{c}\_{u,i,j}) = \\mathbf{I}\[\\mathbf{c}\_{u,i,j} \\in \\mathcal{C}^{+}\_{u}\] \\tag{10}

多样性因子：

 R\_{\\mathrm{div}}(\\mathrm{CoT}\_{u,i}) = \\frac{\\max(0, m\_i^{(1)}-1)}{K-1} \\tag{11}

其中 **m\_i⁽¹⁾** 是同一 reasoning trace 并行生成的 **K** 个物品 token 序列中“首个子 token” **c⁽¹⁾** 的不同取值个数。以 first-digit 粗粒度推动多样性是因为 **c⁽¹⁾** 粗粒度上控制了后续解码路径，近似品类轴，从而避免仅在同一品类中打平产出。

**阶段感知裁剪（Stage-wise Clipping）**：推理 token 和物品 token 表现出截然不同的行为。推理痕迹长且主要支持探索，而最终物品 token 序列短但直接决定推荐奖励。因此使用不同的裁剪范围：

 \\bar{r}\_t(\\theta) = \\mathrm{clip}(r\_t(\\theta), 1-\\epsilon^{-}(t), 1+\\epsilon^{+}(t)) \\tag{12}

 \\boldsymbol{\\epsilon}(t) = \\begin{cases} \\boldsymbol{\\epsilon}\_{\\mathrm{CoT}}, & o\_t \\in \\mathrm{CoT} \\\\ \\boldsymbol{\\epsilon}\_{\\mathrm{item}}, & o\_t \\in \\mathbf{c} \\end{cases} \\tag{13}

> **直觉解读**：为什么推理 token 要宽松、物品 token 要严格？
> 推理痕迹是「**怎么想**」，允许多样探索才能跳出局部最优；一旦裁剪过紧，模型会快速「背诵」出某一种推理路径而不再探索。物品 token 是「**选哪个**」，不能放任分布乱飘——一旦概率分布失控，就会生成不合法 token、出现 illegal item。这个设计本质上是把 _exploration_（探索）和 _exploitation_（利用）交给不同 token 类型管理：CoT 段宽容、item 段紧绷，是推荐 RL 稳定性的关键创新点。

**负样本降权**：由于只有小部分 rollout 命中真实值，大量非命中样本可能主导梯度。为每个 rollout 分配标量权重：

 w\_{u,i,j} = \\begin{cases} 1.0, & \\text{if } R\_{\\mathrm{rule}}(\\mathbf{c}\_{u,i,j}) = 1 \\\\ \\beta, & \\text{otherwise} \\end{cases} \\tag{14}

> **图 11-14：RL 消融实验**（Figures 11-14: Ablation studies on two-stage rollout, diversity reward, stage-wise clipping, and negative-sample down-weighting）— _截图占位_

### 6.2 拒绝采样微调（Rejection Sampling Fine-Tuning, RFT）

RFT 通过选择域专精教师的验证成功轨迹来整合域特定知识：

 \\mathcal{D}\_{\\mathrm{RFT}} = \\{(x\_u, \\mathrm{CoT}\_{u}, \\mathbf{c}\_{u})\\} \\tag{15}

 \\mathcal{L}\_{\\mathrm{RFT}} = -\\mathbb{E}\_{(x\_u, \\mathrm{CoT}\_{u}, \\mathbf{c}\_{u}) \\sim \\mathcal{D}\_{\\mathrm{RFT}}} \\sum\_{t=1}^{|y\_u|} \\log \\pi\_{\\theta}(y\_{u,t} \\mid x\_u, y\_{u,<t}), \\quad y\_u = \[\\mathrm{CoT}\_{u}; \\mathbf{c}\_{u}\] \\tag{16}

> **RFT 不是从 SFT 后端直接起点**。论文明确：首先在所有域的 R2 / R3 混合数据上运行 GRPO 至收敛，获得 **Mix-RL checkpoint**，再以此为起点在 **D\_RFT** 上进一步下一 token 预测训练。
> **图 15：RFT 相对于 Mix-RL 的相对增益**（Figure 15: Relative gains of RFT over Mix-RL across Recall@K · 关键数据：RFT 在较大 K 值时展示更大改善）— _截图占位_

### 6.3 多教师在线策略蒸馏（Multi-Teacher On-Policy Distillation, MOPD）

MOPD 以在线方式将四个域的专精能力整合到统一的学生模型中。采用 **Monte Carlo RL** 公式：对每一个提示 **x**，首先从行为策略 **μ\_θ** 采样完整 rollout **y = (CoT; c)**，再根据 **x** 的能力域路由到对应的域专家教师 **π\_{domain\_i}**。token 级蒸馏优势定义为：

 \\hat{A}\_{\\mathrm{MOPD},t} = \\mathrm{sg}\\left\[\\log\\pi\_{\\mathrm{domain}\_i}(y\_t \\mid x, y\_{<t}) - \\log\\pi\_\\theta(y\_t \\mid x, y\_{<t})\\right\] \\tag{17}

为校正 off-policy 偏差，引入**截断重要性权重**：

 w\_t(\\theta) = \\begin{cases} \\mathrm{sg}\\left\[\\dfrac{\\pi\_\\theta(y\_t \\mid x, y\_{<t})}{\\mu\_\\theta(y\_t \\mid x, y\_{<t})}\\right\], & \\varepsilon\_{\\mathrm{low}} \\le \\dfrac{\\pi\_\\theta}{\\mu\_\\theta} \\le \\varepsilon\_{\\mathrm{high}} \\\\ 0, & \\text{otherwise} \\end{cases} \\tag{18}

代理损失：

 \\mathcal{L}\_{\\mathrm{MOPD}}(\\theta) = -\\mathbb{E}\_{x \\sim \\mathcal{D}, y \\sim \\mu\_\\theta(\\cdot \\mid x)} \\left\[\\frac{1}{|y|} \\sum\_{t=1}^{|y|} w\_t(\\theta) \\hat{A}\_{\\mathrm{MOPD},t} \\log \\pi\_\\theta(y\_t \\mid x, y\_{<t})\\right\] \\tag{19}

**信息增益感知轨迹过滤**：计算每条轨迹的信息量得分：

 s(y\_j) = \\frac{1}{T\_j} \\sum\_{t=1}^{T\_j} |\\hat{A}\_{\\mathrm{MOPD},j,t}| \\tag{20}

确定覆盖目标信息增益比 **ρ** 的最小前缀长度 **M**：

 M = \\min \\left\\{ m : \\frac{\\sum\_{j=1}^m s(y\_{(j)})}{\\sum\_{j=1}^{|\\mathcal{B}|} s(y\_{(j)})} \\ge \\rho \\right\\} \\tag{21}

> **MOPD 的 reverse-KL 本质决定了其能力上限**：reverse-KL 是 mode-seeking 的，不完美复制教师的全分布。因此实验中在 **K ≤ 16** 时学生可以完美追上甚至超过教师，但在 **K > 16** 时仍会保留一个 long-tail 差距。
> **图 16-18：MOPD 性能分析**（Figures 16-18: MOPD relative Recall@K changes, IG filter effects, and student selection comparison）— _截图占位_

### 6.4 比较与讨论

> **表 9：不同优化策略的性能比较**（Table 9: Performance comparison of different optimization strategies · 关键数据：SFT→Mix-RL→Single-RL→RFT/MOPD 在四个跨域推荐任务上的 Recall@K 比较）— _截图占位_

关键发现：

**①** 所有后 SFT 优化方法都显著优于 SFT 基线，表明自我探索学习可以超越监督模仿。

**②** 直接混合所有域进行 RL 并不一致地达到最佳性能；与域特定 RL 相比，混合域 RL 通常较弱。

**③ RFT 和 MOPD 提供互补收益**：

**RFT**　通过离线多次 rollout 和严格保留正确结果来绕过稀疏奖励的探索瓶颈，蒸馏出”黄金”推理路径，**稳定保证思考模式 > 非思考模式**。

**MOPD**　依赖在线策略蒸馏，同步提升思考和非思考能力；在 Product 和 Live 域表现更优。

* * *

### 7\. CoT 分析指标（CoT Analysis Indicators）

我们设计了一个**四准则诊断框架**来评估生成的 CoT 痕迹质量，沿两个正交轴组织：**符号检查 vs. 概率检查** × **局部参考 vs. 全局参考**。

**准则 1：CoT 似然增益（**ΔLL**）**

 \\Delta\\mathrm{LL} = \\log p(y\_{\\mathrm{GT}} \\mid \\mathbf{x}, c) - \\log p(y\_{\\mathrm{GT}} \\mid \\mathbf{x}) \\tag{22}

> \*\*图 19：SFT 和 RFT 模型的平均 **ΔLL**\*\*（Figure 19: Mean **ΔLL** · 关键数据：SFT 在所有域产生负值，RFT 在所有域产生正值）— _截图占位_
> SFT 模型一致产生**负** **ΔLL** 值——表明其生成的 CoT 倾向于分散模型对真实目标预测的注意力。相反，RFT 模型在所有四个域都达到**正** **ΔLL** 值——证明 RFT 显著改善了 CoT 对目标预测的积极贡献。 _— “The SFT model consistently yields negative ΔLL values in all domains, indicating that its generated CoT tends to distract the model from the ground-truth target prediction. In contrast, the RFT model achieves positive ΔLL values across all four domains.”_

**准则 2：沿 CoT 的对数似然进展（**ℓ\_t\*\* 进展）\*\*

 \\ell\_t = \\log p(y\_{\\mathrm{GT}} \\mid \\mathbf{x}, c\_1, \\ldots, c\_t), \\quad t = 1, \\ldots, T \\tag{23}

> **图 20：沿 CoT 前缀的条件对数似然**（Figure 20: **ℓ\_t** along CoT prefixes · 关键发现：RFT 比 SFT 更早饱和——仅需少量推理段即可达到近最优似然）— _截图占位_

**准则 3：物品合法性（**γ\_legal**）**

 \\gamma\_{\\mathrm{legal}} = \\frac{|\\mathcal{S}(c) \\cap \\mathcal{V}\_{\\mathrm{item}}|}{|\\mathcal{S}(c)|} \\tag{24}

SFT 和 RFT 在所有域均饱和于 100%。

**准则 4：历史物品引用有效性（**γ\_{hist|legal}**）**

 \\gamma\_{\\mathrm{hist}|\\mathrm{legal}} = \\frac{|(\\mathcal{S}(c) \\cap \\mathcal{V}\_{\\mathrm{item}}) \\cap \\mathcal{S}(\\mathbf{x}\_{\\mathrm{hist}})|}{|\\mathcal{S}(c) \\cap \\mathcal{V}\_{\\mathrm{item}}|} \\tag{25}

> **表 10：CoT 有效性性能比较**（Table 10: Cross-domain **γ\_legal** and **γ\_{hist|legal}** · 关键数据：**γ\_legal** 在 SFT 和 RFT 中均为 100%；RFT 在 Video 和 Ad 上收紧 **γ\_{hist|legal}**）— _截图占位_

**定性案例：SFT vs. RFT 在同一用户上**

> **表 11-12：SFT 和 RFT 推荐 CoT 案例对比**（Tables 11-12: SFT vs RFT recommendation CoT cases · SFT 停留在表面 IP 层（和平精英），RFT 提升到潜在品类轴（战术竞技）并通过弱广告点击信号桥接到三角洲行动）— _截图占位_
> 两个痕迹的关键差异不是词汇润色，而是**抽象层次**：SFT 在表面 IP 名称上推理（和平精英→更多和平精英），而 RFT 在潜在品类轴上推理（”战术竞技”），因此能从观察到的证据桥接到一个只被弱暴露过的目标 IP。 _— “the difference between the two traces is not lexical polish but abstraction level: SFT reasons over surface IP names, whereas RFT reasons over a latent category axis and is therefore able to bridge from observed evidence to a target IP that has only been weakly exposed in history.”_

* * *

### 8\. 实验（Experiments）

### 8.1 性能比较

**基线方法**分为三类：

**①基于 ID 的**（ID-Based）　SASRec、HSTU。

**②基于文本的**（Text-Based）　Qwen3-8B/32B/235B-A22B、Deepseek-V3.2、Claude-Opus-4.6、Gemini-3-Preview、GPT-4o-mini、GPT-5.4。

**③基于物品 Token 的**（Itemic Token-Based）　TIGER、LC-Rec 多种变体、OneReason 多种变体。

> **表 13：跨域推荐基线比较**（Table 13: Baselines on cross-domain recommendation · 关键数据：OneReason RFT thinking 在所有四个域上达到 SOTA——Cross-Video Pass@64=2.41%, Cross-Product Pass@64=5.47%, Cross-Ad Pass@64=17.78%, Cross-Live Pass@64=21.10%）— _截图占位_

关键发现：

**①冷启动敏感性**　传统 ID 基模型在跨域评估中挣扎，因为 33.69% 的目标物品 ID 在训练中未见过；相比之下，仅 11.55% 的物品模式（Itemic Pattern）未见过。

**②文本基 LLM 推荐的局限**　更先进的通用 LLM 并不一定产生更好的推荐——推荐能力与通用智能或模型规模不可靠相关。

**③OneReason 预训练的效果**　LC-Rec-PT-SFT-8B（使用我们的预训练检查点初始化）在所有域上实现了明显的性能跳跃。

> **表 14：R0–R2 任务性能比较**（Table 14: Performance comparison on R0-R2 tasks · 关键数据：OneReason 在紧凑物品 token 的输入信息劣势下，在 R2 套件上超越了多个更大的先进 LLM）— _截图占位_
> **表 15：通用基准性能比较**（Table 15: General benchmark performance · 关键数据：OneReason RFT thinking 在 MMLU-Pro 72.08%、GPQA-Diamond 54.04%、MATH-500 95.40%、GSM8K 94.69%——与 Qwen3-8B 持平，而 LC-Rec 变体严重退化）— _截图占位_

### 8.2 思考监督带来的非思考增益（Non-Thinking Gains from Thinking Supervision）

我们观察到一个重要的下游现象：**思考监督可以改善直接推荐——即使在推理时抑制显式推理痕迹**。

**Token 对齐实验**：在相同 0.25B token 预算下，用 CoT + unCoT 混合替换部分 unCoT 数据，在 Cross-Video、Cross-Product 和 Cross-Live 上提升了非思考性能。

> **表 16：Token 对齐的非思考推荐性能比较**（Table 16: Token-aligned comparison · 关键数据：CoT+unCoT 混合使 Cross-Live Pass@64 从 18.12% 提升到 20.32%）— _截图占位_

**样本数控制混合扫描**：在固定 100K 推荐样本预算下，变化 CoT 和 unCoT 比例 **α**：

 \\alpha = \\frac{N\_{\\mathrm{unCoT}}}{N\_{\\mathrm{CoT}} + N\_{\\mathrm{unCoT}}} \\tag{26}

> **图 21：不同 CoT/unCoT 混合比下的非思考推荐增益**（Figure 21: Non-thinking recommendation gains across different CoT/unCoT mixture ratios · 关键发现：大多数域展现明确的中间最优点，Cross-Ad 例外——其曲线较平，最佳点偏向 unCoT 侧）— _截图占位_

我们将非思考增益概念性分解为：

 G\_d(\\alpha) = B\_d + A\_d^{\\mathrm{uncot}}(\\alpha) + I\_d^{\\mathrm{cot}}(1-\\alpha) - C\_d^{\\mathrm{trace}}(1-\\alpha) - C\_d^{\\mathrm{format}}(\\alpha, 1-\\alpha) \\tag{27}

> 高质量思考监督可以在适当混合比下改善大多数域的直接非思考推荐，但 Cross-Ad 表明 CoT 并非普遍有益——不应默认更激进地使用。这使 CoT 数据作为推荐基础模型**数据飞轮**的一部分有价值。 _— “High-quality thinking supervision can improve direct non-thinking recommendation in most domains under appropriate mixture ratios, but Cross-Ad shows that CoT is not universally beneficial and should not be used more aggressively by default.”_

* * *

### 9\. 部署（Deployment）

在快手 App 的**本地服务广告场景**（Local-Services Advertising）中，我们成功部署了 OneReason。在严格的延迟、强基线和物品可交付性约束下，系统实现了稳定的在线部署，并贡献了**快手本地服务广告场景中最显著的业务提升之一**。

### 9.1 工业场景适配（Industrial Scenario Adaptation）

快手本地服务广告场景不包含在 OneReason 训练数据中。我们提出了标准化的可复用持续训练流水线，实现快速域适配。

### 9.2 在线部署架构（Online Deployment Architecture）

> **图 22：OneReason 在线部署架构**（Figure 22: The online deployment architecture of OneReason）— _截图占位_

**“快-慢思考”架构（Fast-Slow Thinking Architecture）**：避免完全依赖实时推理，引入快速（在线）和慢速（近线）协作机制：

**Step 1 · 管线设计**　构建解耦的近线检索管线，与下游排序模型联合打分。

**Step 2 · 数据触发**　定期聚合分布式数据任务构建用户行为、画像和物品内容数据。

**Step 3 · 离线推理**　加载最新 OneReason 检查点，离线预测用户最可能的下一个物品 token。

**Step 4 · 在线服务**　解码后的物品 ID 写入 Redis，形成在线服务候选池。

**两种应用范式**：

**慢管线：OneReason**　直接使用 OneReason 进行检索（Slow Pipeline）。

**快管线：OneReason for OneRec**　利用 OneReason 输出增强 OneRec，引入 _Thinking Token_ 将 OneReason 知识蒸馏到在线 OneRec 中（Fast Pipeline）。

### 9.3 在线实验

进行了 10 天的在线 A/B 实验，5% 流量分配。

> **表 17：在线性能比较**（Table 17: Online performance comparison · 关键数据：OneReason 直接使用 Revenue +4.528%；OneReason for OneRec Revenue +4.636%、Impressions +6.831%；组合 Revenue +8.234%、Impressions +10.332%；ROI > 5）— _截图占位_
> 组合两种范式实现了最佳结果——Revenue 提升 8.234%，Impressions 提升 10.332%，对应快手平台**数亿元人民币的年化商业收入**。 _— “Combining both paradigms achieves the best results, with substantial gains in Revenue and Impressions, corresponding to hundreds of millions of RMB in annualized commercial revenue for the Kuaishou platform.”_

* * *

### 10\. 相关工作（Related Works）

推荐系统已越来越多地采用 LLM 来改善用户体验和业务表现。现有方法主要分为两个方向：

**① LLM 骨干范式**：TIGER \[[Rajput et al., 2023](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-rajput2023)\] 引入 RQ-VAE 基物品 token；HSTU \[[Zhai et al., 2024](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-zhai2024hstu)\] 用 Transformer 骨干替代传统多阶段管线；OneRec 系列将其工程化为统一的预训练和后训练框架。

**② 推荐基础模型**：LC-Rec \[[Zheng et al., 2024](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-zheng2024adapting)\] 表明 LLM 可通过物品 ID 对齐理解推荐语义；OpenOneRec \[[Zhou et al., 2026](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-zhou2026openonerec)\] 提出”先思考再推荐”范式。

> 与最直接可比的诊断工作 \[[Zhang et al., 2026](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-zhang2026whythinkinghurts)\] 相比——后者正式测量了 OpenOneRec 上的非思考/思考差距并将其归因于来自通用子空间的文本惯性——我们在训练时跨精心设计的训练管线解决了这一问题，以引导更鲁棒的物品感知和认知 CoT。 _— “In OneReason, we address this issue at training time across elaborate training pipelines to guide more robust itemic perception and cognition CoT.”_

* * *

### 11\. 结论与未来工作（Conclusion & Future Work）

在本工作中，我们提出了 **OneReason**，一个推理基础模型，将大语言模型和推荐系统连接为一个单一模型，赋予推荐以智慧。

具体而言，我们进行了广泛的修改：

**① 预训练阶段**：使用内容理解任务（而非对比监督）获取离散物品 token，并使用 **578B tokens** 对齐物品 token 和文本 token 的语义空间。

**② SFT 阶段**：设计标准化的粗到细、逻辑连贯的 CoT 结构以确保思考痕迹的质量。

**③ RL 阶段**：执行”先专精后统一”方案以改善思考模式能力，同时平衡多服务性能。

> 据我们所知，**OneReason 是第一个思考模式在下游推荐基准上一致优于非思考模式的工作**，表明推理可以转化为真正的推荐增益。 _— “to our knowledge, OneReason is the first work in which the thinking mode consistently outperforms the non-thinking mode on downstream recommendation benchmarks, suggesting that reasoning can be translated into real recommendation gains.”_

未来，我们将探索：

**①** 更稳定的**在线增量学习**机制，持续适应新的通用知识语料和用户兴趣漂移。

**②** 更定制化的**解码和生成策略**，实现吞吐量、延迟、质量和商业价值之间的更好平衡。

**③** 从”先思考再回答”迈向具有**规划和工具调用能力**的 Agent 推荐基础模型。

> 我们相信，随着模型压缩、推理基础设施和硬件效率的持续进步，推荐基础模型将成为推荐链的核心组件，逐步推动向 **Agent 推荐器架构**的演进。 _— “We believe that, with continued progress in model compression, inference infrastructure, and hardware efficiency, recommendation foundation models will become a core component of the recommendation chain, gradually driving the evolution toward an agentic-recommender architecture.”_

* * *

### 术语对照表（Glossary）

-   **生成式推荐模型** — Generative Recommendation Model（首次出现：Abstract / §1）
-   **物品 Token / 物品模式** — Itemic Token / Itemic Pattern（Abstract / §4.1）
-   **思维链** — Chain-of-Thought, CoT（Abstract / §2）
-   **思考模式** — Thinking Mode（Abstract / §1）
-   **非思考模式** — Non-Thinking Mode（Abstract / §1）
-   **感知** — Perception（Abstract / §2）
-   **认知** — Cognition（Abstract / §2）
-   **推导** — Derivation（§2）
-   **演变** — Evolution（§2）
-   **先专精后统一** — Specialize-then-Unify（Abstract / §6）
-   **拒绝采样微调** — Rejection Sampling Fine-Tuning, RFT（§6.2）
-   **多教师在线策略蒸馏** — Multi-Teacher On-Policy Distillation, MOPD（§6.3）
-   **监督微调** — Supervised Fine-Tuning, SFT（§5）
-   **强化学习** — Reinforcement Learning, RL（§6）
-   **组相对策略优化** — Group Relative Policy Optimization, GRPO（§6.1）
-   **画像抽象** — Persona Abstraction（§5.4）
-   **兴趣展开** — Interest Expansion（§5.4）
-   **转移推断** — Transition Inference（§5.4）
-   **快-慢思考架构** — Fast-Slow Thinking Architecture（§9.2）
-   **信息增益感知轨迹过滤** — Information-Gain-Aware Trajectory Filtering（§6.3）
-   **物品分词器** — Itemic Tokenizer（§4.1）
-   **Token 粒度** — Token Granularity（§4.2）
-   **物品粒度** — Item Granularity（§4.2）
-   **关系粒度** — Relational Granularity（§4.2）
-   **用户粒度** — User Granularity（§4.2）
-   **容量感知描述粗粒化** — Capacity-Aware Caption Coarse-Graining（§4.2）
-   **CoT 似然增益** — CoT Likelihood Gain, **ΔLL**（§7）
-   **物品合法性** — Item Legality, **γ\_legal**（§7）
-   **历史物品引用有效性** — History Item Reference Validity, **γ\_{hist|legal}**（§7）
-   **大语言模型** — Large Language Model, LLM（§1）
-   **多模态大语言模型** — Multi-modal Large Language Model, MLLM（§1）

* * *

### 参考文献（References）

1.  J. Kaplan et al. **[Scaling laws for neural language models.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2001.08361)** _《神经语言模型的缩放定律》 · arXiv 2020 · JHU · 揭示模型规模、数据和算力的缩放规律。_
2.  J. Hoffmann et al. **[Training compute-optimal large language models.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2203.15556)** _《训练计算最优大语言模型（Chinchilla）》 · arXiv 2022 · DeepMind · 提出数据与参数的最优缩放比。_
3.  L. Ouyang et al. **[Training language models to follow instructions with human feedback.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2203.02155)** _《通过人类反馈训练语言模型遵循指令（InstructGPT）》 · NeurIPS 2022 · OpenAI · RLHF 范式奠基。_
4.  Z. Shao et al. **[DeepSeekMath: Pushing the limits of mathematical reasoning in open language models.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2402.03300)** _《DeepSeekMath》 · arXiv 2024 · DeepSeek · 提出 GRPO 算法用于数学推理。_
5.  J. Wei et al. **[Chain-of-thought prompting elicits reasoning in large language models.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2201.11903)** _《思维链提示引发大语言模型推理》 · NeurIPS 2022 · Google · CoT 范式奠基工作。_
6.  T. Kojima et al. **[Large language models are zero-shot reasoners.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2205.11916)** _《大语言模型是零样本推理者》 · NeurIPS 2022 · 东京大学 · 零样本 CoT 推理。_
7.  A. Jaech et al. **[OpenAI o1.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2412.16720)** _《OpenAI o1 系统》 · 2024 · OpenAI · 推理增强大模型。_
8.  D. Guo et al. **[DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.12948)** _《DeepSeek-R1》 · arXiv 2025 · DeepSeek · 通过 RL 激励推理能力。_
9.  Y. Zhou et al. **[OneRec technical report.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2502.18965)** _《OneRec 技术报告》 · arXiv 2025 · 快手 · OneRec V1 生成式推荐。_
10.  X. Liu et al. **[OneRec-Think: In-text reasoning for generative recommendation.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2505.06440)** _《OneRec-Think》 · arXiv 2025 · 快手 · 生成式推荐中的文本内推理。_
11.  Y. Zhou et al. **[OpenOneRec technical report.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2505.15863)** _《OpenOneRec 技术报告》 · arXiv 2026 · 快手 · 开放生成式推荐基础模型与基准。_
12.  A. Radford et al. **[Learning transferable visual models from natural language supervision.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2103.00020)** _《CLIP：从自然语言监督学习可迁移视觉模型》 · ICML 2021 · OpenAI · 视觉-语言对比学习。_
13.  A. Yang et al. **[Qwen3 technical report.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2505.09388)** _《Qwen3 技术报告》 · arXiv 2025 · 阿里巴巴 · Qwen3 系列模型。_
14.  Y. Luo et al. **[QARM: Quantitative alignment with residual matching.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2501.12548)** _《QARM》 · arXiv 2025 · RQ-KMeans 量化方法。_
15.  S. Rajput et al. **[Recommender systems with generative retrieval.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2305.05065)** _《TIGER：基于生成式检索的推荐系统》 · NeurIPS 2023 · Google · 生成式推荐先驱工作。_
16.  J. Zhai et al. **[HSTU: Actions speak louder than words.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2402.17152)** _《HSTU》 · arXiv 2024 · Meta · 大规模 Transformer 推荐系统。_
17.  P. Zheng et al. **[Adapting large language models by integrating collaborative semantics for recommendation.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2311.09049)** _《LC-Rec》 · ICME 2024 · 大语言模型适配推荐。_
18.  B. Sun et al. **[Reading or thinking: Understanding and bridging the gap.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2505.13937)** _《阅读还是思考》 · arXiv 2026 · 深度跨模态对齐是推理先决条件。_
19.  Z. Zhou et al. **[Perception and cognition: A survey of vision-language models.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2411.13824)** _《感知与认知》 · arXiv 2025 · MLLM 的感知到认知路径综述。_
20.  X. Jiang et al. **[CORVID: Improving multimodal large model reasoning.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2505.14095)** _《CORVID》 · arXiv 2025 · 粗到细推理格式防止幻觉。_
21.  Z. Zhang et al. **[Why thinking hurts: Understanding the failure mode.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2506.00007)** _《为什么思考会伤害》 · arXiv 2026 · 分析 OpenOneRec 上非思考/思考差距的文本惯性。_
22.  W.-C. Kang, J. McAuley. **[Self-attentive sequential recommendation.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1808.09781)** _《SASRec：自注意力序列推荐》 · ICDM 2018 · UCSD · 经典序列推荐模型。_
23.  A. Vyas et al. **[Pushing the limits of audio encoders.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2506.00000)** _《音频编码器》 · 2026 · 音频编码技术。_
24.  N. Tishby, N. Zaslavsky. **[Deep learning and the information bottleneck principle.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1503.02406)** _《深度学习与信息瓶颈原理》 · ITW 2015 · 信息瓶颈理论在深度学习中的应用。_
25.  Z. Yuan et al. **[Scaling relationship on learning mathematical reasoning with large language models.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2308.01825)** _《拒绝采样微调》 · arXiv 2023 · RFT 方法。_
26.  Y. Xiao et al. **[MIMO: Multi-teacher on-policy distillation.](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2505.13291)** _《MIMO：多教师在线策略蒸馏》 · arXiv 2026 · 多教师知识整合。_

* * *

### AI 解读：OneReason 的工业意义

### 综合评分 8.4⁄10

-   **工业落地价值**：9/10 — 已在快手本地服务广告场景部署，Revenue +8.234%，ROI > 5，直接产生数亿元年化收入
-   **问题定义新颖性**：8/10 — 首次系统性诊断推荐推理中”思考模式不如非思考模式”的根本原因，并提出感知+认知双支柱框架
-   **方法论创新性**：9/10 — 四粒度预训练数据、三阶段 CoT 结构（画像抽象→兴趣展开→转移推断）、先专精后统一 RL 策略、阶段感知裁剪、负样本降权、truncated importance weighting、information-gain-aware filtering 等多处深层创新
-   **实验严谨性**：9/10 — 4 准则 2×2 诊断矩阵（symbolic/probabilistic × local/global）、详尽的消融实验覆盖每个设计选择、与多种基线（含 GPT-5.4、Claude-Opus-4.6）的全面对比
-   **可复现性**：7/10 — 承诺开源 OneReason-8B 和 0.8B 模型，但核心数据为快手内部数据
-   **写作清晰度**：7/10 — 论文极长（约 60+ 页含附录），结构清晰但信息密度极高

加权计算：**0.25 × 9 + 0.15 × 8 + 0.20 × 9 + 0.20 × 9 + 0.10 × 7 + 0.10 × 7 = 8.45 ≈ 8.4**

### 核心思想提炼

**一句话总结**：OneReason 是第一个在工业推荐场景中让”思考模式”真正胜过”不思考”的推荐基础模型——通过深度物品-文本感知对齐、结构化认知 CoT 和先专精后统一的 RL 策略。

**解决什么问题**：此前的生成式推荐模型（如 OpenOneRec）虽然能让模型”思考”（生成推理痕迹），但思考模式在推荐指标上并不优于直接预测。这就像给一个还没看清题目的学生要求”展示解题过程”——解题步骤反而成了噪声。

**怎么解决**： **① 看清题目（感知）**：用 578B token 的四粒度预训练数据，从子 token 语义到用户行为全面对齐物品模态和文本模态

**② 学会解题（认知）**：设计三阶段 CoT（画像抽象→兴趣展开→转移推断），把推荐推理从”列兴趣清单”升级为”压缩-假设-判断”

**③ 独立练习（RL）**：先在每个域单独 RL 训练”思考”技能，再通过 RFT/MOPD 合并为一个全能模型

**关键数字**：

-   思考模式在所有跨域推荐基准上**首次一致超越**非思考模式
-   快手本地服务广告场景 Revenue **+8.234%**，年化收入提升数亿元
-   RFT 模型的 CoT 似然增益 **ΔLL** 从 SFT 的负值**翻转**为正值
-   通用能力保持：MMLU-Pro **72.08%**（Qwen3-8B 基线 72.35%）

### 工程细节（值得借鉴的实践点）

**① 容量感知描述粗粒化**　仅 3 个子 token 无法编码过于详细的描述（如 OCR 文本、精确价格），强行对齐会导致幻觉。按域设计粗粒化规则是务实的工程选择。

**② 两阶段 Rollout**　在推荐 RL 中，先采样 **N** 个推理痕迹，再在每个痕迹上扩展 **K** 个物品预测，将 **N × K** 个有效 rollout 的成本降为 **N** 次长推理生成。

**③ 阶段感知裁剪**　推理 token 用宽松裁剪促进探索，物品 token 用紧裁剪防止分布崩塌——这是 RL 训练稳定性的关键。

**④ 快-慢思考部署架构**　不需要实时运行大模型推理。慢管线离线批量推理 + Redis 缓存 + 快管线实时融合，实现 ROI > 5 的工业可行性。

**⑤ CoT 数据的飞轮效应**　发现即使在非思考推理时，CoT 监督也能提升性能——这意味着 CoT 数据构建是一次性投入但持续产出的资产。

### 后续路线图

**① 在线增量学习**　持续适应新物品和用户兴趣漂移。

**② 定制化解码策略**　平衡吞吐量、延迟、质量和商业价值。

**③ Agent 推荐基础模型**　从“思考再回答”迈向具备规划和工具调用能力的 Agent。

**④ 推理压缩**　RFT 的 CoT 似然进展在少量推理段后即饱和，提示自适应早停的可能。

**⑤ 更细粒度的域特定 CoT/unCoT 混合比优化**　Cross-Ad 的特殊表现表明不同域需要不同策略。

### 延伸阅读

**OpenOneRec** \[[Zhou et al., 2026](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-zhou2026openonerec)\]　_OneReason 的前身，首次提出“先思考再推荐”范式_

**OneRec-Think** \[[Liu et al., 2025](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-liu2025onerecthink)\]　_生成式推荐中文本内推理的早期探索_

**TIGER** \[[Rajput et al., 2023](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-rajput2023)\]　_生成式推荐先驱，引入 RQ-VAE 物品 token_

**DeepSeek-R1** \[[Guo et al., 2025](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-guo2025deepseek)\]　_RL 驱动推理能力的 LLM 范式_

**LC-Rec** \[[Zheng et al., 2024](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-zheng2024adapting)\]　_LLM 适配推荐的物品 ID 对齐方法_

**Why Thinking Hurts** \[[Zhang et al., 2026](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-zhang2026whythinkinghurts)\]　_诊断推荐推理中思考模式失效的根因分析_

**CORVID** \[[Jiang et al., 2025](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-jiang2025corvid)\]　_MLLM 中粗到细 CoT 格式防止推理幻觉_

**HSTU** \[[Zhai et al., 2024](https://zhuanlan.zhihu.com/p/2048412422847870910/edit#ref-zhai2024hstu)\]　_Meta 的大规模 Transformer 推荐系统_

* * *

> **如果这篇精读对你有帮助，欢迎点赞 / 收藏 / 关注**——后续会继续重读 OpenOneRec / TIGER / HSTU / DeepSeek-R1 / OneRec-Think 等生成式推荐与推理基础模型的里程碑工作。