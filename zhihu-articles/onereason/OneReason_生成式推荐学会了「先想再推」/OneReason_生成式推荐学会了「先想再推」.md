# OneReason:生成式推荐学会了「先想再推」
> 发布时间: 发布于 2026-06-08 09:22・北京
> 原文链接: https://zhuanlan.zhihu.com/p/2047245741110867954

---

[生成式推荐](https://zhida.zhihu.com/search?content_id=276439520&content_type=Article&match_order=1&q=%E7%94%9F%E6%88%90%E5%BC%8F%E6%8E%A8%E8%8D%90&zhida_source=entity)(Generative Recommendation, GR)最尖锐的问题，不是模型能不能把下一个物品(item)生成出来，而是它是否知道**为什么应该生成这个物品**。

[OneRec](https://zhida.zhihu.com/search?content_id=276439520&content_type=Article&match_order=1&q=OneRec&zhida_source=entity)系列已经把端到端生成式推荐推到了短视频、直播、广告、电商等真实业务中。论文《OneReason Technical Report》进一步追问:既然[大语言模型](https://zhida.zhihu.com/search?content_id=276439520&content_type=Article&match_order=1&q=%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity)(Large Language Model, LLM)可以通过“先思考后回答”(think-before-answer)激活推理能力，推荐模型是否也能通过[思维链](https://zhida.zhihu.com/search?content_id=276439520&content_type=Article&match_order=1&q=%E6%80%9D%E7%BB%B4%E9%93%BE&zhida_source=entity)(Chain-of-Thought, CoT)获得同类收益？

论文给出的结论并不乐观，也因此更有价值:直接照搬LLM的CoT范式，在推荐任务上并不自然生效。原因在于推荐序列里的核心符号不是自然语言，而是[itemic tokens](https://zhida.zhihu.com/search?content_id=276439520&content_type=Article&match_order=1&q=itemic+tokens&zhida_source=entity)。它们像压缩后的语义坐标，能承载协同过滤和内容信息，却很难天然组成一段可解释、可推理、可监督的“思考过程”。

OneReason要解决的，正是这个断点。

### 01 问题:item-token的生成式推荐为什么难做CoT

推荐不是数学题，也不是代码题。数学推理通常存在相对明确的中间步骤，代码推理可以围绕约束、变量、执行路径展开；推荐则不同，它面对的是用户行为日志、候选物品集合、跨域兴趣迁移和多个合理答案。

论文将推荐推理定义为一种**[溯因推理](https://zhida.zhihu.com/search?content_id=276439520&content_type=Article&match_order=1&q=%E6%BA%AF%E5%9B%A0%E6%8E%A8%E7%90%86&zhida_source=entity)(abductive reasoning)**:模型不是从公理推出唯一答案，而是从用户历史中假设潜在兴趣点(latent interest points)，再判断这些兴趣如何演化，并解释为什么某个后续物品更可能承接当前状态。

1.  **itemic tokens是语义压缩，不是自然语言推理链**

OneReason使用的itemic pattern形如:

```python
<|domain_begin|><a_5028><b_6733><c_2559>
```

其中domain覆盖video、prod、ad、living等业务域。论文采用三层codebook，每层8192个codes，每个item由1个domain-aware begin token加3个sub-tokens表示。

这个表示很紧凑，但也带来一个关键问题:如果CoT只由这类token组成，它很难像自然语言一样表达“需求是什么、证据是什么、兴趣如何迁移”。也就是说，itemic tokens可以很好地做生成，却不能自动构成有意义的思维链。

1.  **直接套LLM的think-before-answer不显示优势**

OneRec-Think和OpenOneRec已经探索过将itemic-text推荐数据与通用推理数据结合，让模型在推荐任务中生成思考过程。但论文指出，一个意外现象是:thinking mode相对non-thinking mode并未稳定显示优势。

这不是CoT无效，而是推荐场景下CoT的前提没有补齐。没有足够强的item语义感知，模型只是在符号表面游走；没有推荐专用的认知结构，模型很容易生成流畅但无用、甚至引入噪声的推理文本。

1.  **推荐CoT的核心不是“说得更长”，而是“压缩得更准”**

推荐历史通常长、杂、跨域，直接把所有行为塞进推理链，只会制造噪声。OneReason的判断是:推荐CoT必须完成两件事。

感知(perception):把itemic tokens对齐到可理解的自然语言语义。

认知(cognition):把用户行为序列重组为少数相干兴趣点，并判断这些兴趣点如何转移到目标域。

这也是OneReason和普通“LLM写推荐理由”的根本区别。后者往往是在答案附近补解释；OneReason试图把解释变成训练信号，并让它服务于itemic token生成。

### 02 关键洞察:推理能力来自感知+认知

OneReason借鉴[多模态LLM](https://zhida.zhihu.com/search?content_id=276439520&content_type=Article&match_order=1&q=%E5%A4%9A%E6%A8%A1%E6%80%81LLM&zhida_source=entity)(Multimodal LLM, MLLM)的失败经验:视觉-文本没有对齐时，模型会读错图；对齐了但CoT结构不好时，模型仍会推理漂移。推荐也是类似问题，只是“视觉对象”换成了itemic tokens，“图像语义”换成了物品内容、协同关系和用户行为上下文。

1.  **R0: Perception，先让token可读**

R0关注itemic tokens能否被解释为明确语义。OneReason-Bench在这一层设置了Item Understanding、Itemic Pattern Grounding、Item QA等任务，分别评估从token到文本、从文本到token、基于item内容回答问题的能力。

如果R0不成立，后续所有推理都是建立在不可靠语义上。论文的item理解案例显示，模型可能从一个短剧视频token中幻觉出错误剧名，甚至把都市情感短剧误判为医疗题材。这个问题不是表达风格问题，而是感知对齐问题。

1.  **R1-R3: Cognition，把行为变成可操作兴趣结构**

OneReason把推荐认知拆成三个递进层次。

R1 Derivation:从单个item语义推导item-to-item关系，解释为什么一个物品可能自然引向另一个物品。

R2 Evolution:从时间序列中识别兴趣演化，判断触发、细化、收敛、替代等过程。

R3 Recommendation:综合用户画像、历史行为和目标域，生成下一步推荐itemic tokens。

这套分层的价值在于:它把“推荐推理”从一个模糊概念拆成可训练、可评估、可消融的能力栈。OneReason-Bench因此不是单纯排行榜，而是模型训练过程的诊断仪表盘。

### 03 方法:OneReason的三件套

OneReason的方法可以压缩成三句话:预训练补感知，SFT补认知，RL解决多域thinking优势不稳定的问题。

1.  **预训练:四粒度语料强化itemic-token perception**

预训练阶段的核心是四粒度推荐语料。

Token Granularity:建模sub-token及其组合语义，例如prefix sub-token pair如何组合成更高层语义。

Item Granularity:做itemic pattern与自然语言描述的双向对齐，并引入多视角Item QA。

Relational Granularity:把item-to-item协同关系改写成带自然语言解释的序列，让协同信号进入文本语义空间。

User Granularity:用用户画像、分域行为序列、时间交错行为序列，建模跨域与长期兴趣。

量化上，预训练采用三阶段recipe:Stage 1训练新增词表和LM head，token budget为110B；Stage 2全参数训练，449B tokens；Stage 3提升单样本长度到32K，19B tokens。总计578B tokens。packed sequence长度为48K。

消融实验使用0.8B模型和固定30B token预算。Table 2显示，加入Token粒度后，R1 Item2Item QA从0.00%升至20.57%；最终引入User粒度后，R1达到29.72%，R2 Evolution Direct Gen达到0.37%，Cross-Live从3.25%升至8.56%，Cross-Ad从8.58%升至10.84%。这说明用户粒度不是简单补数据，而是在把前面的token、item、relation能力放回真实时间行为中整合。

![](https://pic1.zhimg.com/v2-961ae1bf5b65c384c8cd92dcc5f7525e_1440w.jpg)

![](https://pica.zhimg.com/v2-47643763a8be2cfb6e5c645554bb5432_1440w.jpg)

Table 2 —— 多粒度认知增强(Token/Item/Relation/User)消融

1.  **SFT:三层认知增强CoT**

监督微调(Supervised Fine-Tuning, SFT)阶段围绕R0-R3构造数据。论文给出的规模信息较具体:R0约941K样本，其中约682K caption样本、259K QA样本；R1从358K TagNext CF pairs和388K after-play-search pairs中采样约400K训练样本；R2约130K训练实例。

最关键的是R3推荐CoT。OneReason不是让模型泛泛解释“用户可能喜欢什么”，而是设计了三阶段trace construction:

Persona Abstraction:压缩稳定画像和偏好先验，例如家庭消费、直播购物敏感、共享设备等。

Interest Expansion:把近期行为展开为少数证据支撑的候选兴趣方向。

Transition Inference:比较证据强度、时序连续性、画像兼容性、目标域兼容性和泄漏风险，最后提交下一兴趣方向。

论文还专门校准了Interest Expansion的宽度。Figure 8显示，在Cross-Video、Cross-Product、Cross-Ad、Cross-Live上，n取1、3、5通常优于10或20。这个结果很重要:有效CoT不是分支越多越好，而是要把不确定性控制在可比较的小候选集内。

![](https://pic2.zhimg.com/v2-86d5f35d87af948e387931f997ff0b77_1440w.jpg)

Figure 8 —— Interest Expansion 宽度 n 的影响

1.  **RL:专化后统一，避免多域干扰**

强化学习(Reinforcement Learning, RL)阶段采用“specialize-then-unify”。论文先在四个推荐域分别做domain-specific RL，得到Video、Product、Ad、Live四个教师模型，再用两种方式整合回统一模型。

RFT(Rejection Sampling Fine-tuning):从各域教师中采样，保留命中ground-truth且推理质量合格的轨迹，聚合成高质量CoT数据继续训练。

MOPD(Multi-Teacher On-Policy Distillation):学生按自身策略生成轨迹，按任务域路由到对应教师，通过token-level reverse-KL信号进行在线蒸馏，并用information-gain-aware filtering过滤低信息轨迹。

推荐任务的奖励极稀疏。OneReason为GRPO(Group Relative Policy Optimization)设计了两阶段rollout:先采样N条reasoning traces，再对每条trace并行生成K个itemic token sequences，以N×K的有效rollout覆盖更多候选，同时只生成N段长CoT。论文还加入diversity reward、stage-wise clipping和negative-sample down-weighting，以稳定稀疏奖励下的训练。

### 04 实验、部署与局限

从实验结果看，OneReason最值得关注的不是“比某个baseline高多少”，而是它首次把thinking mode稳定转化为推荐收益。

1.  **RFT后thinking mode稳定超过non-thinking mode**

Table 14显示，OneReason RFT thinking在四个跨域推荐任务上均超过RFT non-thinking。以Pass@64 / Recall@64计:

![](https://pic1.zhimg.com/v2-12c72e45f9d3e21b0606f409b9b4aa54_1440w.jpg)

Table 14 —— 四个跨域任务 thinking vs non-thinking(Pass@64/Recall@64)

Cross-Video:RFT thinking为2.41 / 0.24，RFT non-thinking为2.08 / 0.19。

Cross-Product:RFT thinking为5.47 / 4.19，RFT non-thinking为5.20 / 3.96。

Cross-Ad:RFT thinking为17.78 / 7.50，RFT non-thinking为17.56 / 7.26。

Cross-Live:RFT thinking为21.10 / 18.35，RFT non-thinking为21.01 / 18.17。

这组结果对应论文的核心结论:直接SFT后，thinking未必优于non-thinking；经过specialize-then-unify RL，thinking优势才被稳定激活。

1.  **通用能力没有明显坍塌**

Table 16显示，OneReason RFT thinking在MMLU-Pro、GPQA-Diamond、MATH-500、GSM8K上分别为72.08、54.04、95.40、94.69；Qwen3-8B thinking对应为72.35、56.06、95.20、95.68。也就是说，在推荐训练后，OneReason仍基本保留了Qwen3-8B backbone的通用推理与指令能力。

1.  **CoT监督还能提升non-thinking inference，但机制未完全拆开**

Table 17给出一个有意思的发现:在0.25B token预算下，40K CoT + 50K unCoT的混合训练，相比100K unCoT，在Cross-Video、Cross-Product、Cross-Live上提升non-thinking表现。例如Pass@64从1.64升至1.95、4.38升至4.86、18.12升至20.32；Cross-Ad则略有下降，16.08到15.84。

论文对此保持克制:这只是行为证据，说明CoT监督中的压缩信号和推理信号可能迁移到直接解码，但不能证明收益究竟来自compression、reasoning，还是二者交互。

1.  **线上部署采用[Fast-Slow Thinking](https://zhida.zhihu.com/search?content_id=276439520&content_type=Article&match_order=1&q=Fast-Slow+Thinking&zhida_source=entity)**

OneReason已在快手App本地生活广告场景部署。考虑LLM实时推理成本，论文采用Fast-Slow Thinking架构:Slow近线链路用OneReason离线预测itemic tokens并写入Redis候选池；Fast在线链路通过OneReason for OneRec把OneReason输出蒸馏到OneRec的Thinking Token中，再由排序模型融合。

10天A/B实验中，处理组和对照组各5%流量。Table 18显示:OneReason单独用于Slow pipeline，Impressions +0.940%、Revenue +4.528%；OneReason for OneRec用于Fast pipeline，Impressions +6.831%、Revenue +4.636%；Combined方案达到Impressions +10.332%、Revenue +8.234%。论文还估算ROI>5，并在附录中披露全流量服务约400M用户时约消耗600张旗舰GPU/天。

![](https://pic4.zhimg.com/v2-be3b02f7aff3e779fedc8a3e7e21fe4b_1440w.jpg)

Table 18 —— 10 天线上 A/B:Impressions / Revenue 提升

局限也需要明确。第一，报告标注为Work in progress，部分工程细节、开源材料和更长期线上稳定性仍需等待后续版本。第二，CoT监督为何能迁移到non-thinking decoding，目前只有行为层证据。第三，Fast-Slow Thinking本质上仍是成本折中，真正低延迟实时推理仍依赖0.8B级小模型、压缩和推理系统进展。第四，在线增量训练可能影响通用指令能力，论文在Appendix A.4也将其列为未来工作。

### 05 Q&A

**Q1: OneReason是不是简单给OneRec加了CoT？**

不是。OneReason的主线是“感知+认知+RL统一”。CoT只是外在形式，真正的改动包括四粒度预训练、R0-R3能力分层、三阶段推荐CoT构造，以及specialize-then-unify RL。

**Q2: 为什么itemic tokens比自然语言更难推理？**

因为itemic tokens是压缩符号，天然适合生成和检索，不天然适合表达因果、时序和兴趣迁移。OneReason先通过578B tokens预训练把itemic-token空间和文本空间对齐，再通过SFT让模型学会把行为压缩成兴趣点。

**Q3: 论文里最有价值的实验结论是什么？**

我认为有两个。第一，RFT后thinking mode在四个跨域推荐任务上稳定超过non-thinking mode。第二，CoT+unCoT混合训练在多个域提升non-thinking inference，说明“学会思考”可能会改写模型参数中的用户状态表示，而不只是增加推理时token。

**Q4: RFT和MOPD哪个更重要？**

二者定位不同。RFT更像筛选“正确推理路径”后再监督学习，论文指出它能更稳定保证thinking>non-thinking；MOPD更像策略级蒸馏，能同步提升thinking和non-thinking，在Product、Live等域表现强。推荐系统需要覆盖多个候选，二者都服务于从单域教师回到统一模型。

**Q5: OneReason对工业系统的启发是什么？**

不要把推理模型只看成实时大模型接口。OneReason的线上形态是近线Slow与在线Fast协同:近线生成高价值候选，在线模型吸收reasoning knowledge并保持低延迟。对大规模推荐系统而言，这比“把LLM直接塞进主链路”更现实。

### 参考文献

1.  OneRec Team et al., **OneReason Technical Report**, arXiv:2606.06260, submitted on 2026-06-04, cs.IR/cs.AI/cs.CL, Work in progress. [https://arxiv.org/abs/2606.06260](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2606.06260)
2.  Zhou et al., **OpenOneRec**, 2026. 论文中作为OneReason前序工作与RecIF-Bench来源引用。
3.  Liu et al., **OneRec-Think**, 2025. 论文中作为早期推荐推理探索引用。
4.  Rajput et al., **TIGER: Generative Recommenders**, 2023. 论文中作为itemic-token生成式推荐基线引用。
5.  Kang and McAuley, **SASRec**, ICDM 2018. 论文中作为ID-based sequential recommendation基线引用。
6.  Zhai et al., **HSTU**, 2024. 论文中作为Transformer-based生成式推荐基线引用。