# OnePiece：首个全面落地推理能力的工业级生成式搜索框架！
> 发布时间: 发布于 2026-02-06 15:07・新加坡
> 原文链接: https://zhuanlan.zhihu.com/p/1990378803617371940

---

**2026年伊始，让我们回顾2025年，搜推广领域最亮眼的莫过于——[生成式推荐技术](https://zhida.zhihu.com/search?content_id=268472410&content_type=Article&match_order=1&q=%E7%94%9F%E6%88%90%E5%BC%8F%E6%8E%A8%E8%8D%90%E6%8A%80%E6%9C%AF&zhida_source=entity)。**

各个大厂都在更新迭代自己的模型。看起来发展十分迅猛，然而绝大多数工作都主攻生成式推荐基座模型训练方向，很少有工作聚焦在如何将当下[LLM](https://zhida.zhihu.com/search?content_id=268472410&content_type=Article&match_order=1&q=LLM&zhida_source=entity)技术框架中的“推理能力”迁移至推荐系统中。以CoT、[上下文工程](https://zhida.zhihu.com/search?content_id=268472410&content_type=Article&match_order=1&q=%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B7%A5%E7%A8%8B&zhida_source=entity)为代表的推理技术，占据LLM优化的半壁江山，将推理技术应用至推荐有如下好处：

-   强化模型对推荐会话“上下文”的感知，增强模型的个性化
-   小模型 + 推理 大模型，能推理的推荐模型可以有效分摊算力、参数压力

为了给推荐模型赋予推理能力，今年9月，傅聪团队联合人大高瓴学院发表了研究成果《[OnePiece](https://zhida.zhihu.com/search?content_id=268472410&content_type=Article&match_order=1&q=OnePiece&zhida_source=entity): Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System》。该成果在[Shopee Search](https://zhida.zhihu.com/search?content_id=268472410&content_type=Article&match_order=1&q=Shopee+Search&zhida_source=entity)主场景已取得显著收益，全流量生效。

![](https://pic4.zhimg.com/v2-c55afc270ce3835697e6c4e2c3a3f90f_1440w.jpg)

Onepiece是一个从推理能力的scaling出发，融合上下文工程、[隐式推理](https://zhida.zhihu.com/search?content_id=268472410&content_type=Article&match_order=1&q=%E9%9A%90%E5%BC%8F%E6%8E%A8%E7%90%86&zhida_source=entity)和多目标训练的生成式搜推框架。一经发布，就凭借强大的推理能力冲上了[Hugging Face Daily Paper](https://zhida.zhihu.com/search?content_id=268472410&content_type=Article&match_order=1&q=Hugging+Face+Daily+Paper&zhida_source=entity)榜单的前三，引起了海内外的广泛关注，昭示着生成式搜推领域进入了推理时代。

![](https://pic2.zhimg.com/v2-5a30248bbbc3c2e5b13c7cd05cd5ca43_1440w.jpg)

![](https://pic1.zhimg.com/v2-4042914214722c88aea8b65d1254a7fc_1440w.jpg)

1.  **生成式推荐特色的“上下文工程”**

提到“上下文工程”，很多人会自然联想到思维链（CoT）技术。CoT 的核心思想，是通过明确的 step-by-step 文本提示，让 LLM 按照人类推理路径进行思考，从而显著提升模型的逻辑能力。

但当我们把这个思路迁移到搜索推荐（尤其是电商场景）时，问题就变得不再简单。

在传统做法中，我们给生成式推荐模型输入的通常是一段用户行为序列，例如用户最近点击、购买过的商品 ID：item A、item B、item C、item D……

如果要模仿 CoT 的方式，我们就需要根据这些行为序列构造出“用户做下一次消费决策时的完整思考过程”，并以文本形式展示给模型，期望模型通过阅读这些步骤来推断用户的购买动机。

但这在实际中几乎不可行：

第一，每个人的购物思维逻辑都不同，难以还原；
第二，即便我们能人为构造出一套推理链条，这种“自然语言文本 + 商品序列 ID”的混合形式也显得非常不协调，输入序列因此变得“异构且不连贯”，模型往往难以理解。

那么，生成式推荐到底应该如何构造 Prompt？

这里不妨回到最朴素的思路 —— test-time few-shot learning。

Few-shot 的本质，简单说就是“举例子”，让模型通过阅读示例来获得举一反三的能力。后来的 Instruct-Following SFT，本质上也是把这些“示例”固化到训练语料中，让模型逐步学习指令跟随能力。

在推荐任务中，“示例”可以怎么构造？
我们的做法是引入一类非常有效的模式：锚点物品序列（anchor item sequences）。
例如在 Shopee Search 场景下，对于某个搜索词，我们可以收集大量用户的高频点击序列、高频下单序列，并把这些序列作为“专家示例（domain expert knowledge）”拼接在目标用户的交互序列后面。这样模型不仅看到用户自己的行为，还能看到“典型用户在同一语境下会怎么行为”，从而自然引入了一种可控的 inductive bias。

基于这一思想，我们提出了如下的上下文工程框架（如下图所示）：

![](https://pic2.zhimg.com/v2-88e92c8c40d7b9eebac854a41e31a5ff_1440w.jpg)

序列格式包含如下几个部分：

-   Interation History（IH）：就是常规理解的用户行为历史。
-   Preference Anchors（PA）：根据工程师的领域知识，构造的锚点序列，辅助引导预测和思考方向。
-   Situational Descriptor（SD）：一些表达场景或其它异构信息的特殊token，一般放在序列末尾聚合信息，例如我们在搜索场用到的user token、query token等。
-   Candidate Item Set（CIS）：潜在目标候选物品的集合，这个是ranking模式下特有的，也是相对于召回模式的优势所在，ranking模式下，候选物品对模型可见，可提供更多上下文信息。

1.  **生成式“思考”：自回归的隐式推理**
    在以 [GPT-o](https://zhida.zhihu.com/search?content_id=268472410&content_type=Article&match_order=1&q=GPT-o&zhida_source=entity) 为代表的大模型体系中，“推理”能力之所以强大，核心在于 test-time scaling law：模型在推理阶段通过更长链路的计算、更深的思考过程产出更高质量的答案。而这种能力的典型载体是显式的语言推理——用文本一步步描述思考过程。
    但对于生成式推荐（GR）来说，这恰恰是一大挑战。
    GR 模型并不在自然语言上工作，而是直接操作用户行为序列与商品 ID 序列。在这种“无自然语言、无描述性逻辑结构”的环境下，模型缺乏承载思考链条的空间，那么它如何像 LLM 一样激发强大的推理能力？
    幸运的是，2024 年起学术界出现了一个新方向：隐式推理（Implicit Reasoning）。
    这一方向试图让模型在不依赖显式文本推理链条的情况下，仍然能够进行复杂的内部思考。那么，什么是隐式推理？
    如果用最直白的方式来描述：就是在模型进行 next-token 计算时，不进行外显的 decoding，也不向外暴露推理链，而是让思考发生在模型内部的隐空间（latent space）中。
    如下图所示（示意图），模型在前向计算阶段“自回归地”展开一段多步推理，但最终只输出一个 token，而不输出推理过程本身。

![](https://pic4.zhimg.com/v2-57b789b701d854135178883d6966ae8d_1440w.jpg)

1.  **利用用户反馈的“渐进性”引导差异化“思考”**
    在系统性比较了显式推理与隐式推理的优缺点之后，我们发现隐式推理面临的最大挑战是：缺乏“过程监督”（process supervision）。
    也就是说，如果模型在内部进行多步推理，但我们并没有告诉它应该“思考什么”或“沿着什么方向思考”，那模型极容易陷入一种“内部次优解”——推理过程中偏离目标，最终输出看似合理但其实 sub-optimal 的结果。
    基于这一认识，我们借鉴了此前在 GNOLR（Embed Progressive Implicit Preference in Unified Space for Deep Collaborative Filtering）中提出的渐进式多任务建模（progressive multi-task learning）思想，并将其进一步融入 OnePiece 的推理建模框架中。
    具体而言，我们在多步推理的过程中，为每一个中间推理阶段叠加不同深度和颗粒度的监督信号，从而让模型的内部推理路径从“简单模式 → 中阶模式 → 高阶模式”逐级展开。这种由浅入深的渐进式监督，使模型不仅能够完成 latent reasoning，还能以更结构化的方式“向着正确的方向思考”。
    如下图所示，这种策略相当于在模型的多跳隐式思维链中插入了一系列“浅监督锚点”，在不暴露具体推理链条的前提下，有效地引导模型持续向更高质量的决策靠拢。

![](https://pica.zhimg.com/v2-23e0ded11c26ef431acdbdcc3aaf9eb0_1440w.jpg)

1.  **实验结果分析**
    4.1 离线实验
    为了深度分析OnePiece的效果，我们进行了详细的对比和消融实验（ablation study）。

![](https://pic4.zhimg.com/v2-7eb994b9dfa95a8039b7ac0965ba895d_1440w.jpg)

如表 2 所示，Shopee 的 DLRM 是一个经过多年工业打磨的强力基线，面对这样成熟的体系，朴素（naive）的生成式推荐模型往往难以真正竞争。
在此基础上，PA 通过上下文工程方式注入额外的 domain knowledge，这一策略本身与模型骨干结构无关，因此无论是 HSTU 还是 ReaRec 都能够从中获益。
进一步来看，OnePiece 相比 ReaRec+PA 能够取得额外提升的核心原因有二：
第一，OnePiece 的 block-wise reasoning 机制显著提升了模型在多步推理中的“信息带宽”；
第二，我们提出的渐进式训练策略有效增强了模型的推理组织性和稳定性，使其能够在更深的 latent reasoning 链条中保持方向一致。

![](https://picx.zhimg.com/v2-73e35bfb2ffbbffd056974d4dcbf7bdd_1440w.jpg)

从表 3 可以观察到，side information 对模型性能具有显著影响。为了充分利用这些丰富的商品属性信号，OnePiece 当前采用的做法是：在序列输入进入 Transformer 之前，通过一个 Linear Adaptor 进行特征融合。
具体而言，我们在每个 token 位置，将 item ID embedding 与其对应的多模态 side info embedding 进行线性聚合，使模型在最底层输入阶段就能够同时接收到 ID-level 与 attribute-level 的信息。这样做不仅提升了表达能力，也为后续的 block-wise reasoning 提供了更高质量的底层语义。

![](https://pic4.zhimg.com/v2-5dde5888df125a398daa819a9009f645_1440w.jpg)

表 4 和表 5 显示，双向注意力在搜推广任务下具有明显优势。这一现象并不难理解：当前主流的搜推系统仍然采用“单请求—单页面返回”的交互模式，GR 模型在工业场景中也并不会依赖自身生成的 token 进行逐步解码。换言之，每一次请求的生成过程本质上都是“一次性完成”的，而不是语言模型那种“逐步生成”的链式过程。
在不存在自回归解码性能压力的前提下，为 pre-filling 部分引入双向注意力能够显著提升上下文聚合能力，使模型在 token 间进行更充分的信息交互，因此自然更适合该场景的建模需求。
同时，实验结果还表明：
多步推理具有 scaling 效果——随着推理步数增加，模型性能会逐步提升，但提升幅度逐渐收敛；
渐进式引导优于仅监督最终一步——通过对中间推理阶段施加层级化监督信号，模型的内部推理组织性显著增强，最终输出也更稳定。
4.2在线实验
我们在Shopee主搜场景进行了实验，具体地，我们在召回阶段和prerank阶段两个正交的实验层进行了在线AB实验。在召回阶段，我们将OnePiece召回替代了原有的[DeepU2I](https://zhida.zhihu.com/search?content_id=268472410&content_type=Article&match_order=1&q=DeepU2I&zhida_source=entity)召回，取得了1.08%的GMV/user增长；在prerank阶段，我们用OnePiece ranking model替换了原有的DLRM model，取得了1.12%的GMV/user增长和2.9%的广告收入增长，可以说是相当大幅度的提升。

1.  **未来展望：构建更加通用的工业级搜推模型**

![](https://picx.zhimg.com/v2-c46cf42f02890a4a7a30867ab326de3b_1440w.jpg)

OnePiece 可以被视为我们在 “One For All” 通用“推理型”推荐模型方向上的一次初步探索。它的实验结果证明：通过适配推荐场景的上下文工程与推理范式，我们确实能够有效地引导模型的预测路径，使其在无语言环境下展现出类似 LLM 的指令跟随能力。

OnePiece 1.0 已经让我们看到，一个全新的技术方向正在出现：
“提示词优化（Prompt Optimization）” 正在成为搜索推荐技术栈中的潜在核心能力。

这意味着未来的推荐系统，不再只是靠特征工程与模型结构优化驱动，而是可以像 LLM 一样，通过调整上下文与思考模式来显著改变模型行为。

基于此，我们的下一步工作将聚焦于打造OnePiece 2.0版本， 尝试以单一模型，统一建模多场景、多类型的召回策略，构建更加通用的工业级搜推模型：General Recommender Model。

对Onepiece或者其他大厂的生成式算法感兴趣的，可以了解我在今年刚推出的课程《生成式推荐系统算法与实践》。在里面，我系统地对比了不同技术路线的优劣，深入拆解大厂已落地案例的真实架构设计与踩坑经验→[http://gk.link/a/12GEz](https://link.zhihu.com/?target=http%3A//gk.link/a/12GEz)

论文原文链接：

arxiv地址：[https://arxiv.org/pdf/2509.18091](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2509.18091)

hugging face地址：[https://huggingface.co/papers/2509.18091](https://link.zhihu.com/?target=https%3A//huggingface.co/papers/2509.18091)

《生成式推荐系统算法与实践》链接：[http://gk.link/a/12GEz](https://link.zhihu.com/?target=http%3A//gk.link/a/12GEz)