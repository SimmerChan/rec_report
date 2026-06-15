# (开源)细读快手OneRec-Think技术分享
> 发布时间: 编辑于 2025-10-28 08:33・广东
> 原文链接: https://zhuanlan.zhihu.com/p/1963536264767738349

---

​

目录

大家好, 我是州懂, 今天分享OneRec-Think的技术Paper, 是落地[LLM-as-RS](https://zhida.zhihu.com/search?content_id=264569932&content_type=Article&match_order=1&q=LLM-as-RS&zhida_source=entity)范式的不错实践。

> 标题: OneRec-Think: In-Text Reasoning for Generative Recommendation
> 地址: [https://arxiv.org/pdf/2510.11639](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2510.11639)
> 代码: [https://github.com/wangshy31/OneRec-Think](https://link.zhihu.com/?target=https%3A//github.com/wangshy31/OneRec-Think)
> 公司: 快手

## 1\. 前言

周国睿老师在Aicon上分享中有提到OneRec-Think, 不少同学在问快手是否有公开更详细的资料, 这不现在文章终于释放出来了, 抓紧学习。

![](https://pic4.zhimg.com/v2-32d7b238a97c0a94d30efa809046114f_1440w.jpg)

话说LLM最让推荐系统眼馋的两个能力:

-   **Scaling Law:** 推荐系统对响应时延有更高的要求, 如何在成本和时延的双约束下实现推荐系统的Scaling Law是业界在苦苦探索&追寻的方向
-   **[强推理能力](https://zhida.zhihu.com/search?content_id=264569932&content_type=Article&match_order=1&q=%E5%BC%BA%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B&zhida_source=entity):** 逐步推理(Step-by-step Reasoning)能力是LLM最具有代表性的涌现能力之一, LLM利用思维链([Chain-of-Thought](https://zhida.zhihu.com/search?content_id=264569932&content_type=Article&match_order=1&q=Chain-of-Thought&zhida_source=entity),CoT)来加强推理效果着实令人羡慕。

现阶段生成式推荐(如之前OneRec)更多的是在探索Scaling Law, 基于用户行为序列S\_u=(\\mathbf{s}\_{v\_1}, \\dots, \\mathbf{s}\_{v\_{n}})去做next itemic tokens prediction:

\\mathbf{s}\_{v\_{n+1}} \\sim P(\\cdot|\\mathbf{s}\_{v\_1}, \\dots, \\mathbf{s}\_{v\_{n}};\\theta) \\\\

这里, \\mathbf{s}\_v = (s\_v^1, \\dots, s\_v^L)表示语义ID, 快手这里语义ID同OneRec, 也是使用了3 \\times 8192, 共24576个token, 下文会提到, 这些token在OneRec-Think会作为LLM额外的token去做训练微调。

快手提出的OneRec-Think是希望能吃到"LLM强推理能力"的技术红利, 尝试将大模型的显式文本推理能力引入到推荐系统。具体地:

-   首先, 先得到推理序列\\mathbf{\\tau}=(r\_1, \\dots, r\_M), 也就是生成推理原因

 \\mathbf{\\tau} \\sim P\\left(\\cdot \\mid \\mathcal{P}(\\mathbf{s}\_{v\_1}, \\dots, \\mathbf{s}\_{v\_n}); \\theta\\right) \\\\

-   然后, 再将推理序列\\mathbf{\\tau}作为额外输入, 去做语义ID预测

\\mathbf{s}\_{v\_{n+1}} \\sim P\\left(\\cdot \\mid \\mathcal{P}(\\mathbf{s}\_{v\_1}, \\dots, \\mathbf{s}\_{v\_n}), \\mathbf{\\tau}; \\theta\\right) \\\\

其中, \\mathcal{P}(\\cdot)表示提示Prompt(论文给出了很详细的Prompt), 比如下图就是OneRec-Think的一个示例:

![](https://pic1.zhimg.com/v2-3c4ec313fc61f7232ec29e20876223c8_1440w.jpg)

快手在主场景落地了OneRec-Think这种LLM-as-RS范式, 并开源了代码。

## 2\. 方法

OneRec-Think的整体框架如下图所示:

![](https://pic1.zhimg.com/v2-c5f4b11aa366eea193a776c0722a7f9a_1440w.jpg)

主要包含3个模块:

![](https://pic2.zhimg.com/v2-7989818dc8330f264ee91b5bf013ad6b_1440w.jpg)

-   **Itemic Alignment:** 将语义token作为LLM额外的词表去做预训练, 让Item语义token对齐LLM的文本token embedding空间
-   **Reasoning Activation:** 在推荐上下文中激活模型的推理能力
-   **Reasoning Enhancement:** 利用特定于推荐的奖励函数来捕获用户偏好的多样性(multi-validity)

### 2.1 多任务预训练实现Item语义对齐

和OneRec一样, 每个Item使用了3 \\times 8192个语义ID来量化, 例如<|item\_begin|>< item\_a\_1123><item\_b\_5813><item\_c\_4212><| item\_end|>, 共有24,576个语义Token。

论文使用[Qwen-8B](https://zhida.zhihu.com/search?content_id=264569932&content_type=Article&match_order=1&q=Qwen-8B&zhida_source=entity)作为Backbone model, 训练的词表扩充了上面这些语义Token。同时, 为了让这些语义token对齐原有的LLM text token空间, 作者基于推荐系统的用户行为, 从多个维度构造了可用于增量训练的数据(约20B/天)。

### 2.1.1 构造预训练数据语料

![](https://pic3.zhimg.com/v2-ef044c28362223c614551b62fcda3fe8_1440w.jpg)

### 1) 用户画像数据(Interleaved User Persona Grounding)

包含了基础用户画像(如年龄)描述、近期行为(如搜索、点赞、评论、关注)描述、用户主次兴趣刻画。

### 2) 用户序列偏好数据(Sequential Preference Modeling）

基于用户行为序列构造偏好数据, 用于预测用户最感兴趣的Next Item

### 3) Item内容描述数据(Itemic Dense Captioning)

这里是为了让LLM能学习&解码到Item的语义Token \\rightarrow 所表示具体文本描述性内容。

### 4) 通用语义建模数据(General Language Modeling)

这里是一些通用的文本语料库, 增加该训练数据是为了保障LLM在持续预训练过程中不会因为上面的其它任务而丧失原有LLM的世界知识。

### 2.1.2 持续预训练

基于上面的语料数据继续预训练模型, 包含两个阶段

### 1) Token Warm-up阶段

-   **目标:** 主要训练新引入的itemic token embed，确保这些语义token能对齐原有LLM text token embed空间
-   **操作:** 使用6B的用户画像数据, 训练时冻结原LLM的参数，只训练新引入的itemic token的Embed。
-   **学习率:** 使用较高的学习率（如 5\\times10^{-4}，以便新的token embed能快速收敛。

### 2) Multi-Task Integration 阶段

-   **目标:** 将上面的4个数据任务结合起来，共同训练模型。
-   **数据混合:** 将不同任务的数据按照一定的比例混合，确保模型在训练过程中能够同时学习到语言生成能力和推荐任务所需的能力。

![](https://picx.zhimg.com/v2-5bc1da459e77787abe38677cf2faf29b_1440w.jpg)

-   **学习率:** 基于[LoRA](https://zhida.zhihu.com/search?content_id=264569932&content_type=Article&match_order=1&q=LoRA&zhida_source=entity), 使用较低的学习率(如3\\times10^{-4}), 以确保模型在微调过程中能够稳定地适应新的任务。

### 2.1.3 语义对齐的效果验证

为了验证上面Item语义对齐后的效果, 看看通过Item的itemic token，模型是否能生成准确的自然语言描述。

### 1) 公开数据集上

作者让模型基于语义Token来生成对应的描述, 并与其实的描述做对比(计算BertScore)。下图是Beauty数据集中的的两个示例:

![](https://pic4.zhimg.com/v2-370872a3de00d61cf1cb17ed4a587cf9_1440w.jpg)

### 2) 业务数据集上

在短视频推荐场景, 作者分别验证LLM对Item侧和User侧的理解能力, 同样的, 也是基于结果来计算BertScore。

![](https://pic3.zhimg.com/v2-41c070e02f6ace56a23e6132bf3bc618_1440w.jpg)

作者消融验证了TW(Token Warm-up)模块和MI(Multi-Task Integration)模块的效果

![](https://picx.zhimg.com/v2-3e2771ab80d3c4bfd845aaa4885da885_1440w.jpg)

### 2.2 推理激活

为了更好的激活大语言模型（LLM）在推荐系统中的显式推理能力, 这一阶段通过监督学习的方式，引导模型生成高质量的推理路径，从而提升推荐的准确性和可解释性。这个过程包含了两个步骤(从简单\\rightarrow复杂):

![](https://pica.zhimg.com/v2-0dcbd1ba8a0ce7c77e2de5eba7a7d7c4_1440w.jpg)

### 1）简单任务: Bootstrapping with Pruned Contexts

首先, 针对target item \\mathbf{s}\_{v\_{n+1}}, 从用户行为序列(\\mathbf{s}\_{v\_1},\\dots,\\mathbf{s}\_{v\_{n}})中检索出Top-k(k=10)个Item(类似SIM)

g((\\mathbf{s}\_{v\_1},\\dots,\\mathbf{s}\_{v\_{n}}), \\mathbf{s}\_{v\_{n+1}}) = (\\mathbf{s}\_{w\_1},\\dots,\\mathbf{s}\_{w\_k}) \\\\

其中, g(\\cdot,\\cdot)表示基于item embed计算cosine相似度。然后, 再基于这top-k个Item, 生成target item的推理序列\\mathbf{\\tau}

\\mathbf{\\tau} \\sim P\\left(\\cdot \\mid \\mathcal{P}\_{r}((\\mathbf{s}\_{w\_1},\\dots,\\mathbf{s}\_{w\_k}), \\mathbf{s}\_{v\_{n+1}}); \\theta\\right) \\\\

其中, \\mathcal{P}\_{r}表示对应的提示(示例见下图)

### 2) 复杂任务: Learning to Reason from Noisy Sequences

这里是基于完整的用户行为序列数据来激活推理, 包含了两部分损失:

\\begin{aligned} \\mathcal{L}\_{\\text{RA}} = - \\Biggl(& \\sum\_{i=1}^{M} \\log P(r\_i|\\mathcal{P}(\\mathbf{s}\_{v\_1}, \\dots, \\mathbf{s}\_{v\_n}), r\_{\\text{<}i};\\theta) \\\\ + &\\sum\_{j=1}^{L} \\log P(s\_{v\_{n+1}}^j | \\mathcal{P}(\\mathbf{s}\_{v\_1}, \\dots, \\mathbf{s}\_{v\_{n}}), \\mathbf{\\tau}, {s}\_{v\_{n+1}}^{\\text{<}j};\\theta) \\Biggr) \\end{aligned} \\\\

第1部分表示推理序列\\mathbf{\\tau}=\\{r\_1,\\dots,r\_M\\}所对应的损失, 第2部分表示Target Item的L个语义token所对应的损失。

上面两个步骤使用LoRA微调推理过程, 所对应的Prompt示例如下:

![](https://pica.zhimg.com/v2-b841742d57f5915182a0e974a7267396_1440w.jpg)

### 2.3 推理增强

推理增强是通过设计特定的奖励机制, 使用强化学习来进一步优化模型的推理能力和推荐准确性。

![](https://pic4.zhimg.com/v2-1cbc4e4dc1a0074d38ce560763f73eab_1440w.jpg)

在推荐场景中，标准的可验证奖励（如精确匹配）面临稀疏性挑战，因为大多数推理路径无法精确命中Target Item，导致奖励信号过于稀疏，无法有效指导模型训练。为此, 作者基于beam search结果定义了一种新的奖励机制，称为[Rollout-Beam奖励](https://zhida.zhihu.com/search?content_id=264569932&content_type=Article&match_order=1&q=Rollout-Beam%E5%A5%96%E5%8A%B1&zhida_source=entity)。

下面的公式看着略微有些复杂, 但实际上就是"**取Beam Search结果中的最大Reward**"

\\mathcal{R}\_{\\text{Rollout-Beam}} = \\max\_{\\hat{s}\_{v\_{n+1}} \\in \\mathcal{B} }\\sum\_{l=1}^{L} \\mathbb{I}(\\hat{s}\_{v\_{n+1}}^l = s\_{v\_{n+1}}^l) \\\\\\small \\mathcal{B} = \\bigl\\{ \\bigl( \\hat{s}\_{v\_{n+1}}^{1,(j)},\\cdots, \\hat{s}\_{v\_{n+1}}^{L,(j)} \\bigr) \_{j=1}^K\\bigr\\} = \\text{BeamSearch}\\Bigl(P(\\mathbf{s}\_{v\_{n+1}} \\mid \\mathbf{H}, \\mathbf{\\tau};\\theta\\bigr), K\\Bigr) \\\\

其中, \\mathbf{H} = \\mathcal{P}(\\mathbf{s}\_{v\_1}, \\dots, \\mathbf{s}\_{v\_n}) 表示基于用户行为序列构造的Prompt(参考2.1.1的示例)

最后, 基于上述Reward, 使用GRPO算法优化模型, 具体设置如下:

-   学习率1e-5
-   beam search: K=32
-   CoT paths: |G|=16
-   KL divergence系数: \\beta=0.001
-   裁剪比率: \\epsilon=0.2

### 2.4 线上部署("Think-Ahead"架构)

为了在严格的时延约束下部署OneRec-Think, 快手提出了离在线的两阶段"Think-Ahead"架构, 离线阶段提前预估出语义ID的前两位, 在线阶段使用线上的OneRec模型预估语义ID的最后一位。

### 2.4.1 离线推理阶段（Offline Reasoning Stage）

### 1）推理路径生成（Reasoning Path Synthesis）

-   **目标:** 为每个用户生成多个推理路径，这些路径代表了用户行为与潜在兴趣之间的逻辑联系。
-   **方法:**

-   对于用户u的历史行为序列H\_u=\\mathcal{P}(\\mathbf{s}\_{v\_1}, \\dots, \\mathbf{s}\_{v\_n}), 采样T个多样化的推理路径 \\mathbf{\\tau}^{(i)} \\sim P(\\cdot \\mid H\_u; \\theta)
-   每个推理路径\\tau^{(i)}代表了一种可能的逻辑推理方式，连接用户的过去行为与潜在兴趣。

### 2) 约束前缀生成（Constrained Prefix Generation）

-   **目标:** 基于每个推理路径，生成初始的物品前缀(3层语义ID的前两个(\\hat{s}\_{v\_{n+1}}^1,\\hat{s}\_{v\_{n+1}}^2)），这些前缀可以理解为用户可能感兴趣的物品类别。
-   **方法:**

-   对于每个推理路径\\tau^{(i)}, 使用beam search生成候选前缀: \\mathcal{A}\_u^{(i)} \\!= \\text{BeamSearch}\\Bigl(\\!P\\bigl(\\hat{s}\_{v\_{n+1}}^1,\\hat{s}\_{v\_{n+1}}^2 \\mid H\_u,\\mathbf{\\tau}^{(i)};\\theta\\bigr),m\\Bigr)
-   这里，\\mathcal{A}\_u^{(i)}表示从推理路径\\tau^{(i)}生成的m个候选前缀。

### 3) 缓存语义空间(Semantic Space Materialization)

-   **目标:** 将所有生成的前缀合并，形成用户的个性化候选空间。
-   **方法:**

-   将所有推理路径生成的前缀集合并  \\mathcal{C}\_u = \\bigcup\_{i=1}^T \\mathcal{A}\_u^{(i)}
-   这个集合\\mathcal{C}\_u包含了T\\times m个高潜力的物品前缀，会被缓存到工业分布式存储系统中，为后续的实时推理提供支持。

### 2.4.2 在线推理阶段（Online Inference Stage）

基于前面预计算缓存下来的用户语义空间前缀, 继续做约束解码(Constrained Decoding), 具体地:

-   当用户u发起请求时，从分布式存储系统中检索\\mathcal{C}\_u
-   使用实时更新的OneRec模型h\_{online}进行最终的解码：

\\begin{aligned} \\hat{\\mathbf{s}}\_{v\_{n+1}} = \\underset{\\mathbf{s}\_{v\_{n+1}}}{\\arg\\max} \\ & P\_{h\_{\\mathrm{online}}}\\bigl(\\mathbf{s}\_{v\_{n+1}} \\bigm| \\mathbf{s}\_{v\_1},\\dots, \\mathbf{s}\_{v\_n}\\bigr) \\\\ & \\text{s.t.} \\quad (\\hat{s}\_{v\_{n+1}}^1,\\hat{s}\_{v\_{n+1}}^2) \\in \\mathcal{C}\_u \\end{aligned} \\\\

## 3\. 实验部分

### 3.1 整体效果

在多个公开数据集上在Recall@K和NDCG@K指标上全面超过对比模型

![](https://pica.zhimg.com/v2-9e822fc10048d5e5d0c087bebc58ebb8_1440w.jpg)

### 3.2 消融实验

在Beauty数据集上消融验证IA(指Itemic Alignment)模块和R(指enhanced reasoning mechanism)模块的效果

![](https://pic2.zhimg.com/v2-fcbe93ba453a3cdb768dda4eb17bfa5b_1440w.jpg)

### 3.3 线上AB实验

用了1.29%的线上流量做实验, App停留时间显著提升0.159%

![](https://picx.zhimg.com/v2-d33ba3b47c348ada3bfff8c700271f27_1440w.jpg)

### 3.4 Case Study

-   对话式推荐场景, 能动态适应用户需求

![](https://pic2.zhimg.com/v2-c30c3da8625251ebf4790310d623f8f5_1440w.jpg)

-   短视频推荐场景, 细粒度兴趣推理过程

![](https://pic2.zhimg.com/v2-38274d04b72daa9c620d6a98a8c31eb5_1440w.jpg)

-   推理过程能保持语义一致性

![](https://pic4.zhimg.com/v2-04d637fce31ee67510325fd774aa77e1_1440w.jpg)

-   一些其它示例

![](https://pica.zhimg.com/v2-8554c82384f55ae4da4aecf7d34ad188_1440w.jpg)