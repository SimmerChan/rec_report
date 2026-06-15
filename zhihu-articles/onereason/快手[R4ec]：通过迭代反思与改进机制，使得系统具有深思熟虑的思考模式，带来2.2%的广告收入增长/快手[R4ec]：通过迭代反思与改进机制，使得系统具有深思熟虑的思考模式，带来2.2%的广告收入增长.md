# 快手[R4ec]：通过迭代反思与改进机制，使得系统具有深思熟虑的思考模式，带来2.2%的广告收入增长
> 发布时间: 编辑于 2026-04-23 01:44・北京
> 原文链接: https://zhuanlan.zhihu.com/p/1938044297418109393

---

​

目录

```text
标题：R4ec: A Reasoning, Reflection, and Refinement Framework for Recommendation Systems
链接：https://arxiv.org/pdf/2507.17249
机构：快手，中国科学院大学
收录：RecSys‘2025
```

## 摘要

本文借助大型语言模型（[LLMs](https://zhida.zhihu.com/search?content_id=261521713&content_type=Article&match_order=1&q=LLMs&zhida_source=entity)）来提升推荐效果，提出R4ec框架。其包含两个模型：行动者模型（负责推理和根据反馈改进结果）和反思模型（判断行动者模型输出是否合理并给出反馈），两者通过迭代的反思与改进机制，不断完善用户偏好和物品事实知识，最终将这些知识精炼并融入推荐骨干模型。实验表明，在 Amazon-Book、[MovieLens-1M](https://zhida.zhihu.com/search?content_id=261521713&content_type=Article&match_order=1&q=MovieLens-1M&zhida_source=entity) 两个公开数据集上，R4ec相比其他方法，在 [AUC](https://zhida.zhihu.com/search?content_id=261521713&content_type=Article&match_order=1&q=AUC&zhida_source=entity) 和 LogLoss 指标上表现更优。在线A/B实验中，该方法**带来了2.2%的收入增长，其对长尾数据效果更明显**。

## 背景介绍

目前 LLM 增强的推荐系统，核心在于利用大型语言模型丰富的开放世界知识和出色的推理能力为推荐系统赋能。 具体有两阶段尝试：

-   早期尝试：用 “上下文学习” 让 LLM 对传统模型筛选出的候选物料重新排序，但效果一般。
-   近期尝试：用 “思维链提示” 让 LLM 把推荐任务拆成一系列步骤，并逐步生成关于用户偏好知识以及与物品相关的事实性知识，再把这些知识作为额外特征给下游推荐模型使用。

## 现有问题

1.  **推理出错率高：**现有利用 LLM 的推荐方法，多类似 “快速直觉式” 的 System-1 思维（类似人类的直觉快速判断），推理过程易出错（推理过程中只要出一点错，结果就可能完全跑偏）；
2.  **成本高和效率低：**很多方法需要调用像 [GPT-3.5](https://zhida.zhihu.com/search?content_id=261521713&content_type=Article&match_order=1&q=GPT-3.5&zhida_source=entity) 这类模型的 API，成本高、延迟大，导致无法实际应用。

## 核心思想

运用推理、反思与改进这三项核心能力，为推荐系统构建 “System-2 思维”，即：不凭直觉下结论，而是通过反复检查和修正，让结论更可靠。

整个过程主要由行动者模型（actor model, \\pi\_\\theta ）和反思模型（reflection model, \\pi\_\\psi ）两个 LLM 组成：行动者模型先 “思考”，反思模型 “挑错”，行动者模型再 “改”，反复迭代。

![](https://pic3.zhimg.com/v2-73c1bfb878bc7552fdd50194168174c2_1440w.jpg)

迭代的反思和改进机制：行动者模型先推理，反思模型反思挑错，行动者模型再改进，反复迭代

## 技术原理

### 构建用户偏好数据集

在用户偏好推理任务中，**给定用户的交互历史及其对应的评分，模型需要总结用户偏好**。为构建用于用户偏好的**推理、反思与改进数据集**，使用常见的推荐数据 \\{ hist \_{k}, item \_{k}, label \_{k}\\}\_{k=1}^{|D|} ，其中 hist \_{k} 表示用户 u \_{k} 的交互历史、 item \_{k} 表示目标物品，而 label \_{k} 则表示用户 u\_k 是否会喜欢该目标物品 item\_k 。

![](https://pic4.zhimg.com/v2-6de3979cffdf7fa8af1dc038bd35a15f_1440w.jpg)

**第一步推理：**采用 “预测时推理” 范式（reasoning while predicting paradigm），该范式借助大语言模型（LLMs）的推理能力来生成预测结果及相应依据（即知识）。该过程可表示为： \\\[<u\_{pre } , pred >\\leftarrow \\mathcal{M}\\left(\\mathcal{P}\_{reason }^{user }( hist, item )\\right)\\\] ，其中， \\mathcal{M} 表示大语言模型， \\mathcal{P}\_{reason }^{user } 表示用户偏好推理构建提示词。

![](https://pic3.zhimg.com/v2-43a6e0da9d58ca2d3c96c93876025df0_1440w.jpg)

用户偏好推理构建提示词：输入用户历史行为和目标物料，输出用户偏好和判断是否感兴趣

**第二步反思：**采用反思机制：让该模型判断用户偏好知识是否合理，对于不合理的用户偏好知识，模型会生成反思内容，这些内容不仅会指出响应中存在的缺陷，还会提供具有参考价值的修正建议。该过程可表示为： < judge^{u} , reflect^{u}>\\leftarrow \\mathcal{M} \\left(\\mathcal{P}^{user}(hist, item, u\_{pre})\\right) 。

![](https://pic1.zhimg.com/v2-b003f2729feab44714f804bd21dee9ce_1440w.jpg)

用户偏好反思构建提示词：输入用户历史行为、目标物料和用户偏好，输出判断偏好是否合理，不合理需给出反馈

**第三步改进：**对比推理结果和用户的真实反馈（真值），对于推理错误同时反思认为不合理的样本，进行改进。引入用户偏好优化构建提示词（记为 \\(\\mathcal{P}\_{refine }^{user }\\) ），随后让模型生成新的预测结果以及改进后的用户偏好。该过程可表示为： <u\_{pre }^{r}, pred '>\\leftarrow \\mathcal{M}\\left(\\mathcal{P}\_{refine }^{user }\\left( hist, item, u\_{pre }, reflect ^{u}\\right)\\right) 。

![](https://pic3.zhimg.com/v2-c6f4e392a4796f270f55557dab8e36f6_1440w.jpg)

用户偏好改进构建提示词：输入用户历史行为、目标物料、用户偏好和反馈，输出优化后的用户偏好和判断是否感兴趣

最终，根据推理结果是否与真值一致以及是否反思合理，将数据加入对应的数据集：

|  | 推理结果正确 | 推理结果错误 |
| --- | --- | --- |
| 反思认为合理 | 加入推理和反思数据集 | 丢弃 |
| 反思认为不合理 | 丢弃 | 用于改进，改进后预测结果正确的加入反思和改进数据集，错误则丢弃 |

三个数据集中的样本形式如下： s\_{reason}^u = \\left\\{ (hist), (u\_{pre}) \\right\\} \\\\ s\_{reflect}^u = \\left\\{ (hist, u\_{pre}), (judge^u, reflect^u) \\right\\} \\\\ s\_{refine}^u = \\left\\{ (hist, u\_{pre},reflect^u), (u\_{pre}^r) \\right\\}

### 构建物料事实数据集

与用户偏好数据集类似，物料事实数据集的核心是让模型学会 “推理物料知识→检查推理是否合理→根据问题改进知识” 的能力。物料事实知识的合理性无法直接通过 “物料本身属性” 判断，而是间接依赖 “用户是否喜欢该物料”。因此，使用的原始数据集结构如下： \\(\\{ item \_{k}, pos \_{k}, neg \_{k}, tar \_{k}, label \_{k}\\}\_{k=1}^{|D'|}\\) ，其中 item \_{k} 表示目标物料  i\_{k} 的信息； \\(pos \_{k}\\) 和 \\(neg \_{k}\\) 分别代表喜欢和不喜欢物料 \\(i\_{k}\\) 的用户的交互历史，而 \\(tar \_{k}\\) 表示目标用户的交互历史； \\(label \_{k}\\) 用于指示目标用户是否会喜欢该目标物料 \\(i\_{k}\\) 。

采用与构建用户偏好数据集相似的过程构建物料事实数据集。推理、反思和改进三个数据集的样本形式如下： s\_{reason }^{i}=\\left\\{( item, pos, neg ),\\left(i\_{fact }\\right)\\right\\} \\\\ s\_{reflect }^{i}=\\left\\{\\left( item, pos, neg, i\_{fact }\\right),\\left( judge ^{i}, reflect ^{i}\\right)\\right\\} \\\\ s\_{refine}^{i}=\\left\\{( item, pos, neg, i\_{fact }, reflect ^{i}\\right),\\left(i\_{fact }^{r}\\right)\\}

### 训练行动者和反思模型

使用 **LoRA 技术微调**两个模型，节省训练成本：

-   使用推理和改进数据集同时训练行动者模型：  \\mathcal{L}\_{reason }=\\mathbb{E}\_{(x, y) \\sim \\mathcal{D}\_{reason }}\\left\[\\log \\pi\_{\\theta}(y | x)\\right\] \\\\ \\mathcal{L}\_{refine }=\\mathbb{E}\_{\\left(x', y'\\right) \\sim \\mathcal{D}\_{refine }}\\left\[\\log \\pi\_{\\theta}\\left(y' | x'\\right)\\right\]\\\\ \\mathcal{L}\_{actor }=\\mathcal{L}\_{reason }+\\mathcal{L}\_{refine } \\quad
-   使用反思数据集训练反思模型： \\mathcal{L}\_{reflect }=\\mathbb{E}\_{\\left(x^{\*}, y^{\*}\\right) \\sim \\mathcal{D}\_{reflect }}\\left\[log \\pi\_{\\psi}\\left(y^{\*} | x^{\*}\\right)\\right\]

### 推理策略

两种生成知识的推理策略：

**迭代改进**（Iterative Refinement）：这是一种模拟人类 “反复思考改进” 的推理方式，核心是多轮迭代提升知识可靠性，具体流程如下：

-   第一步：行动者模型生成初始知识；
-   第二步：反思模型评估该知识是否合理，若不合理，生成反馈；
-   第三步：行动者模型根据反馈改进知识
-   循环步骤：重复”评估→改进”的过程，直到反思模型认为知识合理，或达到预设的最大迭代次数（默认 1 次）

这种策略的核心价值是将知识获取过程转化为 “深思熟虑”（System-2 思维），通过多次修正减少推理错误。

**反思过滤策略**（Reflection as a Filter）: 这是一种基于 “多轮生成 + 筛选” 的推理方式，核心是通过过滤剔除不合理知识，具体流程如下：

-   第一步：行动者模型生成多个版本的初始知识；
-   第二步：反思模型逐一评估这些知识，过滤掉不合理的知识；
-   第三步：对剩余合理的知识进行平均，作为最终知识。若所有知识都被判定为不合理，则对所有初始知识取平均。

这种策略的核心价值是通过 “多生成 + 严筛选” 提升知识的稳健性，借鉴了 “自我一致性”（self-consistency）技术的思路。

![](https://pic2.zhimg.com/v2-05c84fd201fffd0101e2f4a17aa72dcf_1440w.jpg)

不同推理策略在不同骨干网络（AutoInt和GRU4Rec）上的效果：都有效果，迭代改进效果更好，因为引入了改进

![](https://picx.zhimg.com/v2-24431132581d8f2a82a5db74ff74deb5_1440w.jpg)

不同最大迭代次数对效果的影响：次数越多，效果越好，但边际效益递减，可能是反思模型能力不足

### 知识应用

为了在现有的推荐模型中使用用户偏好知识和物料事实知识，需要将文本形式的知识转化为稠密向量，使用知识编码器（如BGE-M3）：

e^{u}=\\varepsilon n c o d e r\\left(u\_{p r e}\\right), e^{i}=\\varepsilon n c o d e r\\left(i\_{f a c t}\\right)

![](https://pica.zhimg.com/v2-cdf3ac539754cc0ef30638d2e8120ec0_1440w.jpg)

不同知识编码器在公开数据集上的效果：都有效果，BGE-M3效果最好

将生成的稠密向量通过MLP后，与其他稀疏特征一起，**作为推荐骨干模型的输入**。

## 实验效果

### 离线实验

-   数据集

-   Amazon-Book：评分＞5 为 “喜欢”
-   MovieLens-1M：评分＞3 为 “喜欢”
-   生产数据集：以用户点击为 “喜欢”

-   选用的骨干模型：[DIEN](https://zhida.zhihu.com/search?content_id=261521713&content_type=Article&match_order=1&q=DIEN&zhida_source=entity)、GRU4Rec、AutoInt、FiGNN、DCN、[DeepFM](https://zhida.zhihu.com/search?content_id=261521713&content_type=Article&match_order=1&q=DeepFM&zhida_source=entity)
-   Metrics：AUC、LogLoss
-   对比基线

-   Base：没有来自 LLM 的知识
-   KAR：用 GPT-3.5 生成用户和物料知识
-   R2ec：只用推理数据集训练行动者模型，无反思和改进（ System-1 思维）

![](https://pic2.zhimg.com/v2-8699b8b169f7cde0972dbcdbff9b8f03_1440w.jpg)

不同骨干模型下，R4ec/KAR/R2ec对比：1）都优于基线，说明LLM知识有效；2）R4ec优于KAR，KAR直接使用LLM存在幻觉等问题，微调更可靠；3）R4ec优于R2ec，说明System-2思维优于System-1

![](https://pic2.zhimg.com/v2-6b27b807dbc7f888fe66963ef3139c19_1440w.jpg)

行动者模型的Scaling Law：行动者模型规模越大，效果越好，反思模型带来的提升越小。（w表示有反思模型，w/o表示没有反思模型）

![](https://picx.zhimg.com/v2-e9ee3ed2d23f98e463898a09ab871eeb_1440w.jpg)

反思模型的Scaling Law：反思模型规模越大，效果越好

### 在线实验

在线A/B实验，基线方法20%流量，R4ec10%流量，14天在线实验，收入提升2.2%, 对长尾场景效果更明显。

![](https://picx.zhimg.com/v2-4446887fa8ce4b2b61eaf76f3dd09a61_1440w.jpg)

在线A/B实验效果：收入提升2.2%

在线部署：

-   从400百万用户中选取1百万用户，构建推理、反思、改进数据集。
-   为节省成本，**仅对活跃用户计算偏好知识**，对低活用户用历史行为的物料知识代替用户偏好知识。

## 总结

基于多智能体反复思考和改进，生成高效稳定的用户和物料知识，使得轻量级LLM即可超越直接调用复杂LLM的效果，解决直接将LLM用于推荐场景的模型幻觉、训推成本高等问题，可用于推荐系统的各个阶段。

## 参考资料

-   [RecSys'25|快手R4ec：用 “反思” 让推荐更懂你](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/zYkZE8fqARBfywWFeJwiNQ)