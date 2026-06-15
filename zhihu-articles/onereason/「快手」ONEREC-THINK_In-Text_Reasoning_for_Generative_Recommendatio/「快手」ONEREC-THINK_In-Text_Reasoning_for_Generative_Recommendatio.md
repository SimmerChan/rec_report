# 「快手」ONEREC-THINK: In-Text Reasoning for Generative Recommendatio
> 发布时间: 编辑于 2025-10-16 20:31・四川
> 原文链接: https://zhuanlan.zhihu.com/p/1961917655096201392

---

![](https://pica.zhimg.com/v2-3b24601899e25a890dc5a77bc40fed76_1440w.jpg)

[ONEREC-THINK: In-Text Reasoning for Generative Recommendation](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2510.11639)

又一篇[LLM-as-RS](https://zhida.zhihu.com/search?content_id=264273104&content_type=Article&match_order=1&q=LLM-as-RS&zhida_source=entity)（onerec系列），之前介绍过谷歌PLUM做对齐SID+微调llm的工作，这篇文章增加了**[显式推理](https://zhida.zhihu.com/search?content_id=264273104&content_type=Article&match_order=1&q=%E6%98%BE%E5%BC%8F%E6%8E%A8%E7%90%86&zhida_source=entity)**。那么显式推理的label怎么来？经典的左手倒右手，用一个llm生成另一个llm的样本。

本文在附录中详细给出了每个步骤的prompt demo，值得一读。

## 1 背景

传统推荐系统中缺乏可解释和可控推理能力的问题，而近期的生成式推荐模型（如[OneRec](https://zhida.zhihu.com/search?content_id=264273104&content_type=Article&match_order=1&q=OneRec&zhida_source=entity)）主要作为隐式预测器，缺乏现代LLMs的关键优势——显式推理能力。本文旨在研究如何将LLMs的推理能力与推荐系统无缝集成。

之前的生成式，利用n个历史行为生成第n+1个：

![](https://pic2.zhimg.com/v2-f0f774a9bf3d276c8d5ba23ad7a33de7_1440w.jpg)

本文引入推理序列$\\tau$，先生成推理原因，再生成第n+1个：

![](https://pica.zhimg.com/v2-83b36364eb5af124d9ee3dce902b796a_1440w.jpg)

所以核心在于推理序列$\\tau$如何生成。

## 2 方法

![](https://pica.zhimg.com/v2-d61b883fcd7752d22e5a62ea535ea1f6_1440w.jpg)

## 2.1 [Itemic Alignment](https://zhida.zhihu.com/search?content_id=264273104&content_type=Article&match_order=1&q=Itemic+Alignment&zhida_source=entity)（项目对齐）:

通过跨模态的项目-文本对齐实现语义对齐。具体任务包括：

-   交替用户画像对齐（Interleaved User Persona Grounding）：将项目标记和文本标记交替排列，创建丰富的双模态训练实例。

![](https://pic3.zhimg.com/v2-5a1cfc760ee8d74afcee7bf08885c9a2_1440w.jpg)

-   顺序偏好建模（Sequential Preference Modeling）：从用户历史时间序列中预测后续项目交互。

![](https://pic1.zhimg.com/v2-0951529f8e92e219df0985f19b6addde_1440w.jpg)

-   项目密集描述（Itemic Dense Captioning）：从项目标记中解码项目的描述性内容。

![](https://picx.zhimg.com/v2-0e5801d3daefec099a267fb7e22ceff9_1440w.jpg)

-   通用语言建模（General Language Modeling）：在通用文本语料库上继续预训练模型。

## 2.2 [Reasoning Activation](https://zhida.zhihu.com/search?content_id=264273104&content_type=Article&match_order=1&q=Reasoning+Activation&zhida_source=entity)（推理激活）:

-   先sim选出和target最相关的top-k（10）个行为：

![](https://pic2.zhimg.com/v2-f05073938d28f203566c0be410b943db_1440w.jpg)

-   再用llm（文中只提到是一个语义对齐的模型）生成用户行为理由：

![](https://pic1.zhimg.com/v2-e4029d091d9434a54896555e5ca0c750_1440w.jpg)

-   最后作为监督信号，lora微调推理过程，loss为：

![](https://pica.zhimg.com/v2-7a6c726dd2ddbb4d0558b266d0ffbf62_1440w.jpg)

## 2.3 Reasoning Enhancement（推理增强）:

通过强化学习进一步优化推理路径，使用针对推荐任务设计的奖励机制。 具体方法包括：通过Beam搜索评估推理能力，使用GRPO算法优化模型。

![](https://pica.zhimg.com/v2-3f63cb00875ceb341a233a8433f29c08_1440w.jpg)

## 3 实验

llm基座模型为[Qwen-8B](https://zhida.zhihu.com/search?content_id=264273104&content_type=Article&match_order=1&q=Qwen-8B&zhida_source=entity)。

推荐demo：

![](https://pic4.zhimg.com/v2-12b5b70a02c850ebe89ea6eb045490a1_1440w.jpg)

在线效果：

![](https://pic3.zhimg.com/v2-1d6b723b16f583e2c05b3c26aaeb01f2_1440w.jpg)

## 一点碎碎念

感觉除了搜索场景存在query外，其他推荐场景的语义信息（SID）都不是很明显，搞NTP有点强行，所以现在很多声称生成式方法其实还是判别式任务。本文也提供了一个思路吧，没有语义信息我就构造语义信息（Reasoning Activation）然后去学习。

llm的token id序列是【 2，415，231，551】这样的，NTP很分散，生成式有分布可学，上限较高。推荐场景下，用户的兴趣较为集中（甚至存在信息茧房），本地生活场景更甚，用户的token序列可能是【2，2，3，1，2，2，1】，用户喜欢的商家（兴趣）是固定的，NTP很集中（其他场景SID之后估计也是这样）。在这样分布很简单的业务上，生成式是否大材小用。

![](https://picx.zhimg.com/v2-7154c98fa94dfa0bf17d315c0937b6b9_1440w.jpg)