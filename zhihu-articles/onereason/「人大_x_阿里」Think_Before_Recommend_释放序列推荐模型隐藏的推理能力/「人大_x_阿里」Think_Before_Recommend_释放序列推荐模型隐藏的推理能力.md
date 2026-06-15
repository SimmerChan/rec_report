# 「人大 x 阿里」Think Before Recommend: 释放序列推荐模型隐藏的推理能力
> 发布时间: 编辑于 2025-04-01 16:52・美国
> 原文链接: https://zhuanlan.zhihu.com/p/1890431761223222035

---

![](https://pic1.zhimg.com/v2-a933f09c9648ca74f9d6fb85255f5f94_1440w.jpg)

## 研究背景

在电商、短视频、音乐流媒体等场景中，**[序列推荐系统](https://zhida.zhihu.com/search?content_id=255860534&content_type=Article&match_order=1&q=%E5%BA%8F%E5%88%97%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F&zhida_source=entity)**（Sequential Recommendation）通过捕捉用户历史行为的时序模式预测下一交互物品，是提升用户体验的核心技术。然而，传统方法（如SASRec、BERT4Rec）采用直接前向计算范式（如图 1a），仅依赖序列编码器的最终隐藏状态作为用户表示，存在两大核心缺陷：

-   **动态偏好建模不足**：有限计算深度难以捕捉用户兴趣的复杂演化；
-   **长尾物品理解偏差**：对低频、冷启动物品的表征能力受限。

![](https://pic4.zhimg.com/v2-8be8db40bb41ef546d156758064fd323_1440w.jpg)

## [ReaRec](https://zhida.zhihu.com/search?content_id=255860534&content_type=Article&match_order=1&q=ReaRec&zhida_source=entity)框架：**新一代推理增强的序列推荐**

受[大语言模型](https://zhida.zhihu.com/search?content_id=255860534&content_type=Article&match_order=1&q=%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity)（LLM）慢思考的启发，我们提出首个**推理时计算框架ReaRec**，在仅增加平均延迟 3.51%（固定2步隐式推理）的情况下提升 7.49% 的推荐表现。值得注意的是，通过事后的最优推理步数分析，我们惊喜地发现ReaRec 显著提升了序列推荐性能的天花板，在 [Yelp 数据集](https://zhida.zhihu.com/search?content_id=255860534&content_type=Article&match_order=1&q=Yelp+%E6%95%B0%E6%8D%AE%E9%9B%86&zhida_source=entity)不同序列 backbone 上实现了大约 30-50% 的平均性能提升，这展示了ReaRec这一新的范式的巨大前景。

![](https://pic3.zhimg.com/v2-83d440303e246c68af8aaf7767cf3e42_1440w.jpg)

我们具体通过以下方法实现推理增强的序列推荐：

### 1\. **[多步隐式推理机制](https://zhida.zhihu.com/search?content_id=255860534&content_type=Article&match_order=1&q=%E5%A4%9A%E6%AD%A5%E9%9A%90%E5%BC%8F%E6%8E%A8%E7%90%86%E6%9C%BA%E5%88%B6&zhida_source=entity)**

-   **推理链增强序列表征**：将序列编码器的最终隐藏状态自回归地输入模型，通过**K步推理**深化特征交叉（如图1b）。
-   **[推理位置嵌入](https://zhida.zhihu.com/search?content_id=255860534&content_type=Article&match_order=1&q=%E6%8E%A8%E7%90%86%E4%BD%8D%E7%BD%AE%E5%B5%8C%E5%85%A5&zhida_source=entity)（RPE）**：设计专用位置编码区分编码阶段与推理阶段，避免任务混淆。

### 2\. **轻量化推理学习策略**

**（1）集成推理学习（ERL）**

-   **多步监督优化**：利用不同推理步的输出构建多视角用户表示；
-   **KL散度正则化**：抑制推理输出的同质化，提升推理多样性。

**（2）渐进推理学习（PRL）**

-   **温度退火机制**：从平滑分布探索到尖峰分布聚焦，逐步逼近真实兴趣分布；
-   **对比去噪学习**：模拟推理误差自纠正，增强模型的推理鲁棒性。

## 挑战和机遇

目前我们的工作还是一个初步探索，因此我们也希望这个工作可以吸引大家一起来关注ReaRec这样的推荐范式，我们也列了一些未来潜在的研究问题：

-   **自适应推理深度选择**：事后的分析发现，对于低活跃的用户和长尾的用户组，越长的推理步数会带来更高的性能提升，而对于高活用户和流行物品，则提升有限甚至出现性能下降，这种类似过思考的问题导致我们提出的推理机制还与性能上界存在不小的差距。
-   **Inference-time Scaling Law**：实验结果来看，我们仍然没有实现理想中随着推理步数增加，性能逐渐上升（即使可能边际增益递减）的现象。这自然引起一些疑问：序列推荐任务是否能够实现 inference-time scaling law？如果可以的话，又该以怎样的精妙设计实现这一美好愿望呢？
-   **编码和推理参数空间的解耦**：本文提出推理位置编码来缓解物品编码和序列推理的任务差距，然而，未来是否可以考虑从模型参数解耦的角度来更好区分不同的任务也是一个值得探索的方向。
-   **高效的推理机制**：假如存在 scaling law 存在的话，如何降低推理时延对于实际工业界推荐系统的部署也是非常关键的一环，引入如线性注意力、模型量化、long-to-short 推理蒸馏机制可能是一条有前景的实现方法。
-   **理论分析**：直觉上，延伸序列推荐模型的inference-time reasoning 能力，能够让模型进行更深层次的特征交叉。进一步，是否可以从理论分析或者证明上给出方法的理论保证，提供原理驱动的推理增强推荐方法。

## 论文链接

[](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2503.22675)