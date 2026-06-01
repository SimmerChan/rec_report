# TokenMixer-Large 深度解读｜字节 RankMixer 2.0：推荐系统首个 15B 纯粹大模型
> 发布时间: 发布于 2026-04-27 12:06・北京
> 原文链接: https://zhuanlan.zhihu.com/p/2032067215688733888

---

## [TokenMixer-Large](https://zhida.zhihu.com/search?content_id=273843633&content_type=Article&match_order=1&q=TokenMixer-Large&zhida_source=entity)：字节跳动推荐系统首个 15B 纯粹大模型架构深度解读

**一句话概括：**在 [Self-Attention](https://zhida.zhihu.com/search?content_id=273843633&content_type=Article&match_order=1&q=Self-Attention&zhida_source=entity) 的二次方 FLOPs 和工业部署的毫秒级延迟双重夹击下，字节终于把推荐排序的参数规模捅到了百亿级——离线 15B、线上 7B，抖音电商 GMV **+2.98%**，广告主满意度 **+2.0%**，算力吞吐 **+96.6%**。这是 [RankMixer](https://zhida.zhihu.com/search?content_id=273843633&content_type=Article&match_order=1&q=RankMixer&zhida_source=entity) 之后，推荐大模型工业落地最硬核的一步。

本文基于 TokenMixer-Large 原始论文与字节跳动官方技术解析（《深度学习炼丹》公众号），结合自己在搜广推大模型领域的实践，做一次硬核的从动机到实现的全景梳理。

## 一、研究背景与动机

### 1.1 推荐系统的 Scaling Law 为什么一直"水土不服"

NLP 和 CV 已经用 GPT-4、Llama、ViT 把 Scaling Law 打穿：参数一加、数据一喂，智能就呈幂律上升。但把同样一套搬到推荐，大家很快发现：

-   **输入空间极度异构：**稀疏的 user id / item id、稠密的连续特征、超长行为序列、各种交叉统计量，挤在一个模型里的时候，[Transformer](https://zhida.zhihu.com/search?content_id=273843633&content_type=Article&match_order=1&q=Transformer&zhida_source=entity) 那套同构的 token 假设直接失效。
-   **Memory-bound 算子泛滥：**Wide&Deep、[DeepFM](https://zhida.zhihu.com/search?content_id=273843633&content_type=Article&match_order=1&q=DeepFM&zhida_source=entity)、DCNv2 这一票经典 DLRM 为了做特征交叉堆了大量琐碎算子，GPU 的 MFU（Model FLOPs Utilization）经常只有个位数。
-   **扩参无感：**AutoInt / [HiFormer](https://zhida.zhihu.com/search?content_id=273843633&content_type=Article&match_order=1&q=HiFormer&zhida_source=entity) / DHEN / [Wukong](https://zhida.zhihu.com/search?content_id=273843633&content_type=Article&match_order=1&q=Wukong&zhida_source=entity) 把 Self-Attention 强行塞进排序，参数一上来延迟就爆，而 AUC 几乎纹丝不动。

一句话：**推荐系统的 Scaling Law 不是不存在，而是被架构的维度不一致性和硬件层面的计算瓶颈死死按住了**。

### 1.2 前代 RankMixer 的四座大山

字节自家的 RankMixer（核心算子 TokenMixer）在 1B 规模下用无参数的 Token Mixing 替换 Self-Attention，MFU 被拉满。但继续往 10B 堆，出现了四个结构性问题：

-   **残差语义错位：**Mixing 改变了 Token 的数量/维度，直接 Add & Norm 会把不同语义的 token 在空间上叠在一起。深层一堆，原始语义彻底丢。
-   **深层梯度枯竭：**工业界默认只部署 2 层，堆到几十层底层参数根本更新不起来。
-   **[MoE](https://zhida.zhihu.com/search?content_id=273843633&content_type=Article&match_order=1&q=MoE&zhida_source=entity) 稀疏化是妥协：**前代走的是"稠密训练 + 稀疏推理"，训练成本还是吃满，ReLU-MoE 的激活数动态漂移让推理调度极其痛苦。
-   **架构不纯粹：**骨干里还残留 LHUC、DCNv2 这些历史包袱，拖累 GPU 并行效率。

### 1.3 为什么这个问题一直没被解决

本质上，这需要在**算法表达 × 系统架构 × 硬件通信拓扑**三个维度同时做极致协同。光改 Attention 公式不考虑内存访问连续性和 NVLink 带宽，最后就只能是个离线玩具。这也是为什么百亿级排序模型在工业界一直只停留在 PPT 上。

## 二、核心研究问题

> 在毫秒级在线延迟的严苛约束下，如何打破模型深度与参数规模的扩展壁垒，设计一种硬件友好的纯粹大模型架构，让推荐系统可以像 LLM 一样平滑、高效地扩展到 10B+ 参数，并把 Scaling Law 的红利真实地兑换成业务指标？

-   **输入：**抖音真实场景的 One-hot 稀疏特征（user 画像、item 属性、超长行为序列、交叉特征）。
-   **输出：**CTR / CVR 等排序概率，以及绑定的 GMV、ADSS 等商业目标。
-   **难点：**加宽 → 显存二次方爆；加深 → 语义漂移 + 梯度消失；不稀疏 → 推理打穿工业部署底线。

**核心理论假设：**推荐系统的 Scaling Law 是客观存在的，真正阻碍它发生的是_架构的维度不一致性_与_硬件层面的计算瓶颈_。只要能构造一种"输入输出维度绝对对称 + 梯度全层流畅传导 + 算力分布高度聚焦"的纯粹架构，扩参就是无限的。

## 三、TokenMixer-Large 方法论

整体哲学：**"纯粹模型（Pure Model）" + "先放大，后稀疏（First Enlarge, Then Sparse）"**。前向管线分三段：

1.  **语义分组 Token 化**：稀疏 One-hot → Embedding → 分组 MLP → 维度对齐的稠密 Token 矩阵。
2.  **TokenMixer-Large Block 堆叠**：Pre-[RMSNorm](https://zhida.zhihu.com/search?content_id=273843633&content_type=Article&match_order=1&q=RMSNorm&zhida_source=entity) → Mixing & Reverting → Pertoken SwiGLU / Sparse-Pertoken MoE → 残差，循环几十层。
3.  **输出聚合**：Mean Pooling → 多任务预测头（CTR / CVR / 多目标）。

### 3.1 语义分组 Tokenizer & Global Token

作者没有粗暴地把所有特征拼成一个大向量，而是按语义分成 G 组（用户画像组、长短期行为组、交叉组……），每组过独立 MLP 映射到同一维度 d\_t。在此基础上借鉴 BERT 的 \[CLS\]，用额外网络融合出一个 **Global Token**，拼成标准化的 N×d\_t 输入矩阵。这样第 0 层就自带全局视野。

### 3.2 Mixing & Reverting：整篇论文最关键的魔改

这是 TokenMixer-Large 对前代最核心的修正。假设输入 X ∈ R^{N×d}：

-   **Mixing**：沿特征维度切 H 个 head，把所有 Token 的第 i 个 head 拼在一起 → M × H' 的中间张量。全局信息在物理空间上强制交融，零参数成本。
-   **非线性处理**：中间张量过一层 pSwiGLU，并做一次本地残差。
-   **Reverting**：按原比例逆向切割，把属于同一个原始 Token 的 H 个切片重新拼回 N×d。维度 100% 无损还原。

这一步把"解构 → 融合 → 重构"做成了严格对称，**每个 Token 的"原始位置坐标"被精准保留**，标准残差 Residual 终于可以在对位上无损累加，前代那个语义错位的老毛病一次性根治。

### 3.3 让深层网络能被训起来：区间残差 + 辅助损失 + Pre-Norm

-   **区间残差（Inter-Residual）**：每隔 2~3 层搭一条跨层高速通道，底层梯度能直接上浮。
-   **辅助损失（Auxiliary Loss）**：抽浅层 logits 和深层 logits 联合计算 loss，强迫浅层提前拟合深层偏差，相当于给底层新手拉了一根"对标老专家"的强督促绳。
-   **Pre-Norm + RMSNorm**：相比前代的 Post-Norm，彻底根治了深层堆叠中常见的数值爆炸与梯度震荡。

### 3.4 Sparse-Pertoken MoE：冲击 15B 的终极武器

这是整篇论文最"算法美学"的一段。核心路线是**先放大、后稀疏**：

1.  先把单 Token 专属 SwiGLU 的 hidden dim 大幅拉高；
2.  再把这个放大的 SwiGLU 物理切成 E 个 sub-expert；
3.  推理时 Router 打分 Top-K 激活，并配一个永远激活的 **Shared Expert** 做兜底；
4.  训练和推理统一走稀疏路径，不再"稠密训练、稀疏推理"。

有两个工程细节值得单独标出来：

-   **Gate Value Scaling：**Softmax 路由的"和为 1"约束会让 SwiGLU 梯度偏小。作者把门控值乘上稀疏率的倒数（例如 1/4 激活时乘 4），完美补偿梯度，这是 MoE 训练能稳住的关键。
-   **Down-Matrix Small Init：**把 SwiGLU 降维矩阵的方差初始化为原来的 0.01 倍，训练初期逼近恒等映射，深层参数更新就会非常平滑。

### 3.5 系统级协同：Token Parallel + 自定义 Kernel + FP8

-   **MoEGroupedFFN 自定义 [CUDA Kernel](https://zhida.zhihu.com/search?content_id=273843633&content_type=Article&match_order=1&q=CUDA+Kernel&zhida_source=entity)**：所有专家在一个 kernel 里并行，大量减少算子调度开销。
-   **FP8 E4M3 量化推理**：AUC 不掉点的前提下，算力 ×1.7。
-   **Token Parallel**：沿 Token 维度切参数，和 Mixing/Reverting 的逻辑刚好咬合，L 层的 All-to-All 次数从 4L 压到 2L + 2，再用流水线掩盖通信延迟。**线上推理吞吐量 +96.6%**。

## 四、实验结果与分析

### 4.1 离线跨代打榜

在同 500M 参数量下（Table 2）：

-   TokenMixer-Large **CTCVR AUC 相对 +0.94%**，全面碾压 DLRM / DCNv2 / AutoInt / HiFormer / RankMixer。推荐 1% AUC = 巨幅提升，这个量级在工业排序里极其罕见。
-   算力性价比：FLOPs/Batch **4.2T**，DCNv2 是 125.8T，AutoInt 是 138.6T，差了 30 倍以上。

### 4.2 Scaling Law 硬核验证

这是全文最震撼的一段。作者在抖音 Feed Ads、E-Commerce、Live Streaming 三大场景把模型从 1B 一路推到 **15B**：

-   AUC 曲线几乎呈笔直的线性上升，**没有观察到传统的边际收益递减**。
-   想继续享受扩参红利，必须做"宽度 × 深度 × SwiGLU 扩展系数"三维平衡扩展（Balanced Expansion）。
-   数据饥渴真实存在：500M → 2B 需要 60 天以上的数据才能收敛。

### 4.3 Sparse-Pertoken MoE 带来的"免费午餐"

同样 4.6B 总参数下，1:2 稀疏（激活 2.3B）：

-   FLOPs 从 29.8T → **15.1T**（计算成本腰斩）；
-   CTCVR AUC 仍稳稳 **+1.14%**，与稠密版完全一致。

Gate Value Scaling + 小初始化的组合拳让 MoE 真正做到了"稀疏化零性能折损"。

### 4.4 抖音在线 A/B 收益

-   **直播电商（4B）**：订单 +1.66%，人均预览支付 GMV **+2.98%**。
-   **信息流广告（7B）**：广告主满意度 ADSS **+2.0%**。
-   **泛直播分发（2B）**：平台支付收入 **+1.4%**。

放到抖音这个基数上，每个小数点都是天文数字的商业价值。

## 五、对未来的启发

1.  **更高稀疏度 × 动态负载均衡：**目前线上稳定在 1:2，论文附录显示推到 1:8 会出现严重负载失衡和轻微掉点。接下来 Router Z-loss、无辅助损失路由会是下一片战场，1:16 的稀疏比值得期待。
2.  **多模态 × 长序列生成融合：**架构规整、高并行，天然适合当底层 Encoder。往后和 Vision Transformer、超长上下文 LLM 对齐，就是"从短视频像素 → 用户心智"的端到端统一基础模型。
3.  **软硬协同成为胜负手：**Token Parallel + FP8 已经证明了——AI 下半场，能不能把张量切分方式和 GPU 集群的 NVLink 拓扑深度绑定，比算法公式本身更关键。未来推荐算法工程师的能力边界要一直向下延伸到 CUDA 乃至指令集。

## 六、通俗版总结

如果把抖音的推荐模型想象成一家超级图书馆的调度系统：

-   **传统 DLRM**：一堆老式管理员手工分拣包裹，卡在流水线上动弹不得，GPU 算力用了不到 10%；
-   **RankMixer**：换了一套自动分拣机，效率飙升。但传送带一加长，不同城市的包裹就在深处混成一团，而且末端的传送带没有动力（梯度消失）；
-   **TokenMixer-Large**：为超长传送带重新设计了"拆解后再精确还原"的机制（Mixing & Reverting），每隔几段加了一个"中继加速马达"（区间残差 + 辅助损失），拆掉了老旧手工检查站（去除碎片化算子），最后还上了"弹性上班制"的专家调度（Sparse-Pertoken MoE）。

最终结果是一套 150 亿参数的超级大脑，不仅跑得快、还真的懂你——在抖音的真实流量上，直接给出了 GMV +3% 的答卷。

## 七、个人观点与思考

1.  **"纯粹模型"才是工业大模型的真解。**推荐领域常年堆算子的缝合怪时代该结束了，规整矩阵运算 + 纯粹 token 抽象才能兑现硬件红利。这和 Transformer 能在 NLP 一统天下的底层逻辑是一致的。
2.  **MoE 的关键不在稀疏本身，在梯度补偿。**Gate Value Scaling 和 Down-Matrix Small Init 看起来像两个小 trick，实际上是整套稀疏训练能稳下来的核心。任何人尝试 MoE 扩参之前，都建议先把这两个点吃透。
3.  **工业排序的下一步大概率是 Token Parallel + 多模态 Encoder。**接下来的排序会越来越像"超长上下文 + 多模态输入 + 稀疏专家 + 极致通信"的组合，边界会越来越模糊，搜推广和 LLM 终将走向同一个技术栈。

"纯粹模型"才是工业大模型的真解。推荐领域常年堆算子的缝合怪时代该结束了，规整矩阵运算 + 纯粹 token 抽象才能兑现硬件红利。这和 Transformer 能在 NLP 一统天下的底层逻辑是一致的。
MoE 的关键不在稀疏本身，在梯度补偿。Gate Value Scaling 和 Down-Matrix Small Init 看起来像两个小 trick，实际上是整套稀疏训练能稳下来的核心。任何人尝试 MoE 扩参之前，都建议先把这两个点吃透。
工业排序的下一步大概率是 Token Parallel + 多模态 Encoder。接下来的排序会越来越像"超长上下文 + 多模态输入 + 稀疏专家 + 极致通信"的组合，边界会越来越模糊，搜推广和 LLM 终将走向同一个技术栈。
有问题欢迎评论区讨论，也欢迎关注后续的搜广推大模型论文精读系列～