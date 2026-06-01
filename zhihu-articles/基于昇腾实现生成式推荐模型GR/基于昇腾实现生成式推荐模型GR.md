# 基于昇腾实现生成式推荐模型GR
> 发布时间: 发布于 2025-06-09 15:09・上海
> 原文链接: https://zhuanlan.zhihu.com/p/1915425008597079412

---

## **摘要**

本文聚焦于[生成式推荐器](https://zhida.zhihu.com/search?content_id=258754254&content_type=Article&match_order=1&q=%E7%94%9F%E6%88%90%E5%BC%8F%E6%8E%A8%E8%8D%90%E5%99%A8&zhida_source=entity)（GRs）的创新技术与应用实践，着重介绍其核心组件 —— [分层序列转导单元](https://zhida.zhihu.com/search?content_id=258754254&content_type=Article&match_order=1&q=%E5%88%86%E5%B1%82%E5%BA%8F%E5%88%97%E8%BD%AC%E5%AF%BC%E5%8D%95%E5%85%83&zhida_source=entity)（HSTU）和jagger-Tensor并阐述了 GR 模型在昇腾设备上的适配流程。

在论文《Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations》中提出的生成式推荐器（GRs），创新性地将推荐问题转化为序列转导任务。通过设计分层序列转导单元（HSTU）架构，有效提升了推荐系统的性能。研究团队在合成数据、公共数据集以及大规模互联网平台上开展了全面的实验，验证了该方法的有效性，为推荐系统的发展开辟了新方向。 ​ 此外，值得关注的是，在相同参数条件下，昇腾设备的 NPU（神经网络处理器）与 GPU 相比，展现出了更优的性能表现。 ​ 在合成和公开数据集上的实验表明，HSTU 相比基线模型在 NDCG 指标上最多提升了 65.8%，且在处理 8192 长的序列时，比基于 [FlashAttention](https://zhida.zhihu.com/search?content_id=258754254&content_type=Article&match_order=1&q=FlashAttention&zhida_source=entity)2 的 Transformer 快 5.3x - 15.2x。基于 HSTU 的生成式推荐系统（包含 1.5T 参数量）在在线 A/B 测试中指标提升 12.4%，并已部署于拥有数十亿用户的互联网平台的多项服务中。更重要的是，生成式推荐的模型呈现出跨三个数量级的幂律增长特性，直到与 GPT-3/[LLaMa-2](https://zhida.zhihu.com/search?content_id=258754254&content_type=Article&match_order=1&q=LLaMa-2&zhida_source=entity) 的规模相当。

## **一. 背景与意义**

## **1.1 背景**

深度学习推荐模型（DLRMs）依赖大量特征，但计算扩展性差。受 Transformer 成功启发，重新审视推荐系统设计，将推荐问题作为序列转导任务，提出 GRs 和 HSTU 架构。分层序列转导单元（HSTU），通过多种优化技术如最小化激活内存使用、利用并增加稀疏性等提升编码器效率。 GRs 将排序和检索作为序列转导任务，通过 HSTU 编码器和新算法，提升了推荐系统性能和扩展性，简化了特征处理，为推荐、搜索和广告领域的基础模型发展奠定基础，对隐私保护和减少内容问题有积极影响。该模型适用于生成式推荐任务，能够根据用户的行为序列生成个性化的推荐内容，相比一些传统的推荐模型，在推荐的多样性和准确性方面可能更具优势，能够更好地满足用户的个性化需求。

## **1.2 优势**

**提升效率:** HSTU 编码器通过简化架构（减少线性层数量）、融合计算操作（如将查询、键、值和门控权重计算融合）以及优化层归一 化和随机失活等方式，降低激活内存使用，使编码器在处理用户行为序列和商品信息时更高效。 **提供个性化推荐**: 利用随机长度（SL）算法等技术，生成式推荐系统可以根据用户的历史行为和实时行为动态调整推荐策略。 **处理长序列数据**：在电商中，用户的购买历史和浏览历史通常构成较长的序列。生成式推荐系统通过利用输入序列的稀疏性，如通过 SL 算法增加稀疏性，能够有效处理长序列数据。

## **二. GRs设计**

分层序列转导单元（HSTU）编码器在生成式推荐系统中扮演着核心角色，其设计融合了多种优化策略，旨在提升编码器效率、减少内存使用并增强模型对用户行为序列的处理能力。

## **2.1 随机长度（SL）算法**

用户的历史交互行为通常具有不同的长度。[随机长度算法](https://zhida.zhihu.com/search?content_id=258754254&content_type=Article&match_order=1&q=%E9%9A%8F%E6%9C%BA%E9%95%BF%E5%BA%A6%E7%AE%97%E6%B3%95&zhida_source=entity)能够根据输入序列的特点动态调整计算量，提高模型的训练效率。通过随机长度算法，模型能够处理不定长的用户行为序列，提升推荐系统的性能。

```bash
# 封装历史用户行为序列的数据结构
features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids, # （B， seq_length）
        past_embeddings=None,
        past_payloads={
            "timestamps": historical_timestamps, # （B， seq_length）
            "ratings": historical_ratings,  # （B， seq_length）
        })
# 将物品 ID 转换为低维嵌入表示
item_feature = torch.nn.embedding(features.past_ids)
# 计算序列偏移长度
x_offsets=torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths)
# Jagged 到密集张量的转换：
padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
              values=item_feature, offsets=[x_offsets], max_lengths=[n], padding_value=0.0)
```

## **2.2 分层设计**

### **2.2.1 Pointwise Projection（逐点投影）**

特征提取逐点投影通常是将输入的特征向量通过线性变换（即与权重矩阵相乘）映射到新的特征空间

```bash
normed_x = self._norm_input(x)
batched_mm_output = torch.mm(normed_x, self._uvqk)
batched_mm_output = F.silu(batched_mm_output)
u, v, q, k = torch.split(
                batched_mm_output,
                [self._linear_dim * self._num_heads, self._linear_dim * self._num_heads, self._attention_dim * self._num_heads, self._attention_dim * self._num_heads],
                dim=1,
            )
```

### **2.2.2 Spatial Aggregation 特征交互**

通过注意力机制，计算不同位置特征之间的注意力分数，根据分数对特征进行加权聚合。 整合多个维度和位置信息。

```bash
qk_attn = torch.einsum(
                "bnhd,bmhd->bhnm",
                padded_q.view(B, n, self._num_heads, self._attention_dim),
                padded_k.view(B, n, self._num_heads, self._attention_dim),
            )
if all_timestamps is not None:
    qk_attn = qk_attn + self._rel_attn_bias(all_timestamps).unsqueeze(1)
```

### **2.2.3 Pointwise Transformation 表征转换**

通过逐点变换，模型可以对聚合后的特征进行精细化调整，挖掘特征中隐藏的信息，增强特征的表达能力，为生成准确的推荐结果提供更有效的特征表示。

```bash
o_input = u * self._norm_attn_output(attn_output)
```

### **2.3 内存高效的注意力机制**

HSTU 编码器基于 Memory-efficient Attention 和 FlashAttention，采用内存高效的注意力机制。该机制将输入划分为块，避免在反向传播中具体化大的中间注意力张量，从而减少了内存使用。在处理用户行为序列时，HSTU 编码器的注意力机制能够更有效地捕捉序列中的依赖关系，提升模型的推荐准确性。

```bash
qk_attn_seq_only = qk_attn_seq_only.reshape(
    (bs, self._num_heads, n //2,2, n //2,2))
rel_attention_mask = rel_attention_mask.unsqueeze(3).unsqueeze(-1)
qk_attn_seq_only = qk_attn_seq_only + rel_attention_mask
```

## **2.4 将排序和检索重新表述为生成式任务**

传统的排序和检索任务通常是从大量候选项目中找到与用户兴趣匹配的项目并进行排序。而生成式任务则是将其视为一个序列转导任务，即根据用户的历史行为（如浏览记录、购买记录等）以及其他相关信息（如用户属性、项目特征等）生成一个与用户兴趣相关的项目序列。 给定一个按时间顺序排列的 n 个 Token 序列 x₀, x₁, ..., xn-1（其中 xi ∈ X），以及这些 Token 被观察到的时间点 t₀, t₁, ..., tn-1，生成式任务将该输入序列映射为输出 Token y₀, y₁, ..., yn-1：

-   其中 yi ∈ X ∪ {∅}），yi = ∅ 表示 yi未定义。
-   用 Φi ∈ Xc（Xc ⊆ X）表示系统提供给用户的Content（例如，图像或视频）。由于新内容不断生成，Xc 和 X 不是固定不变的。
-   用 ai 表示用户对 Φi 做出某种Action（例如，点赞、跳过、视频完成并分享），其中 ai ∈ X。
-   用 nc 表示用户已交互的 Content 总数。

### **2.4.1 检索**

在推荐系统的检索环节，针对 Φi+1 ∈ Xc 学习一个分布 p(Φi+1 | ui)，其中 ui 表示用户在第 i 个 Token 处的表征。典型的目标是通过选择 arg maxΦ∈Xc p(Φ | ui) 来最大化某种奖励。这与标准的自回归设置有两点不同。 a.首先，对于 xi 和 yi 的监督信号不一定就是 Φi+1，因为用户可能对 Φi+1作出负面反应。 b.其次，当 xi+1 代表与非交互相关的分类特征（如人口统计信息）时，yi 是未定义的。

### **2.4.2 排序**

在 GR 中，排序面临独特的挑战，因为工业推荐系统通常需要“Target-aware”的公式化表示。在这种情境下，目标Φi+1与历史特征的“交互”需要尽早发生，这在标准自回归设置中是不可行的，因为“交互”通常发生较晚（例如，通过 Encoder 输出后的 Softmax 函数）。作者通过在如下图 Table 1 中交错排列 Item 和 Action 来解决这一问题，使得排序可以表述为 p(ai+1 |Φ0 ,a0 ,Φ1 ,a1 ,...,Φi+1)（在分类特征之前）。在实践中，作者应用一个小型神经网络将 Φi+1 处的输出转换为多任务预测。重要的是，这使得能够在一次处理中对所有nc 个交互应用目标感知的交叉注意力机制。

## **2.4 整体网络架构**

![](https://pica.zhimg.com/v2-2c4ccf6cdda85e0dedb72db92b9907a8_1440w.jpg)

## **三. NPU适配推荐系统GR模型流程**

## **3.1 适配说明**

本样例为开源Generative Recommendations模型, 将其迁移至NPU侧训练，并使用 NPU的HSTU融合算子来实现性能的优化。 模型参考的开源链接为[https://github.com/facebookresearch/generative%02recommenders](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/generative%2502recommenders)克隆源码并固定版本为:Commits on Dec 16, 2024，提交的SHA-1 hash值（提交ID）：bb389f9539b054e7268528efcd35457a6ad52439 验证运行的算力平台：Atlas 800T A2

![](https://pic4.zhimg.com/v2-dd75c8a5d9afb862db3d1a60f0c3aa19_1440w.jpg)

## **3.2 启动容器**

镜像下载地址：[https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5](https://link.zhihu.com/?target=https%3A//www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5) 该镜像中部分配套版本说明： 启动容器命令参考：

```bash
docker run \
-u root \
-it \
--name ${container_name} \
--net=host \
--shm-size="300g" \
--privileged \
-v /etc/localtime:/etc/localtime \
-e ASCEND_VISIBLE_DEVICES=0-7 \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /home:/home \
-v /root/ascend:/root/ascend \
-v /root/.ssh:/root/.ssh \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
${image_name} \
/bin/bash
​
```

## **3.3 安装依赖**

下载最新版本的mindxsdk-mxec-add-ons安装包： [https://clouddrive.huawei.com/hwshare/f3ea4909559eae5305e42c02b5c3f06c?type=email&fileId=135327&ownerId=2088277&fileSize=22379519&fileName=QXNjZW5kLW1pbmR4c2RrLW14cmVjLWFkZC1vbnMtcG9jLWxpbnV4LXg4Nl82NC50YXIuZ3o=&isFolder=false](https://link.zhihu.com/?target=https%3A//clouddrive.huawei.com/hwshare/f3ea4909559eae5305e42c02b5c3f06c%3Ftype%3Demail%26fileId%3D135327%26ownerId%3D2088277%26fileSize%3D22379519%26fileName%3DQXNjZW5kLW1pbmR4c2RrLW14cmVjLWFkZC1vbnMtcG9jLWxpbnV4LXg4Nl82NC50YXIuZ3o%3D%26isFolder%3Dfalse) 解压之后，进入文件夹mindxsdk-mxec-add-ons-poc：

```bash
cd mindxsdk-mxec-add-on-poccd torch_plugin
# 执行命令安装：
pip3 install torch_npu-2.1.0.post9-cp311-cp311-linux_x86_64.whl
```

## **3.4 安装适配算子**

重新进入文件夹 mindxsdk-mxec-add-ons, 安装需要的昇腾适配算子： jagged\_to\_padded\_dense、IndexSelect优化、 dense\_to\_jagged、asynchronous\_complete\_cumsum、gather\_for\_rank1

```bash
cd mindxsdk-mxec-add-ons-poc/mxrec_ops
bash mxrec_opp_asynchronous_complete_cumsum.run
bash mxrec_opp_dense_to_jagged.run
bash mxrec_opp_index_select_for_rank1_backward.run
bash mxrec_opp_jagged_to_padded_dense.run
bash mxrec_opp_gather_for_rank1.run
bash mxrec_opp_hstu_dense_forward.run
bash mxrec_opp_hstu_dense_backward.run复制
```

## **3.5 编译融合算子依赖的lib**

进入 torch\_library 文件夹：

```bash
cd mindxsdk-mxec-add-ons-poc/torch_library
cd 2.1.0/hstu
bash build_ops.sh
```

如果在编译过程中有gcc版本的问题：

![](https://pic4.zhimg.com/v2-927c31b244f8661eccd5ade986832991_1440w.jpg)

重新设置gcc路径:

```bash
which gcc
which g++unset CC CXX
CC=/usr/local/gcc10.2.0/bin/gcc
CXX=/usr/local/gcc10.2.0/bin/g++ bash build_ops.sh
```

执行完以上命令之后，融合算子的依赖包libhstu\_dense\_ops.so会生成在同目录下的 build文件夹下，可将该so包拷贝到某固定目录下。示例如下：

```bash
cp ./build/libhstu_dense_ops.so /home/torch_ops/
```

## **3.6 下载源码**

```bash
git clone https://gitee.com/ascend/RecSDK.gitcd RecSDK
git checkout branch_v7.0.0-POC_torchcd ./torch/examples/generative_recommenders/npu
git clone https://github.com/facebookresearch/generative%02recommenders.git
```

下载开源的gr模型，以及昇腾的RecSDK **模型修改：** 将 Generative Recommendations 模型迁移到NPU上并适配NPU融合HSTU算子，代码 修改部分已经编写在 NPU\_GR.patch 中，载入命令如下：

```bash
cd generative-recommenders
cp ../NPU_GR.patch ./
git checkout bb389f9539b054e7268528efcd35457a6ad52439
git apply NPU_GR.patch
```

数据集下载

```bash
python preprocess_public_data.py
```

**代码修改：**

```bash
vim ./generative_recommenders/trainer/train.py
注释掉from msprobe.pytorch import seed_all
```

msprobe是精度模式下对比，与GPU的loss进行对比（Ascend内部插件) msprobe路径：[https://gitee.com/ascend/mstt/tree/master/debug/accuracy\_tools/msprobe](https://link.zhihu.com/?target=https%3A//gitee.com/ascend/mstt/tree/master/debug/accuracy_tools/msprobe) **config文件：** 创建一个hstu-mt-3400.gin文件，文件内容如下。将该gin文件放置在 generative\_recommenders/configs/ml-1m/ 目录下

```bash
train_fn.dataset_name = "ml-1m"
train_fn.max_sequence_length = 3389
train_fn.local_batch_size = 32
train_fn.main_module = "HSTU"
train_fn.dropout_rate = 0.2
train_fn.user_embedding_norm = "l2_norm"
train_fn.num_epochs = 1
train_fn.item_embedding_dim = 512
hstu_encoder.num_blocks = 3
hstu_encoder.num_heads = 2
hstu_encoder.dqk = 256
hstu_encoder.dv = 256
hstu_encoder.linear_dropout_rate = 0.2
train_fn.learning_rate = 1e-3
train_fn.weight_decay = 0
train_fn.num_warmup_steps = 0
train_fn.interaction_module_type = "DotProduct"
train_fn.top_k_method = "MIPSBruteForceTopK"
train_fn.loss_module = "SampledSoftmaxLoss"
train_fn.num_negatives = 128
train_fn.eval_interval = 50
train_fn.sampling_strategy = "local"
train_fn.temperature = 0.05
train_fn.item_l2_norm = True
train_fn.l2_norm_eps = 1e-6
train_fn.enable_tf32 = True
train_fn.precision_mode = False
​
create_data_loader.prefetch_factor = 128
create_data_loader.num_workers = 8
```

hstu\_encoder.dqk和hstu\_encoder.dv参数配置必须是16的整数倍。注意力机制的query,key,value的维度信息） **运行命令：** 修改run.sh 脚本， 使用hstu-mt-3400.gin配置文件： 当使用hstu融合算子时，hstu\_encoder.dqk，hstu\_encoder.dv必须为16的整数倍

```bash
export USE_NPU_HSTU=1# 使用融合算子
export ENABLE_RAB=0export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
ASCEND_RT_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/ml-1m/hstu-mt-3400.gin --master_port=12345 | tee temp.log
```

使用hstu-sampled-softmax-n128-large-final.gin，配置文件： 不使用融合算子

```bash
export USE_NPU_HSTU=0# 不使用融合算子
export ENABLE_RAB=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
​
ASCEND_RT_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-final.gin --master_port=12345 | tee temp.log
```

**执行命令：**

```bash
bash run.sh
```

## **四. NPU与GPU性能测试结果数据集**

![](https://pica.zhimg.com/v2-87a254e703071e0365f4ed76da386224_1440w.jpg)