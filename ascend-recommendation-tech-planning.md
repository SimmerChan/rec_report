# 昇腾推荐团队技术规划白皮书

> 面向互联网客户的推荐系统昇腾平台支撑团队
> 定位：从GPU迁移支撑到昇腾原生推荐模型开发的全栈技术团队
> 撰写时间：2026年5月

---

## 一、背景与定位

### 1.1 团队核心定位

华为昇腾推荐团队的核心使命是：**支撑互联网客户从GPU平台迁移到昇腾平台，或在昇腾平台原生开发推荐模型，进行训练和推理**。

这一一定位包含三层含义：

| 层次 | 含义 | 核心能力要求 |
|------|------|-------------|
| **迁移支撑** | 帮助客户将已有推荐模型从CUDA/GPU迁移到昇腾NPU | PyTorch API兼容、迁移工具链、精度调优 |
| **性能对标** | 在昇腾平台上达到或超过GPU的训练/推理性能 | 算子优化、MFU提升、CANN深度调优 |
| **原生开发** | 支撑客户在昇腾平台上开发新一代推荐模型 | 框架支持（TorchRec）、算子库（FBGEMM/HKV/RecOps）、生成式推荐模型适配 |

### 1.2 目标客户画像

昇腾推荐团队服务的互联网客户（中台推荐团队）具有以下特征：

- **头部互联网公司**（字节/快手/腾讯/阿里/美团/小红书等）
- 拥有千人级推荐中台团队，具备独立模型研发能力
- 现有推荐系统构建在NVIDIA GPU集群上（以A100/H100/H800为主）
- 正在探索或推进生成式推荐、Transformer序列建模、判别式Scaling等新技术
- 核心痛点：GPU算力资源受限、训练成本高、推理效率瓶颈

### 1.3 技术趋势对团队的影响

基于本报告对生成式推荐技术演进的调研，以下趋势直接影响昇腾推荐团队的技术规划：

| 技术趋势 | 对昇腾团队的影响 | 时间窗口 |
|---------|----------------|---------|
| **判别式Scaling路线主导**（字节TokenMixer-Large/快手UniMixer）| 70B+参数大模型训练→需要大算力集群，昇腾替代GPU窗口期 | 当前~2027 |
| **生成式GR路线深耕**（Meta ULTRA-HSTU）| 自定义attention+超长序列→需要昇腾NPU专用算子支持 | 2026~2028 |
| **广告端到端生成突破**（美团EGA-V2）| 生成式广告模型→new computation pattern，昇腾需提前适配 | 2026~2027 |
| **推理serving成一等公民**（字节UG-Separation）| 推理优化→NPU推理加速库需求激增 | 当前~2027 |
| **LLM推荐Scaling Law验证**（首个幂律）| LLM推荐预训练→合成数据生成+训练pipeline，PyTorch生态深度依赖 | 2026~2028 |
| **推理时Scaling**（beam search/dynamic serving）| 推荐系统竞争延伸至推理阶段，类似LLM test-time scaling | 2026~2028 |
| **统一化接口形成**（字节OneTrans/MixFormer/MDL）| 推荐系统从"模型更大"演进到"接口重写" | 当前~2028 |

---

## 二、行业现状：互联网客户推荐系统中台技术栈

### 2.1 头部互联网公司推荐系统中台现状

基于调研，互联网公司推荐中台的技术栈呈现以下分层结构：

**基础设施层：**
- GPU集群：千卡~万卡级别，以NVIDIA A100/H100/H800为主
- 特征平台：支持百亿级特征、毫秒级特征获取
- 样本存储：PB级样本存储，支持高速读取
- 推理引擎：TF-Serving/TorchServe/vLLM等，支持线上毫秒级响应

**模型训练层：**
- 框架：PyTorch（为主）+ TensorFlow
- 分布式训练：PyTorch DDP/FSDP、DeepSpeed
- 推荐专用框架：TorchRec（Meta开源）、FASTER-CSR
- 序列建模：Standard Transformer → HSTU → ULTRA-HSTU演进

**模型层（按技术路线分）：**
| 技术路线 | 代表模型 | 主流厂商 |
|---------|---------|---------|
| 传统判别式CTR | DeepFM/DIN/DIEN/DCN | 各厂商标配 |
| **判别式Transformer CTR** | RankMixer/TokenMixer-Large/GRAB | 字节/快手/百度 |
| **生成式GR** | Meta GR/HSTU、快手OneRec、美团MTGR | Meta/快手/美团 |
| **Semantic ID检索** | Google TIGER、腾讯OnePiece | Google/腾讯 |
| **LLM增强推荐** | 字节HLLM、阿里LUM、京东xLLM | 字节/阿里/京东 |

### 2.2 互联网客户对昇腾迁移的核心需求

头部互联网客户在考虑GPU→昇腾迁移时，关注点按优先级排序：

| 优先级 | 需求 | 信息来源与依据 |
|--------|------|--------------------------|
| **P0** | **PyTorch API兼容性** | 行业通用迁移需求。GPU→昇腾迁移的核心前提是PyTorch代码零改动或最小改动，参考CUDA→NPU通用迁移经验。 |
| **P0** | **性能对标GPU** | 行业通用迁移需求。MFU、延迟、吞吐是评估AI加速器的通用指标，各厂商采购昇腾前必然要求性能对标测试。 |
| **P1** | **算子支持完整性** | 基于技术分析推断。embedding look-up和sparse operation是推荐模型的计算瓶颈（占整体延迟60%+），是PyTorch NPU适配的优先级焦点。参考知乎知乎话题讨论中工程师提到的"embedding必须快"。 |
| **P1** | **迁移工具链成熟度** | 基于行业最佳实践推断。CUDA→NPU迁移需要自动迁移工具（AST重构）、精度调优工具、Profile工具，是昇腾官网文档明确列出的核心能力。 |
| **P2** | **分布式训练支持** | 行业通用需求。头部互联网客户使用千卡集群进行推荐模型训练，多机多卡通信效率（HCCL）和梯度同步是刚需，昇腾官网有对应文档。 |
| **P2** | **生成式模型适配** | 基于技术演进趋势推断。生成式推荐（HSTU、OneRec等）是2025-2026年业界重点研究方向，互联网客户技术团队在跟进，昇腾官网已发布Recsys-GR生成式推荐样例。 |

### 2.3 当前迁移的核心挑战

| 挑战类型 | 具体问题 | 昇腾现状 |
|---------|---------|---------|
| **API兼容性** | `torch.cuda` → NPU后端API差异，需要代码修改 | 昇腾提供torch_npu插件，但部分API有差异 |
| **算子支持** | 推荐模型依赖FBGEMM（sparse embedding）、HKV（hash embedding）、RecOps（专用算子） | 昇腾已支持FBGEMM/HKV/RecOps，但版本更新滞后 |
| **自定义算子** | HSTU的HSTUAttention、MoE、FlashAttention等自定义算子 | 需要昇腾自定义算子适配 |
| **通信瓶颈** | 多机多卡集合通信（NCCL→HCCL） | HCCL已支持，但性能调优经验不足 |
| **精度问题** | FP16/BF16混合精度训练精度对齐 | 需要专业精度调优工具链 |
| **生成式推荐支持** | 生成式推荐的自回归解码、beam search等模式 | 昇腾Recsys-GR（Rec SDK）已支持，但需验证 |

---

## 三、昇腾推荐开发生态现状分析

### 3.1 昇腾官方推荐解决方案全栈

基于昇腾官网（hiascend.com/developer/recommendation）信息，昇腾推荐的完整技术栈如下：

```
┌─────────────────────────────────────────────────────┐
│                    应用层                            │
│  TorchRec-V1（传统推荐）│ TorchRec-V2（生成式推荐）    │
│  Recsys-GR（Meta GR适配）│ 客户自研模型               │
├─────────────────────────────────────────────────────┤
│                 AI框架层                            │
│  PyTorch（主）│ TensorFlow（辅）│ MindSpore（可选）    │
├─────────────────────────────────────────────────────┤
│                 加速库层                            │
│  Rec SDK（推荐专用算子）│ FBGEMM（稠密Embedding）    │
│  HKV（Hash Embedding）│ RecOps（推荐专用操作）       │
├─────────────────────────────────────────────────────┤
│                 运行时层                            │
│  CANN（异构计算架构）│ torch_npu（PyTorch NPU后端）  │
├─────────────────────────────────────────────────────┤
│                 硬件层                              │
│  昇腾910系列（NPU）│ 昇腾910B/910C（当前主力）      │
└─────────────────────────────────────────────────────┘
```

### 3.2 昇腾推荐解决方案关键产品

| 产品 | 类型 | 说明 |
|------|------|------|
| **MindSDK** | 训练框架 | 昇腾推荐的统一训练SDK，封装底层CANN |
| **Rec SDK** | 推理/训练加速库 | 搜索推荐场景专用，支撑客户快速部署 |
| **CANN** | 异构计算架构 | 昇腾硬件的底层驱动，类似CUDA Toolkit |
| **MindStudio** | 全流程工具链 | 模型开发、调试、性能分析、部署 |
| **TorchRec-V1** | 训练框架 | 基于Meta TorchRec的传统推荐模型开发框架 |
| **TorchRec-V2** | 训练框架 | 基于Meta TorchRec的生成式推荐（Recsys-GR）开发框架 |
| **FBGEMM** | 算子库 | Sparse Embedding查表加速，已在昇腾上支持 |
| **HKV** | 算子库 | Hash Embedding查表加速，已在昇腾上支持 |
| **RecOps** | 算子库 | 推荐场景专用算子集合 |

### 3.3 PyTorch NPU后端支持现状

昇腾通过`torch_npu`插件提供PyTorch NPU后端，关键支持情况：

| 维度 | 支持现状 | 说明 |
|------|---------|------|
| **API兼容度** | 约85%（昇腾官方称） | 常用API（`torch.tensor`/`torch.nn`/`torch.optim`等）基本兼容；`torch.cuda.*`部分有对应NPU API |
| **自定义算子** | 需通过CANNacl/C++扩展 | 推荐模型的FBGEMM/HKV等有昇腾实现，HSTU等自定义算子需适配 |
| **分布式训练** | DDP/FSDP多机多卡支持 | 单机多卡成熟，多机多卡在完善中 |
| **混合精度** | FP16/BF16/FP32 | CANN支持混合精度训练，精度调优工具链完善 |
| **推理优化** | AOTI（ ahead-of-time ）编译推理 | 支持静态图编译优化推理延迟（昇腾官方称"敬请期待"） |
| **动态Shape** | 部分支持 | 序列建模的变长序列需关注 |
| **FlashAttention** | 需验证 | 推荐模型attention变体多，需确认昇腾支持情况 |

### 3.4 生成式推荐模型支持现状

**TorchRec-V2（生成式推荐框架）：**
- 昇腾官网已提供TorchRec-V2模型基础样例和基础模型开发文档
- 特别推出了"TorchRec-V2模型开发（Recsys-GR）"专项文档，对应Meta GR/HSTU的生成式推荐范式
- 这说明昇腾已有**生成式推荐的初步支持框架**，但具体模型（如HSTU/OneRec）的适配完整度待验证

**Recsys-GR（生成式检索）：**
- 昇腾在Recsys-GR上的定位：支撑生成式推荐模型在昇腾上的开发
- 关键问题：HSTU的自定义attention机制是否已在昇腾上完整实现？

---

## 四、技术规划：三大战略方向

### 战略方向一：GPU→昇腾迁移支撑（根基）

**核心目标**：让互联网客户"零阻力"将现有推荐模型从GPU迁移到昇腾平台。

#### 1.1 PyTorch API兼容度提升（P0）

| 工作项 | 目标 | 验收标准 |
|--------|------|---------|
| `torch.cuda` → NPU API统一抽象层 | 提供透明替换接口，客户代码改动最小化 | 90%+的CUDA代码无需修改即可迁移 |
| PyTorch原生API支持度达到95%+ | 消除迁移阻断项 | 主流推荐模型（DIN/DIEN/DeepFM等）无需修改即可运行 |
| 动态Shape API补充 | 支持推荐模型变长序列 | 序列建模类模型支持变长batch |

#### 1.2 推荐模型专用算子库完善（P0）

| 工作项 | 目标 | 说明 |
|--------|------|------|
| **FBGEMM算子对齐** | 性能对标NVIDIA CUTLASS | Sparse embedding lookup是推荐模型最大瓶颈，FBGEMM性能直接影响模型训练吞吐 |
| **HKV算子优化** | hash效率提升 | Hash embedding是多哈希场景的核心，昇腾已有支持但需优化 |
| **RecOps算子扩展** | 覆盖新增推荐专用算子 | 如HSTU的HSTUAttention、MoE的topK等 |
| **FlashAttention适配** | 在昇腾NPU上实现FlashAttention变体 | 推荐模型的attention不是标准Transformer，需要定制 |

#### 1.3 迁移工具链完善（P1）

| 工作项 | 目标 |
|--------|------|
| **自动迁移脚本** | 输入PyTorch CUDA代码，自动输出昇腾NPU代码 |
| **精度调优工具链** | 快速定位和修复精度问题（FP16溢出等） |
| **Profiling工具升级** | 支持昇腾NPU的MFU、算子级耗时分析 |
| **多机多卡迁移指南** | 从单机单卡→单机多卡→多机多卡的完整迁移文档 |

#### 1.4 分布式训练能力补齐（P1）

| 工作项 | 目标 | 关键指标 |
|--------|------|---------|
| FSDP on 昇腾 | 支持70B+参数模型在昇腾上的分布式训练 | 千卡集群训练MFU达到40%+ |
| 梯度同步优化 | HCCL通信效率提升 | 多机多卡训练线性加速比达到85%+ |
| 断点续训 | 支持超大规模训练的故障恢复 | 故障后训练恢复时间<30分钟 |

**→ 战略意义**：迁移支撑是团队的根本。只有迁移成本足够低、性能足够好，互联网客户才有动力选择昇腾。这是团队存在的根基，也是短期最直接的收益来源。

---

### 战略方向二：昇腾原生推荐模型开发（差异化竞争力）

**核心目标**：在昇腾平台上构建比GPU平台更具竞争力的推荐模型开发环境，吸引客户"原生开发"而非"迁移"。

#### 2.1 TorchRec-V2生成式推荐框架深化（P0）

| 工作项 | 目标 | 参考来源 |
|--------|------|---------|
| **Recsys-GR完整适配** | 支撑Meta GR/HSTU在昇腾上的完整训练推理 | 昇腾官网已有"TorchRec-V2模型开发（Recsys-GR）"文档 |
| **HSTU算子集完整实现** | HSTUAttention、RoPE、Asoftmax等昇腾原生实现 | Meta GR论文arXiv:2402.17152 |
| **OneRec适配** | 支撑快手OneRec在昇腾上的训练和部署 | 快手OneRec论文arXiv:2502.18965 |
| **超长序列支持** | 支持16K+序列的Training | Meta ULTRA-HSTU已验证16K序列是关键节点 |
| **端到端生成推理** | 支持生成式推荐模型的自回归解码 | 生成式推荐的核心计算模式 |

#### 2.2 判别式Scaling模型支撑（P1）

| 工作项 | 目标 | 说明 |
|--------|------|------|
| **TokenMixer-Large适配** | 支撑70B/150B参数判别式大模型在昇腾上的训练 | 字节TokenMixer-Large已验证这条路线的有效性 |
| **MoE稀疏激活优化** | topK gating等操作在昇腾上的性能优化 | MoE是控制计算量的关键 |
| **UG-Separation推理适配** | 支撑字节UG-Separation的推理优化（用户组分离） | 字节的推理优化方案 |
| **混合精度精细化** | BF16+FP8+INT4混合精度，内存优化 | ULTRA-HSTU已用混合精度将内存从7GB→2.3GB/层 |

#### 2.3 广告端到端生成支持（P1）

| 工作项 | 目标 | 说明 |
|--------|------|------|
| **美团EGA-V2适配** | 支撑广告端到端生成在昇腾上落地 | 美团EGA-V2已验证广告生成的可行性 |
| **多目标Loss融合** | 推荐/广告/互动的多目标训练pipeline | 广告场景的核心需求 |
| **出价/计费逻辑融合** | 支持广告系统的出价逻辑与生成式模型对接 | 生成式推荐在广告场景落地的最大难点 |

#### 2.4 训推一体平台建设（P2）

| 工作项 | 目标 |
|--------|------|
| **推理AOTI编译** | 昇腾官方"敬请期待"的AOTI推理需优先实现 |
| **推理延迟对标** | 达到或超过GPU推理延迟（TF-Serving/TorchServe对比） |
| **Batch推理优化** | 支持动态batch，最大化推理吞吐 |
| **Pipeline并行** | 训练-推理一体化，减少数据重传 |

**→ 战略意义**：差异化竞争力在于让客户在昇腾平台上能开发出在GPU上无法实现或成本更高的模型。超长序列、端到端生成、训推一体是三个关键差异化点。

---

### 战略方向三：PyTorch生态跟进与前沿模型适配（与时俱进）

**核心目标**：确保昇腾推荐团队能紧跟PyTorch生态和生成式推荐前沿发展，不掉队。

#### 3.1 PyTorch生态跟进机制（P0）

| 工作项 | 目标 | 说明 |
|--------|------|------|
| **PyTorch版本同步** | PyTorch新版本发布后3个月内完成昇腾适配 | 推荐系统依赖的PyTorch新API、分布式训练新特性等 |
| **CANN版本对齐** | 与PyTorch版本同步更新CANN适配层 | CANN是昇腾的底层异构计算架构 |
| **开源框架同步** | TorchRec/TorchServe等Meta开源框架的昇腾适配 | Meta推荐团队（miheyl）已开源TorchRec，昇腾需快速跟进 |

#### 3.2 前沿模型快速适配通道（P1）

| 工作项 | 目标 | 时间要求 |
|--------|------|---------|
| **arXiv新论文算子评估** | arXiv推荐论文发布后1个月内完成算子需求分析 | 建立arXiv论文快速评估流程 |
| **Meta HSTU官方实现适配** | Meta HSTU开源实现发布后快速移植到昇腾 | Meta GR/HSTU是生成式推荐的事实标准 |
| **生成式推荐Benchmark** | 建立昇腾vs GPU的生成式推荐性能Benchmark | 量化昇腾相对GPU的优劣势 |
| **LLM推荐Scaling Law适配** | 首个LLM推荐Scaling Law（arXiv:2602.07298）的方法论跟进 | 合成数据+幂律验证是新的训练范式 |

#### 3.3 技术预研储备（P2）

| 工作项 | 目标 | 说明 |
|--------|------|------|
| **Linear Attention预研** | 为超长序列（如ULTRA-HSTU的16K序列）储备轻量attention方案 | 线性attention可降低序列建模的计算复杂度 |
| **状态空间模型预研** | SUAN/Mamba等状态空间模型在推荐序列建模上的可行性 | 美团SUAN已探索此方向 |
| **合成数据生成Pipeline** | 为LLM推荐Scaling Law验证提供合成数据生成工具 | 生成式推荐Scaling依赖合成数据 |
| **推理时Scaling预研** | beam search、path-level reward等推理时计算分配方案 | 字节的推理优化思路 |

**→ 战略意义**：推荐系统技术演进快（2-3年一个技术周期），昇腾团队必须建立快速跟进能力。掉队的技术窗口期就是客户流失的窗口期。

---

## 四-补充：行业调研新发现（2026年5月调研）

### 补充1：互联网客户迁移昇腾的真实痛点（来自知乎社区）

**核心发现**：一位开发者在知乎分享了从NVIDIA迁移到昇腾300I Duo的实际经历，揭示了真实的迁移痛点：

**痛点一：CPU-NPU数据传输误判**
- 迁移后 `.to(cpu)` 操作耗时异常，误以为是数据传输慢
- 根因：推理计算本身（NPU侧）比GPU慢，`to(cpu)` 只是在等待计算完成
- 启示：昇腾调优需要**profiling导向**，不能凭直觉判断瓶颈

**痛点二：算子融合不足**
- 模型中存在大量碎片化 Small Ops（如频繁 Tensor 转换、未融合的激活函数）
- 导致 NPU AI Core 利用率呈"锯齿状"
- 启示：昇腾的算子融合优化空间巨大，是性能提升的关键

**痛点三：JIT在线编译开销**
- 动态Shape场景下，昇腾的JIT编译开销显著
- 解决方案：禁用算子在线编译，使用 `torch_npu.npu.set_compile_mode(jit_compile=False)`
- 启示：小模型/静态图场景有成熟优化方案，动态场景需特别关注

**优化结果**：经系统调优后，昇腾300I Duo推理耗时从1.51s降至0.68s，**优于NVIDIA A10的1.02s**，实现30%性能超越。

### 补充2：Meta GR/HSTU在昇腾上的适配进展

**核心发现**：昇腾官方和社区已有基于昇腾实现生成式推荐GR的详细教程，关键信息：

**适配进展**：
- 昇腾已支持Meta GR模型向NPU迁移训练
- 使用昇腾HSTU融合算子实现性能优化
- 适配平台：Atlas 800T A2
- 配套工具：mxrec_ops（昇腾推荐算子集）

**已实现的融合算子**：
- `jagged_to_padded_dense` - 变长序列转密集张量
- `index_select优化` - embedding查表加速
- `dense_to_jagged` - 密集到变长转换
- `asynchronous_complete_cumsum` - 累积和算子
- `gather_for_rank1` - 排序查表
- `hstu_dense_forward/backward` - HSTU核心融合算子

**关键参考资源**：
- 镜像：rec_sdk-torch（内含TorchRec V1/V2、PyTorch、torch_npu、FBGEMM、HKV、CANN）
- 来源：昇腾官网 https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5

### 补充3：行业最新技术路线总结（2025-2026大厂路线）

**四条并行主线**（基于知乎文章《从 RankMixer 到 OneRanker》综合分析）：

| 主线 | 代表工作 | 核心命题 | 昇腾支撑需求 |
|------|---------|---------|-------------|
| **大Ranking Backbone可扩展化** | RankMixer→TokenMixer-Large→MSN→UG-Separation | 70B+参数判别式模型训练与Serving | FSDP、混合精度、MoE优化、推理复用 |
| **长序列建模工业化** | STCA→LEMUR→LASER→LONGER | 16K+序列如何训、推、存 | 超长序列算子、变长batch优化 |
| **统一Backbone** | OneTrans→MixFormer→MDL→MTFM | 多模块统一为更少更强的主干 | 统一算子库、灵活算子融合 |
| **Semantic Token与生成式One-Model** | OneRec→GR4AD→GPR→OneRanker→OxygenREC | 端到端生成式推荐+广告 | 生成式解码、beam search、Semantic ID |

**各公司技术路线速览**：

| 公司 | 技术路线 | 代表模型 | 关键创新 |
|------|---------|---------|---------|
| **字节** | 判别式Scaling为主 | RankMixer→TokenMixer-Large→UG-Separation | Token Mixer、用户组分离Serving |
| **快手** | 生成式One-Model优先 | OneRec→OneLoc→OneMall→GR4AD→GRank | 端到端生成、Lazy Decoder、GR广告 |
| **美团** | 双线并进 | MTFM（判别式统一）+MTGR/EGA（生成式广告） | Foundation Model、统一多分布 |
| **腾讯** | 生成式广告深耕 | GPR→OneRanker | Target-aware生成、KV透传 |
| **阿里** | 判别式Transformer优化 | SORT | Systematically Optimized Ranking TRM |
| **京东** | 生成式电商推荐 | OxygenREC | 慢思考+快生成分离架构 |
| **LinkedIn** | 稳健判别式 | Feed-SR | Transformer序列排序、RoPE |

### 补充4：生成式推荐落地现状与瓶颈分析

**当前落地进度**（基于社区调研）：

| 场景 | 头部公司 | 进度评估 | 瓶颈 |
|------|---------|---------|------|
| 短视频推荐 | 快手OneRec | **已上线**（+1.6%停留时长） | 训练成本、RLHF标注 |
| 广告生成 | 腾讯GPR/OneRanker、快手GR4AD | **快速推进**（2026年加速） | 价值对齐、beam constraint |
| 电商推荐 | 京东OxygenREC | **早期** | 在线延迟、instruction-following |
| 排序精排 | 各公司生成式Ranking尝试 | **探索中** | 判别式vs生成式效果差距 |

**生成式推荐的三个关键收益来源**：
1. **端到端优化**：消除多阶段目标割裂，全局最优
2. **GPU MFU提升**：传统CTR模型MFU仅4-5%，生成式架构MFU可达23-28%
3. **推理时Scaling**：beam size动态调整，计算资源按需分配

### 补充5：昇腾竞争态势SWOT补充

**Strengths补充**：
- 昇腾NPU在特定场景（如HSTU推理）实测**已能超越同档次NVIDIA GPU**
- 推荐专用算子（FBGEMM/HKV）是昇腾差异化竞争点
- 国产替代政策加持，头部互联网有强动力适配昇腾

**Weaknesses补充**：
- **JIT编译开销**：动态Shape场景性能受损明显
- **Profiling工具链**：MindStudio上手成本高于NVIDIA Nsight
- **社区生态**：PyTorch NPU社区远小于CUDA社区

**Opportunities补充**：
- DeepSeek V4优先适配昇腾示范效应，互联网客户关注度提升
- 生成式推荐Scaling Law验证，新架构对GPU依赖降低，昇腾窗口期出现
- 推理时Scaling新范式，昇腾可在推理侧建立差异化优势

**Threats补充**：
- NVIDIA通过H200/H100持续增强AI生态，CUDA惯性持续
- 寒武纪、海光等国产芯片也在抢占替代市场
- 生成式推荐快速演进，如果昇腾跟进慢，客户可能选择其他国产方案

---

## 四-补充2：竞争力项目建议（基于昇腾现有资产的差异化定位）

> **说明**：基于对昇腾现有代码资产的调研（RecSDK、torch_npu、vLLM-ascend），重新校准各项目的定位。
> 核心原则：**不重复造轮子，基于昇腾已有能力构建差异化竞争力**。

---

### 项目定位校准总结

| 拟建项目 | 昇腾现有资产 | 重新定位 | 竞争力来源 |
|---------|-------------|---------|-----------|
| **Ascend-RecOps** | RecSDK `cust_op/` 目录 | RecSDK已有能力，**独立代码仓是暴露方式** | 算子性能对标NVIDIA CUTLASS |
| **Ascend-RecFlow** | RecSDK TorchRec-V1/V2 | 复用RecSDK训练框架，**强化适配层** | 无缝接入客户PyTorch训练代码 |
| **Ascend-RecBench** | `develop_torch_benchmark/` 分支 | 独立代码仓 + 场景扩充 + 可视化增强 | 客户自助评估 + 销售武器 |
| **Ascend-RecServe** | torch_npu AOTI + vLLM-ascend | **重点建设方向**：推理框架整合 + 生成式推理 | 推理性能对标NVIDIA T4 |
| **Ascend-RecToolkit** | torch_npu自动迁移 + 工具迁移 | 在官方工具基础上**强化推荐场景专用工具** | 客户迁移效率提升 |

---

### 项目一：Ascend-RecBench（昇腾推荐Benchmark平台）【P1强化】

**定位**：标准化性能对比平台，让客户能直观看到昇腾vs GPU的性能差异。

**昇腾现有资产**：`RecSDK` 的 `develop_torch_benchmark/` 分支已包含：
- 40+ 模型的Benchmark配置（CTR模型、DIN/DIEN/DeepFM/DCNv2等）
- 生成式推荐模型（HSTU_META_1M/20M/LARGE、RANKMAXER、RECSYS_RANKING_GR_7B等）
- 标准运行脚本 `run.py` + 对标GPU的 `compare_output.py`
- patch机制支持客户适配

**改进方向**：

| 方向 | 现状 | 改进目标 |
|------|------|---------|
| **独立代码仓** | 在RecSDK子目录 | 拆分为独立代码仓，快速迭代 |
| **模型覆盖** | 40+模型 | 补充字节TokenMixer-Large、快手OneRec、美团EGA-V2等前沿模型 |
| **数据集** | Movielen-20m等标准数据集 | 支持客户私有数据脱敏导入，客户用自己的数据做评估 |
| **可视化报告** | 手动运行对比脚本 | 自动生成HTML对比报告（MFU/QPS/延迟/内存/成本） |
| **自动化评估** | 手动运行 | CI/CD自动化，客户提交PR即可触发完整评估 |
| **与客户系统对接** | 无 | 支持导出为客户中台所需的指标格式 |

**代码仓设计**：
```
Ascend-RecBench/
├── models/                  # 标准模型集（扩充至60+）
│   ├── ctr/                # 传统CTR模型
│   ├── hstu/               # Meta HSTU/ULTRA-HSTU
│   ├── rank_mixer/         # 字节TokenMixer-Large
│   ├── onerec/             # 快手OneRec
│   └── llm_rec/           # LLM推荐（HLLM/LUM）
├── datasets/               # 标准测试数据集
│   ├── public/            # 公开数据集（Movielen、Amazon等）
│   └── private/           # 客户私有数据导入（脱敏处理）
├── runners/
│   ├── gpu_runner.py       # GPU基准运行器
│   ├── npu_runner.py       # 昇腾基准运行器
│   └── compare_runner.py   # 对比分析器
├── reports/
│   ├── generator.py        # 自动生成HTML对比报告
│   └── template/           # 报告模板
├── integration/
│   ├── ci_plugin.py        # GitHub/GitLab CI插件
│   └── webhook.py          # 触发客户评估流程
└── README.md
```

**使用方式**：
```bash
# 客户运行自己的昇腾基准测试
python run_benchmark.py \
    --model token_mixer_large \
    --dataset customer_private \
    --hardware ascend_910b \
    --compare gpu_a100

# 自动生成对比报告（昇腾 vs A100）
# 包含：MFU、延迟、吞吐量、内存占用、成本对比
```

**关键指标**：
- 支持60+模型，覆盖主流推荐场景
- 报告生成时间<5分钟
- 支持客户私有数据脱敏导入

---

### 项目二：Ascend-RecServe（昇腾推荐推理Serving）【P0强化 - 重点建设】

**核心问题**：互联网客户都有自己的推理框架（TensorRT/ONNX/Triton/自研），RecServe的生态位是什么？

---

#### 2.1 推理框架适配的核心挑战

互联网客户（中台推荐团队）的推理架构呈现高度碎片化：

| 客户 | 主流推理框架 | 推理优化重点 |
|------|------------|------------|
| **字节** | 内部自研推理引擎 + TensorRT | 超低延迟、Batch调度、特征拉取融合 |
| **快手** | TensorRT + 内部优化框架 | 短视频推荐延迟敏感 |
| **腾讯** | TorchServe + TensorRT | 广告/推荐混合部署 |
| **阿里** | OFS(OpenFaaS) + MNN + TensorRT | 电商大促峰值支撑 |
| **美团** | 自研推理框架 + TRITON | 外卖/酒旅多业务复用 |
| **小红书** | ONNX Runtime + TensorRT | 内容推荐场景 |

**客户的根本诉求**：
1. 不愿意改线上推理代码（风险大）
2. 不愿意换掉已有的推理框架（团队已投入大量优化）
3. 只接受"换硬件后端，不换软件框架"

---

#### 2.2 RecServe的生态位：Backend Adapter（推理后端适配层）

**定位**：RecServe不是另一个推理框架，而是**推理框架和昇腾硬件之间的适配层**。

```
┌─────────────────────────────────────────────────────┐
│                 客户推理框架（不改代码）               │
│  TensorRT │ ONNX Runtime │ TRITON │ 自研框架          │
└─────────────────┬───────────────────────────────────┘
                  │ ← 标准化接口（ONNX/TorchScript/...）
                  ▼
┌─────────────────────────────────────────────────────┐
│            Ascend-RecServe Backend Adapter          │
│  ┌─────────────────────────────────────────────┐   │
│  │  Backend抽象层（Core）                       │   │
│  │  ├── ONNX Backend → 昇腾ONNX Runtime        │   │
│  │  ├── TensorRT Backend → 昇腾TensorRT        │   │
│  │  ├── TRITON Backend → 昇腾TRITON Server     │   │
│  │  └── TorchScript Backend → torch_npu AOTI   │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │  推荐场景增强层                              │   │
│  │  ├── 动态Batch调度（推荐变长序列）            │   │
│  │  ├── KV Cache用户侧复用（生成式推荐）         │   │
│  │  ├── 特征拉取融合（Feature Store对接）        │   │
│  │  └── 多目标推理（CTR/CVR/互动联合推理）       │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│                 昇腾NPU硬件层                        │
│  CANN │ torch_npu │ RecOps算子库                    │
└─────────────────────────────────────────────────────┘
```

**核心价值**：
- 客户**不改推理代码**，只需配置"backend=ascend"
- 昇腾作为**透明硬件后端**替换GPU
- 推荐场景优化由RecServe统一提供，不需要客户二次开发

---

#### 2.3 四种Backend Adapter设计

**Backend 1：ONNX Runtime → 昇腾**

```python
# 客户代码（原有）
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])
result = sess.run(None, {"input": x})

# 迁移到昇腾（改1行）
sess = ort.InferenceSession("model.onnx", providers=['AscendExecutionProvider'])
# 其他代码完全不变，自动切换到昇腾
```

**Backend 2：TensorRT → 昇腾**

```python
# 客户原有TensorRT代码
import tensorrt as trt
# ... TRTexecutor setup ...

# 昇腾提供等价的TRT backend
import ascend_recserve.backends.tensorrt as ascend_trt
executor = ascend_trt.Executor(plan_file="model.plan")
result = executor.execute({"input": x})
```

**Backend 3：TRITON Server → 昇腾**

```python
# 客户原有TRITON配置（triton.models.pbtxt）
# name: "recommendation_model"
# platform: "pytorch_torchscript"

# 昇腾方案：提供TRITON的昇腾backend插件
# 客户无需改模型配置，只需改backend
# backend: "ascend"  # 替换"tensorrt"或"pytorch"
```

**Backend 4：TorchScript → 昇腾AOTI**

```python
# 客户原有TorchScript代码
model = torch.jit.load("model.pt")
result = model(input)

# 昇腾AOTI编译（性能最优）
import torch_npu.aoti
compiled = torch_npu.aoti.compile(model, input_signature=(input,))
result = compiled(input)
```

---

#### 2.4 推荐场景增强层（RecServe差异化竞争力）

在Backend Adapter之上，提供推荐场景特有的优化：

```python
from ascend_recserve.enhanced import DynamicBatcher, KVCacheManager, FeatureFetcher

# 1. 动态Batch调度（推荐场景变长序列）
batcher = DynamicBatcher(
    max_batch_size=256,
    max_wait_ms=5,  # 5ms凑batch，超时即发送
    padding_style="jagged"  # 推荐场景用jagged tensor
)

# 2. KV Cache用户侧复用（生成式推荐）
cache_mgr = KVCacheManager(
    max_cache_size="100GB",
    eviction_policy="lru"
)
user_embeddings = cache_mgr.get(user_id)

# 3. 特征存储对接（客户已有Feature Store）
feature_fetcher = FeatureFetcher(
    store="redis",  # 对接客户Redis/Feast
    lookup_timeout_ms=2
)
features = feature_fetcher.get(user_id, feature_names)

# 4. 多目标联合推理
multi_task_output = model.forward_multi_target(
    user_seq, item_candidates,
    tasks=["ctr", "cvr", "互动率", "停留时长"]
)
```

---

#### 2.5 与客户框架解耦的关键设计原则

| 原则 | 实现方式 |
|------|---------|
| **接口标准化** | 支持ONNX/TorchScript作为统一中间表示，客户模型无需改格式 |
| **后端可插拔** | Backend Adapter是插件，客户可以同时保留GPU和昇腾后端 |
| **配置驱动** | 客户通过配置文件切换后端，不需要改代码 |
| **版本隔离** | 昇腾后端与客户原有后端版本独立，互不影响 |
| **灰度切换** | 支持5%/10%/50%/100%流量逐步切换到昇腾 |

---

#### 2.6 快速迁移适配方案

针对不同客户的框架，提供差异化的迁移路径：

| 客户框架 | 迁移路径 | 客户改动量 | 迁移周期 |
|---------|---------|-----------|---------|
| **TensorRT** | 提供Ascend-TensorRT Backend | **0行代码**，改config | 1天 |
| **ONNX Runtime** | 提供AscendExecutionProvider | **改1行** providers | 1天 |
| **TRITON Server** | 提供昇腾TRITON Backend | **改1行** backend config | 1天 |
| **自研框架** | ONNX导出 + Ascend ONNX Backend | 客户模型导出ONNX | 3-7天 |
| **TorchScript** | torch_npu AOTI编译 | **改1行** compile | 1天 |

---

#### 2.7 现有资产整合

| 资产 | 说明 | 如何整合 |
|------|------|---------|
| **torch_npu AOTI** | 官方推理编译能力 | 作为Backend Adapter的编译后端 |
| **vLLM-ascend** | 社区维护的vLLM昇腾适配 | 整合为vLLM Backend |
| **RecOps算子** | 推荐专用算子 | 作为CANN上的算子实现，供各Backend调用 |
| **CANN** | 昇腾异构计算架构 | 各Backend的底层执行引擎 |

**代码仓设计**：
```
Ascend-RecServe/
├── core/                    # 核心抽象层
│   ├── backend.py           # Backend基类定义
│   ├── session.py           # 统一会话管理
│   └── config.py            # 配置驱动
├── backends/               # 推理后端（可插拔）
│   ├── onnx/               # ONNX Runtime → 昇腾
│   │   ├── provider.py     # AscendExecutionProvider
│   │   └── compile.py      # ONNX优化
│   ├── tensorrt/           # TensorRT → 昇腾
│   │   ├── executor.py     # TRT兼容接口
│   │   └── plugin.py        # TRT插件
│   ├── triton/             # TRITON Server → 昇腾
│   │   ├── backend.py      # TRITON Backend实现
│   │   └── model_repository # 标准Model Repo
│   ├── torchscript/        # TorchScript → AOTI
│   │   ├── aoti_compiler.py
│   │   └── runtime.py
│   └── vllm/               # vLLM → 昇腾
│       └── npu_backend.py
├── enhanced/               # 推荐场景增强层
│   ├── dynamic_batcher.py   # 动态Batch调度
│   ├── kv_cache.py         # KV Cache管理
│   ├── feature_fetcher.py   # 特征存储对接
│   ├── multi_task.py       # 多目标推理
│   └── beam_search.py       # Beam Search生成式推荐
├── integration/            # 客户系统对接
│   ├── kubernetes.py       # K8s部署插件
│   ├── prometheus.py       # 监控指标
│   └── feature_store.py     # Redis/Feast对接
└── README.md
```

---

#### 2.8 优先级规划

| 阶段 | 时间 | 建设内容 | 目标 |
|------|------|---------|------|
| **Phase 1** | 2026 Q3 | ONNX Backend + TorchScript Backend | 覆盖80%客户的框架 |
| **Phase 2** | 2026 Q4 | TRITON Backend + TensorRT Backend | 覆盖头条/快手/腾讯 |
| **Phase 3** | 2027 Q1 | vLLM Backend + 动态Batch + KVCache | 生成式推荐推理支持 |
| **Phase 4** | 2027 Q2 | 多目标推理 + Feature Fetcher | 完整推荐场景覆盖 |

---

### 项目三：Ascend-RecToolkit### 项目三：Ascend-RecToolkit（昇腾推荐工具包）【P1 - 推荐场景增强】

**定位**：在torch_npu官方迁移工具基础上，构建推荐场景专用的增强工具集。

**昇腾现有资产**：
- **自动迁移**（推荐）：`import torch_npu` 运行时自动替换CUDA→NPU
- **工具迁移**：PyTorch GPU2Ascend工具，生成迁移报告
- **手工迁移**：官方文档有详细的手工迁移指南

**推荐场景增强方向**：

| 工具 | 昇腾官方能力 | 昇腾推荐团队增强 |
|------|------------|----------------|
| **迁移评估** | 通用CUDA代码分析 | 推荐模型专用算子依赖分析（FBGEMM/HKV/HSTU） |
| **自动迁移** | `torch.cuda` → `torch.npu` | 推荐模型专用迁移patch（DIN/DIEN/HSTU等模板） |
| **精度调优** | 通用FP16/BF16调优 | 推荐模型loss spike检测（embedding lookup不稳定） |
| **性能分析** | MindStudio Profiler | 推荐模型专用瓶颈识别（embedding占比高、attention变体） |
| **模型转换** | 通用ONNX/TorchScript | 生成式推荐专用converter（HSTU→推理模型） |

**代码仓设计**：
```
Ascend-RecToolkit/
├── migrate/                # 迁移工具
│   ├── rec_analyzer.py     # 推荐模型代码分析（识别FBGEMM/HSTU算子）
│   ├── patch_generator.py  # 生成推荐模型专用迁移patch
│   ├── risk_detector.py    # 识别迁移风险点（推荐专用）
│   └── diff_viewer.py      # 生成迁移差异报告
├── profile/                # 性能分析
│   ├── rec_profiler.py     # 推荐模型专用Profiler封装
│   ├── bottleneck.py       # 自动识别瓶颈（embedding/attention/hstu）
│   └── optimize_hints.py  # 生成推荐专用优化建议
├── tune/                   # 精度调优
│   ├── loss_analyzer.py    # 推荐模型loss分析（检测embedding不稳定）
│   ├── numerical_check.py # 数值精度检查
│   └── auto_tune.py       # 推荐模型超参自动调优
├── deploy/                # 部署工具
│   ├── model_converter.py  # 训练→推理模型转换（支持HSTU/OneRec）
│   ├── batch_config.py    # Batch大小自动配置
│   └── health_check.py    # 生产环境健康检查
└── README.md
```

**核心价值**：
- **迁移评估**：5分钟给出推荐代码的迁移难度评估（自动识别HSTU/FBGEMM/HKV依赖）
- **推荐专用patch**：DIN/DIEN/HSTU等模型的迁移patch，开箱即用
- **一键优化**：根据Profiler结果自动生成推荐模型优化建议

---

### 项目四：Ascend-RecOps & Ascend-RecFlow（建议维持现状）

**Ascend-RecOps**：RecSDK已有 `cust_op/ascendc_op/` 算子库（30+算子），**无需重建代码仓**。
- 建议：作为RecSDK的子模块持续完善，突出"昇腾推荐算子"的品牌
- 竞争力：算子性能对标NVIDIA CUTLASS

**Ascend-RecFlow**：RecSDK已有 TorchRec-V1/V2 训练框架，**无需重建代码仓**。
- 建议：强化RecSDK的适配层，让客户DDP/FSDP代码能无缝切换到昇腾
- 竞争力：训练代码零改动

---

### 竞争力项目优先级矩阵（修订版）

| 项目 | 定位 | 客户价值 | 技术壁垒 | 开发周期 | 优先级 |
|------|------|---------|---------|---------|--------|
| **Ascend-RecServe** | 推理性能对标GPU | 推理延迟降低20%+ | vLLM-ascend持续跟进、AOTI编译、Batch调度 | 6个月 | **P0** |
| **Ascend-RecBench** | 客户决策支撑 | 降低采购不确定性，销售武器 | 模型覆盖、报告可视化 | 3个月 | **P1** |
| **Ascend-RecToolkit** | 迁移效率提升 | 迁移效率提升50% | 推荐模型专用分析/优化 | 4个月 | **P1** |
| **Ascend-RecOps** | 维持现状 | 已有资产品牌化 | 已有能力完善 | 持续 | 维持 |
| **Ascend-RecFlow** | 维持现状 | 已有资产品牌化 | 已有能力完善 | 持续 | 维持 |

### 三种无缝接入模式（修订版）

| 模式 | 适用场景 | 客户改动量 | 实现方式 |
|------|---------|-----------|---------|
| **插件模式** | vLLM推理（推荐生产环境） | **0行代码** | `from vllm_ascend import LLM`，自动用NPU |
| **适配器模式** | 客户有自定义训练代码 | **改1-2行** | import切换+配置参数（torch_npu） |
| **工具模式** | 客户需要迁移评估 | **运行工具** | Ascend-RecToolkit分析CUDA代码 |

---

## 五、技术规划总## 四-补充2：竞争力项目建议（基于昇腾现有资产的差异化定位）

> **说明**：基于对昇腾现有代码资产的调研（RecSDK、torch_npu、vLLM-ascend），重新校准各项目的定位。
> 核心原则：**不重复造轮子，基于昇腾已有能力构建差异化竞争力**。

---

### 项目定位校准总结

| 拟建项目 | 昇腾现有资产 | 重新定位 | 竞争力来源 |
|---------|-------------|---------|-----------|
| **Ascend-RecOps** | RecSDK `cust_op/` 目录 | RecSDK已有能力，**独立代码仓是暴露方式** | 算子性能对标NVIDIA CUTLASS |
| **Ascend-RecFlow** | RecSDK TorchRec-V1/V2 | 复用RecSDK训练框架，**强化适配层** | 无缝接入客户PyTorch训练代码 |
| **Ascend-RecBench** | `develop_torch_benchmark/` 分支 | 独立代码仓 + 场景扩充 + 可视化增强 | 客户自助评估 + 销售武器 |
| **Ascend-RecServe** | torch_npu AOTI + vLLM-ascend | **重点建设方向**：推理框架整合 + 生成式推理 | 推理性能对标NVIDIA T4 |
| **Ascend-RecToolkit** | torch_npu自动迁移 + 工具迁移 | 在官方工具基础上**强化推荐场景专用工具** | 客户迁移效率提升 |

---

### 项目一：Ascend-RecBench（昇腾推荐Benchmark平台）【P1强化】

**定位**：标准化性能对比平台，让客户能直观看到昇腾vs GPU的性能差异。

**昇腾现有资产**：`RecSDK` 的 `develop_torch_benchmark/` 分支已包含：
- 40+ 模型的Benchmark配置（CTR模型、DIN/DIEN/DeepFM/DCNv2等）
- 生成式推荐模型（HSTU_META_1M/20M/LARGE、RANKMAXER、RECSYS_RANKING_GR_7B等）
- 标准运行脚本 `run.py` + 对标GPU的 `compare_output.py`
- patch机制支持客户适配

**改进方向**：

| 方向 | 现状 | 改进目标 |
|------|------|---------|
| **独立代码仓** | 在RecSDK子目录 | 拆分为独立代码仓，快速迭代 |
| **模型覆盖** | 40+模型 | 补充字节TokenMixer-Large、快手OneRec、美团EGA-V2等前沿模型 |
| **数据集** | Movielen-20m等标准数据集 | 支持客户私有数据脱敏导入，客户用自己的数据做评估 |
| **可视化报告** | 手动运行对比脚本 | 自动生成HTML对比报告（MFU/QPS/延迟/内存/成本） |
| **自动化评估** | 手动运行 | CI/CD自动化，客户提交PR即可触发完整评估 |
| **与客户系统对接** | 无 | 支持导出为客户中台所需的指标格式 |

**代码仓设计**：
```
Ascend-RecBench/
├── models/                  # 标准模型集（扩充至60+）
│   ├── ctr/                # 传统CTR模型
│   ├── hstu/               # Meta HSTU/ULTRA-HSTU
│   ├── rank_mixer/         # 字节TokenMixer-Large
│   ├── onerec/             # 快手OneRec
│   └── llm_rec/           # LLM推荐（HLLM/LUM）
├── datasets/               # 标准测试数据集
│   ├── public/            # 公开数据集（Movielen、Amazon等）
│   └── private/           # 客户私有数据导入（脱敏处理）
├── runners/
│   ├── gpu_runner.py       # GPU基准运行器
│   ├── npu_runner.py       # 昇腾基准运行器
│   └── compare_runner.py   # 对比分析器
├── reports/
│   ├── generator.py        # 自动生成HTML对比报告
│   └── template/           # 报告模板
├── integration/
│   ├── ci_plugin.py        # GitHub/GitLab CI插件
│   └── webhook.py          # 触发客户评估流程
└── README.md
```

**使用方式**：
```bash
# 客户运行自己的昇腾基准测试
python run_benchmark.py \
    --model token_mixer_large \
    --dataset customer_private \
    --hardware ascend_910b \
    --compare gpu_a100

# 自动生成对比报告（昇腾 vs A100）
# 包含：MFU、延迟、吞吐量、内存占用、成本对比
```

**关键指标**：
- 支持60+模型，覆盖主流推荐场景
- 报告生成时间<5分钟
- 支持客户私有数据脱敏导入

---

### 项目二：Ascend-RecServe（昇腾推荐推理Serving）【P0强化 - 重点建设】

**定位**：高性能推理Serving框架，整合torch_npu AOTI和vLLM-ascend，形成统一推理入口。

**昇腾现有资产**：

| 资产 | 说明 | 能力现状 |
|------|------|---------|
| **torch_npu AOTI** | `torch_npu` 中的AOT编译推理 | 基础可用，文档较少 |
| **vLLM-ascend** | https://github.com/vllm-project/vllm-ascend | 2.2k stars，NVIDIA vLLM的昇腾适配，活跃度高 |
| **Recsys-GR** | 昇腾官方的生成式推荐样例 | 覆盖HSTU训练和推理 |

**竞争格局**：
- NVIDIA：vLLM（事实标准）+ TensorRT-LLM
- 昇腾：vLLM-ascend（社区维护）+ torch_npu AOTI（官方）
- **差距**：昇腾推理生态明显落后于NVIDIA，需要重点投入

**代码仓设计**：
```
Ascend-RecServe/
├── vllm_plugin/            # vLLM-ascend整合
│   ├── npu_backend.py       # 昇腾NPU后端实现
│   ├── paged_attention.py   # PagedAttention昇腾实现
│   ├── tensor_parallel.py   # 昇腾张量并行
│   └── integration.py       # vLLM API封装
├── aoti_compiler/          # torch_npu AOTI编译
│   ├── model_compiler.py    # AOTI模型编译
│   ├── quantize.py          # BF16/FP8/INT4量化
│   └── optimize.py          # 计算图优化
├── serving/
│   ├── native_server.py     # 昇腾原生高性能Serving
│   ├── batch_scheduler.py   # 动态Batch调度
│   ├── kv_cache.py         # KV Cache管理
│   └── beam_search.py       # 生成式推荐beam search
├── backends/
│   ├── npu_backend.py       # 昇腾NPU推理后端
│   ├── memory_optimize.py   # 内存优化
│   └── graph_optimize.py    # 算子融合
├── integration/
│   ├── kubernetes_plugin.py # K8s部署插件
│   ├── prometheus_metrics.py# 监控指标对接
│   ├── feature_store.py     # 特征存储对接（Redis/Feast）
│   └── triton_adapter.py    # Triton协议适配（客户原有系统兼容）
└── README.md
```

**接入方式**：

```python
# 方式1：vLLM插件（推荐，推荐系统生产环境）
from vllm_ascend import LLM
llm = LLM(model="model.pt", tensor_parallel_size=4)
# vLLM-ascend已支持昇腾，客户无需改代码

# 方式2：torch_npu AOTI编译（高性能场景）
from torch_npu.contrib import torch_npu_aoti
compiled_model = torch_npu_aoti.compile(model, ...)
output = compiled_model(input)

# 方式3：昇腾原生Serving（统一入口）
from ascend_recserve import Server
server = Server(model="model.pt", backend="vllm" or "aoti")
server.start(port=8080)
```

**生成式推荐推理增强**（差异化重点）：
```python
# 生成式推荐的KV Cache复用（用户侧复用，降低推理成本）
from ascend_recserve import KVCacheManager
cache_mgr = KVCacheManager(max_cache_size="100GB")
user_history = cache_mgr.get_user_history(user_id)
response = llm.generate(query, past_tokens=user_history)

# Beam Search for 生成式推荐
from ascend_recserve import BeamSearchDecoder
decoder = BeamSearchDecoder(model, beam_width=4)
candidates = decoder.decode(query, user_context)
```

**性能目标**：
- vLLM-ascend：持续跟进NVIDIA vLLM最新版本（每月同步）
- AOTI推理：延迟对标T4（INT8量化）
- 动态Batch：吞吐量最大化NPU利用率
- 生成式推理：支持HSTU/OneRec的自回归解码

**关键差异化**：
1. **统一入口**：整合vLLM-ascend + torch_npu AOTI，客户按场景选择
2. **生成式推理增强**：KV Cache复用、Beam Search是推荐场景特有需求
3. **客户系统兼容**：Triton协议适配，让客户原有TF-Serving/Triton系统无需修改即可切换到昇腾

---

### 项目三：Ascend-RecToolkit（昇腾推荐工具包）【P1 - 推荐场景增强】

**定位**：在torch_npu官方迁移工具基础上，构建推荐场景专用的增强工具集。

**昇腾现有资产**：
- **自动迁移**（推荐）：`import torch_npu` 运行时自动替换CUDA→NPU
- **工具迁移**：PyTorch GPU2Ascend工具，生成迁移报告
- **手工迁移**：官方文档有详细的手工迁移指南

**推荐场景增强方向**：

| 工具 | 昇腾官方能力 | 昇腾推荐团队增强 |
|------|------------|----------------|
| **迁移评估** | 通用CUDA代码分析 | 推荐模型专用算子依赖分析（FBGEMM/HKV/HSTU） |
| **自动迁移** | `torch.cuda` → `torch.npu` | 推荐模型专用迁移patch（DIN/DIEN/HSTU等模板） |
| **精度调优** | 通用FP16/BF16调优 | 推荐模型loss spike检测（embedding lookup不稳定） |
| **性能分析** | MindStudio Profiler | 推荐模型专用瓶颈识别（embedding占比高、attention变体） |
| **模型转换** | 通用ONNX/TorchScript | 生成式推荐专用converter（HSTU→推理模型） |

**代码仓设计**：
```
Ascend-RecToolkit/
├── migrate/                # 迁移工具
│   ├── rec_analyzer.py     # 推荐模型代码分析（识别FBGEMM/HSTU算子）
│   ├── patch_generator.py  # 生成推荐模型专用迁移patch
│   ├── risk_detector.py    # 识别迁移风险点（推荐专用）
│   └── diff_viewer.py      # 生成迁移差异报告
├── profile/                # 性能分析
│   ├── rec_profiler.py     # 推荐模型专用Profiler封装
│   ├── bottleneck.py       # 自动识别瓶颈（embedding/attention/hstu）
│   └── optimize_hints.py  # 生成推荐专用优化建议
├── tune/                   # 精度调优
│   ├── loss_analyzer.py    # 推荐模型loss分析（检测embedding不稳定）
│   ├── numerical_check.py # 数值精度检查
│   └── auto_tune.py       # 推荐模型超参自动调优
├── deploy/                # 部署工具
│   ├── model_converter.py  # 训练→推理模型转换（支持HSTU/OneRec）
│   ├── batch_config.py    # Batch大小自动配置
│   └── health_check.py    # 生产环境健康检查
└── README.md
```

**核心价值**：
- **迁移评估**：5分钟给出推荐代码的迁移难度评估（自动识别HSTU/FBGEMM/HKV依赖）
- **推荐专用patch**：DIN/DIEN/HSTU等模型的迁移patch，开箱即用
- **一键优化**：根据Profiler结果自动生成推荐模型优化建议

---

### 项目四：Ascend-RecOps & Ascend-RecFlow（建议维持现状）

**Ascend-RecOps**：RecSDK已有 `cust_op/ascendc_op/` 算子库（30+算子），**无需重建代码仓**。
- 建议：作为RecSDK的子模块持续完善，突出"昇腾推荐算子"的品牌
- 竞争力：算子性能对标NVIDIA CUTLASS

**Ascend-RecFlow**：RecSDK已有 TorchRec-V1/V2 训练框架，**无需重建代码仓**。
- 建议：强化RecSDK的适配层，让客户DDP/FSDP代码能无缝切换到昇腾
- 竞争力：训练代码零改动

---

### 竞争力项目优先级矩阵（修订版）

| 项目 | 定位 | 客户价值 | 技术壁垒 | 开发周期 | 优先级 |
|------|------|---------|---------|---------|--------|
| **Ascend-RecServe** | 推理性能对标GPU | 推理延迟降低20%+ | vLLM-ascend持续跟进、AOTI编译、Batch调度 | 6个月 | **P0** |
| **Ascend-RecBench** | 客户决策支撑 | 降低采购不确定性，销售武器 | 模型覆盖、报告可视化 | 3个月 | **P1** |
| **Ascend-RecToolkit** | 迁移效率提升 | 迁移效率提升50% | 推荐模型专用分析/优化 | 4个月 | **P1** |
| **Ascend-RecOps** | 维持现状 | 已有资产品牌化 | 已有能力完善 | 持续 | 维持 |
| **Ascend-RecFlow** | 维持现状 | 已有资产品牌化 | 已有能力完善 | 持续 | 维持 |

### 三种无缝接入模式（修订版）

| 模式 | 适用场景 | 客户改动量 | 实现方式 |
|------|---------|-----------|---------|
| **插件模式** | vLLM推理（推荐生产环境） | **0行代码** | `from vllm_ascend import LLM`，自动用NPU |
| **适配器模式** | 客户有自定义训练代码 | **改1-2行** | import切换+配置参数（torch_npu） |
| **工具模式** | 客户需要迁移评估 | **运行工具** | Ascend-RecToolkit分析CUDA代码 |

---

## 五、技术规划总览## 五、技术规划总览