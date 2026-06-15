# (快手,生成式推荐,GR) OpenOneRec论文解读
> 发布时间: 编辑于 2026-01-02 14:59・浙江
> 原文链接: https://zhuanlan.zhihu.com/p/1990434758312863235

---

​

目录

[](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2512.24762)

* * *

## **相关资料**

-   **研究机构**：

-   **快手 OneRec 团队（Kuaishou OneRec Team）**

-   该团队长期致力于推荐系统与大模型的融合研究，此前已发布 OneRec 系列模型（如 OneRec、OneRec-v2），推动了[生成式推荐](https://zhida.zhihu.com/search?content_id=268479295&content_type=Article&match_order=1&q=%E7%94%9F%E6%88%90%E5%BC%8F%E6%8E%A8%E8%8D%90&zhida_source=entity)（Generative Recommendation）的发展。
-   本工作由 Guorui Zhou、Honghui Bao、Jiaming Huang 等人主导，作者列表中所有成员均来自快手。

-   **开源资源**：

-   **GitHub 仓库**：[https://github.com/Kuaishou-OneRec/OpenOneRec](https://link.zhihu.com/?target=https%3A//github.com/Kuaishou-OneRec/OpenOneRec)
-   **Hugging Face 模型发布页**：[https://huggingface.co/OpenOneRec](https://link.zhihu.com/?target=https%3A//huggingface.co/OpenOneRec)
-   **完整训练框架、数据处理脚本、评估代码均开源**，支持端到端复现。

-   **模型发布**：

-   发布 **OneRec-Foundation 系列模型**，包含 **1.7B 和 8B 参数版本**。
-   提供 Standard 与 Pro 两种变体：

-   **Standard**：基于公开数据集训练，确保可复现性。
-   **Pro**：融合快手百亿 token 工业级数据，性能更强。

-   **数据发布**：

-   公开 **9600 万条用户交互记录**，来自 **16 万名匿名用户**。
-   包含短视频、电商、广告等多模态行为，涵盖观看、点赞、购买、搜索等多行为类型。

-   **基准测试集**：

-   提出 **RecIF-Bench**，首个面向生成式推荐的综合性指令跟随基准。
-   支持从基础推荐到复杂推理的 8 项任务，覆盖 4 层能力层级。

-   **背景与动机**：

-   当前推荐系统多为领域专家（domain specialists），缺乏通用智能，受限于孤立数据，难以实现跨任务泛化。
-   大语言模型（LLMs）虽具备强大推理与生成能力，但缺乏对用户行为与物品语义的深度建模。
-   本文旨在弥合“推荐系统”与“通用智能”之间的鸿沟，构建**真正智能的推荐基础模型**。

* * *

## **主要贡献**

### ✅ **1\. RecIF-Bench：首个综合性推荐指令跟随基准**

-   提出 **RecIF-Bench**，系统性组织 **8 个多样化任务**，覆盖从语义对齐、基础推荐、指令跟随到复杂推理的四层能力体系。
-   首次支持 **交错数据输入**（Interleaved Data，文本与 itemic token 混合）、**跨域推荐**、**自然语言解释生成** 等关键能力评估。
-   弥补现有基准（如 PixelRec、NineRec、KuaiSAR）在多任务、多模态、推理能力上的不足。

**对比优势**：

-   ✅ 支持 Multi-Behavioral、Interleaved Data、Recommendation Explanation
-   ❌ 现有基准均未同时支持上述三项

![](https://pica.zhimg.com/v2-e555edd91b3756c3534b2b172c7a67b6_1440w.jpg)

### ✅ **2\. OneRec-Foundation：开放的基础模型家族**

-   发布 **OneRec-Foundation（1.7B & 8B）**，基于 [Qwen3](https://zhida.zhihu.com/search?content_id=268479295&content_type=Article&match_order=1&q=Qwen3&zhida_source=entity) 架构，首次实现推荐能力与通用智能的统一。
-   在 RecIF-Bench 上实现 **所有任务 SOTA**，并在 Amazon 十大领域平均 Recall@10 提升 **26.8%**。
-   Pro 版本进一步融合工业数据，显著增强性能。

### ✅ **3\. 可复现的端到端训练框架**

-   开源完整训练流程，包括：

-   数据预处理
-   两阶段预训练（Itemic-Text Alignment + Full-Parameter Co-Pretraining）
-   三阶段后训练（Multi-task SFT、On-policy Distillation、Rec-RL）

-   提供详细的数据混合策略、token 预算、学习率设置，确保完全可复现。

### ✅ **4\. 验证推荐能力的可扩展性（Scaling Laws）**

-   实验验证推荐能力随模型规模、数据规模**可预测地提升**。
-   提出数据混合策略以缓解灾难性遗忘，实现推荐与通用能力的平衡。

* * *

## **具体做法**

### **1\. 任务定义与统一建模**

### ✅ 统一任务公式化

所有任务统一建模为序列生成问题： Y = F(X), \\quad \\text{其中 } X = \[I, C\]

-   I ：任务特定指令
-   C ：用户上下文，包含：

-   交互历史 H^u = \\{s\_{i\_1}, s\_{i\_2}, ..., s\_{i\_t}\\} （itemic token 序列）
-   用户画像 P^u ：自然语言与 itemic token 交错的叙事式描述

### ✅ 四层能力分类（Task Taxonomy）

<table data-draft-node="block" data-draft-type="table" data-size="normal" data-row-style="normal"><tbody><tr><td>层级</td><td>任务</td><td>目标</td></tr><tr><td>Layer 0: 语义对齐</td><td>Item Understanding</td><td>生成物品的文本描述，验证 itemic token 与语言对齐</td></tr><tr><td>Layer 1: 基础推荐</td><td>Short Video Rec, Ad/Product Rec, Label Prediction</td><td>预测下一个物品或行为</td></tr><tr><td>Layer 2: 指令跟随</td><td>Interactive Rec, Label-Conditional Rec</td><td>根据自然语言意图或行为标签推荐</td></tr><tr><td>Layer 3: 推理</td><td>Recommendation Explanation</td><td>生成为何推荐某物品的自然语言解释</td></tr></tbody></table>

![](https://pic2.zhimg.com/v2-0c64fed482fdee02585ae105d9c39619_1440w.jpg)

![](https://pic4.zhimg.com/v2-8157dab8a7a707aeebe1d6b1d20fabf9_1440w.jpg)

### **2\. 数据构建与预训练**

### ✅ Itemic Token 构建

-   使用 **RQ-Kmeans** 对物品多模态嵌入进行三级量化，每层码本大小 8192。
-   每个物品映射为三元组(c\_1, c\_2, c\_3)，展平为 token 序列：<|item\_begin|><item\_a\_5028><item\_b\_6733><item\_c\_2559><|item\_end|>

### ✅ 推荐领域数据构建

<table data-draft-node="block" data-draft-type="table" data-size="normal" data-row-style="normal"><tbody><tr><td>数据类型</td><td>任务</td><td>示例</td></tr><tr><td>Itemic Dense Caption Data</td><td>语义对齐</td><td>输入 itemic token，输出文本描述</td></tr><tr><td>Sequential User Behavior Data</td><td>基础推荐</td><td>输入行为序列，预测下一物品</td></tr><tr><td>Interleaved User Persona Grounding Data</td><td>深度用户理解</td><td>交错式用户画像，融合行为、属性、兴趣</td></tr></tbody></table>

![](https://pic2.zhimg.com/v2-618e928ffab3153afacc81d5502ceee7_1440w.jpg)

![](https://pic1.zhimg.com/v2-d878e10d00b6675a5748aca7aba47292_1440w.jpg)

![](https://pic1.zhimg.com/v2-eab6bac113c32058332eb728b4932f76_1440w.jpg)

### ✅ 两阶段预训练策略

![](https://pica.zhimg.com/v2-4de9b1e6b87d923719b85e115267d374_1440w.jpg)

<table data-draft-node="block" data-draft-type="table" data-size="normal" data-row-style="normal"><tbody><tr><td>阶段</td><td>目标</td><td>训练策略</td></tr><tr><td>Stage 1: Itemic-Text Alignment</td><td>建立 itemic token 与语言空间的语义桥接</td><td>仅训练 itemic token 的 embedding 参数，其余冻结</td></tr><tr><td>Stage 2: Full-Parameter Co-Pretraining</td><td>注入推荐知识，保留通用能力</td><td>解冻全部参数，混合推荐与通用数据（33B tokens）</td></tr></tbody></table>

![](https://picx.zhimg.com/v2-f74e67b8d0b32d088360ae7f8fd5c2a9_1440w.jpg)

-   基于训练语料库的规模开发了两种模型变体：OneRec 和 OneRec-Pro。标准版本仅在公开发布的数据集上进行训练，该数据集包含 4130 万条样本，总计 330 亿个标记，从而为社区建立了一个可复现的基线。相比之下，Pro 版本利用了一个规模更大的内部语料库，用户覆盖范围更广，总计包含 1300 亿个标记和 1.791 亿条样本，以实现更强的鲁棒性。
-   在 Qwen3 中，较小的模型（例如0.6 B,1.7 B,4 B）采用参数共享的嵌入，即嵌入层与输出投影层共享参数，而较大的模型（例如 8 B 及以上）则具有独立的输出投影参数。对于较大的模型，项目标记对应的输出投影参数也可训练，以确保输出空间的正确对齐。
-   使用 AdamW 优化器，其中 β\_1 =0.9，β\_2 =0.95，权重衰减为 0.1。学习率遵循余弦退火调度，并包含线性预热阶段，其中第一阶段的峰值学习率设为 1×10 −3 ，第二阶段的峰值学习率设为 1×10−4 ，最小学习率分别设为 1×10 −4 和 2×10 −5 。预热阶段持续整个训练步数的前 10%。为了适应用户行为数据的长序列特性，将最大上下文长度设置为 32 K 个标记。

### **可拓展性**

1.  **推荐系统的扩展行为遵循可预测的幂律关系**：在推荐领域，模型损失随计算量（FLOPs）的增加呈现出平滑且凸的前沿，表明其扩展行为具有规律性，类似于自然语言领域的扩展规律。
2.  **最优计算分配偏向数据规模（数据密集型扩展）**：与通用语言模型（如Chinchilla）建议的模型参数与数据量均衡扩展不同，推荐系统更倾向于“数据密集型”扩展，即随着计算预算增加，应优先扩大训练数据量而非模型规模。
3.  **经验拟合得出具体的扩展指数**：文中拟合得到的最优模型大小和数据量与计算预算 C 的关系为：

-   最优模型大小 N\_{\\text{opt}} \\propto C^{0.44}
-   最优数据量 D\_{\\text{opt}} \\propto C^{0.56} 这进一步验证了数据扩展占比更高的趋势。
-   **较低的数据缩放指数** \\beta **导致更强的数据需求**：拟合结果显示数据缩放指数 \\beta \\approx 0.19 ，低于文本领域的典型值（约0.28），**说明数据收益衰减更快，因此需要更多数据来维持最优性能。**
-   **推荐任务的固有熵较低**：模型拟合出的不可约损失 E = 0.42 显著低于自然语言任务（约1.69），表明推荐任务本身信息熵更低、更具确定性，这也**意味着对数据多样性和质量的要求更高，以避免模型过早饱和。**

* * *

### **3\. 后训练（Post-Training）三阶段**

-   预训练过程完成后，模型已经学会将物品标记与文本标记空间对齐，并将协同过滤信号编码到预训练模型中。观察到预训练模型在遵循指令和推理能力方面出现了一定程度的退化，仍然无法胜任复杂的推荐任务。
-   后训练旨在增强预训练模型的推荐能力并恢复其通用任务能力。采用三个后训练阶段：多任务监督微调、基于策略的通用能力蒸馏，以及推荐用强化学习。

![](https://pic1.zhimg.com/v2-bea06b39964b5f5e965374f007d03222_1440w.jpg)

### ✅ **阶段一：多任务监督微调（Multi-task SFT）**

-   构建多任务 SFT 数据集，融合推荐与通用任务（数学、推理、编码）。

![](https://pica.zhimg.com/v2-e3c1b4c9fc9f1a2339df280ec43f4036_1440w.jpg)

-   使用 Qwen3 聊天模板格式化数据。
-   学习率： 5 \\times 10^{-6}

### ✅ **阶段二：面向通用能力的在线策略蒸馏（On-policy Distillation）**

-   用通用数据和通用模型来强化GR模型。

![](https://pic3.zhimg.com/v2-ceb65a9da465d43bc561abd18e69ad1e_1440w.jpg)

-   **目标**：恢复因 SFT 导致的通用能力退化。
-   **方法**：使用**原始 Qwen3**作为教师模型，通过 **Policy Gradient** 优化学生模型。
-   **损失函数**：每 token 反向 KL 散度 \\mathcal{L} = \\mathbb{E}\_{o \\sim \\pi\_\\theta} \\left\[ -\\sum\_{t=1}^T \\mathbb{D}\_{KL}\\left( \\pi\_\\theta(x\_t|x\_{<t}) \\| \\pi\_{\\text{teacher}}(x\_t|x\_{<t}) \\right) \\right\]
-   **关键技巧**：

-   通用任务 prompt 不含 itemic token，因为qwen3无法理解语义ID。
-   对 itemic token 施加惩罚（log prob 设为 -1e9），以模拟该标记的零概率，并截断该轨迹。该惩罚通过奖励裁剪机制加以稳定，在通用领域上下文中为项目化标记提供即时且强烈的负信号。
-   使用高温度采样增强探索。

-   从SFT数据集中采样了200 K个通用领域问题。为了更好地恢复原始Qwen3在思维方面的指令遵循能力，遵循Qwen3技术报告中描述的方法，随机在用户提示后附加一个后缀（/think、/no\_think或空字符串）。该策略旨在使模型的行为与强制思考、非思考和自动思考范式对齐，从而确保在各种场景下对其推理过程实现稳健控制。

### ✅ **阶段三：面向推荐的强化学习（Rec-RL）**

-   使用 **Group Relative Policy Optimization (GRPO)**，无需 critic 模型。
-   **奖励函数**：基于“命中”（Hit）的稀疏规则奖励
    r(R\_i) = \\begin{cases} 1, & \\text{if target item的语义ID in } R\_i \\\\ 0, & \\text{otherwise} \\end{cases}
-   **KL 约束**：防止过度偏离蒸馏后的模型。
-   **目标**：优化 Recall、NDCG 等离散指标，缓解 SFT 的暴露偏差。

* * *

### **4\. 评估协议**

### ✅ 推荐任务（Layer 1 & 2）

-   **Pass@1, Pass@32, Recall@32**：衡量目标物品是否在生成结果中。

### ✅ 文本生成任务（Layer 0 & 3）

-   **LLM-as-Judge**：使用 Gemini-2.5-Pro 作为裁判。
-   **Item Understanding**：采用 **Double-Weighted F1**，结合 BERTScore 与信息点重要性。
-   **Explanation**：由 Gemini 生成参考答案，评估生成解释的准确性与连贯性。

**Ground Truth 生成**：Gemini-2.5-Pro with full metadata access

* * *

## **实验结果**

### **1\. 主实验结果**

### ✅ **RecIF-Bench 性能**

![](https://pic2.zhimg.com/v2-4821e6ab9d7d761fd8cb8ce316cc2047_1440w.jpg)

-   **OneRec-Foundation** 在所有任务上 **全面超越基线**，包括：

-   判别式模型（BERT4Rec, GRU4Rec, SASRec, HSTU）
-   生成式模型（TIGER, LC-Rec）

-   **8B > 1.7B**，**Pro > Standard**，验证模型与数据双重扩展的有效性。

### ✅ **通用能力保留**

![](https://pic3.zhimg.com/v2-5ce4fd0a7cda321301bb32640e47efe6_1440w.jpg)

![](https://picx.zhimg.com/v2-cffa3eaffda6caf7aa0ad7273b6a27f7_1440w.jpg)

-   在 MATH-500、GPQA-Diamond、LiveCodeBench 等 7 个通用基准上保持竞争力。
-   数学推理能力几乎无退化，验证“通用能力保留”成功。

### ✅ **[Amazon 跨域迁移性能](https://zhida.zhihu.com/search?content_id=268479295&content_type=Article&match_order=1&q=Amazon+%E8%B7%A8%E5%9F%9F%E8%BF%81%E7%A7%BB%E6%80%A7%E8%83%BD&zhida_source=entity)**

-   在 **10 个 Amazon 领域**上平均 Recall@10 提升 **26.8%**。
-   **[Text-Augmented Itemic Tokens](https://zhida.zhihu.com/search?content_id=268479295&content_type=Article&match_order=1&q=Text-Augmented+Itemic+Tokens&zhida_source=entity)** 策略最优，碰撞率仅 **0.47%**。

![](https://pica.zhimg.com/v2-8c2ed91cfe27c8eba27ff6c385589f32_1440w.jpg)

-   迁移学习中的一个主要挑战是物品标识符的分布偏移。预训练分词器在广泛且开放领域的语料库（例如，短视频、电子商务产品）上进行优化，导致词典可能无法精细地区分特定垂直领域（如亚马逊商品）中的物品。直接应用会导致较高的冲突率（>30%），造成灾难性的信息丢失：

-   **策略1：扩展残差量化**。通过从预训练的第三层计算残差，并应用有限标量量化（FSQ）（Mentzer et al., 2023）生成第四层码本，将碰撞率降低至3.05%。剩余的碰撞通过基于流行度的解码加以解决。该策略在平均性能上实现了10.0%的提升。
-   **策略2：仅文本适配**。完全绕过物品标记，通过从**物品元数据中提取的5个独特关键词**来表示每个物品，将冲突率降低至4.27%。该策略在平均R@10上相比扩展残差量化提升了18.8%：模型的语言核心保持完整，从而实现稳健的语义理解，同时自然语言表示在狭窄领域中展现出更强的表达能力。然而，该方法**牺牲了预训练物品标记中嵌入的协同过滤信号**。
-   **策略3：文本增强的令牌**。将原始的三层预训练项级令牌与关键词表示进行**拼接**：\[itemic\_tokens\] + \[keywords\]。关键在于，保留了原始预训练的令牌，未进行结构扩展，从而维持了层次化语义。关键词提供了语义消歧（冲突率 0.47%），并实现了语言能力的充分使用。表8显示，该策略在几乎所有数据集上均达到了最先进的性能。持续的性能提升验证了有效的迁移学习需要在严格保持预训练结构完整性的同时，最大化利用基础模型的多样化能力——协同过滤、知识与语义理解。

* * *

### **2\. 消融实验（Ablation Study）**

### ✅ **预训练策略消融**

-   去除 **Itemic-Text Alignment Stage** 导致性能下降，尤其在小模型（0.6B, 1.7B）上显著。
-   验证该阶段是冷启动 itemic token 的**必要语义桥接**。

![](https://pic3.zhimg.com/v2-cfe82efefb98b0b70a1ab17b277c4154_1440w.jpg)

### ✅ **后训练阶段分析**

![](https://pic2.zhimg.com/v2-7477b03a7a85d7f72c8da2a39a077d91_1440w.jpg)

![](https://pic1.zhimg.com/v2-c90add49bfda4c45153504e2b08521a6_1440w.jpg)

![](https://pic4.zhimg.com/v2-afdde6649c09d70c3cc8740a11154f9f_1440w.jpg)

<table data-draft-node="block" data-draft-type="table" data-size="normal" data-row-style="normal"><tbody><tr><td>阶段</td><td>通用能力</td><td>推荐能力</td></tr><tr><td>Multi-task SFT</td><td>下降</td><td>上升</td></tr><tr><td>On-policy Distillation</td><td>显著恢复</td><td>保持</td></tr><tr><td>Rec-RL</td><td>保持</td><td>显著提升</td></tr></tbody></table>

-   **Rec-RL 不仅提升推荐，也增强解释能力（Reco Reason）**，表明“推荐直觉”可迁移。

* * *

## **思想总结**

### 🔍 **核心思想：构建推荐与智能的统一范式**

1.  **从“模式匹配”到“智能代理”** ：当前推荐系统多为“统计模式匹配器”，缺乏理解、推理、解释能力。本文提出将推荐系统升级为**具备指令跟随与自然语言推理能力的智能体**。
2.  **双通道知识融合**

1.  **推荐知识通道**：通过 itemic token 与用户行为注入协同过滤信号。
2.  **通用知识通道**：通过混合通用数据保留世界知识与推理能力。
3.  二者在统一框架下协同进化。

4.  **三阶段能力演进**

1.  **预训练**：建立语义对齐与行为建模
2.  **SFT**：恢复指令跟随
3.  **蒸馏 + RL**：分别优化通用性与推荐精度

6.  **可复现、可扩展、可迁移**

1.  开源完整流程，推动社区发展
2.  验证推荐能力可扩展
3.  在 Amazon 上实现跨域 SOTA

* * *

### 🚀 **未来方向**

1.  **Tokenizer 可迁移性**：如何更好复用 LLM 先验知识，同时保证 item indexing 质量。
2.  **数据混合最优比例**：需更精细控制通用与领域数据比例。
3.  **推理策略优化**：当前 CoT 仅在有限场景有效，需探索 test-time scaling。