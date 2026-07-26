# 论文阅读地图

这里保存本项目在 2026-07-22 使用的论文快照。目标不是把论文全部复现，而是用最少阅读支撑一条可在一周内完成、能被面试官追问的实验主线。

论文版权归作者或出版方所有；若复用官方代码，仍需单独检查对应仓库的许可证和固定版本。本目录中的
PDF（便携式文档格式）SHA-256（256 位安全哈希算法）校验值见 [`SHA256SUMS`](SHA256SUMS)，可在 `papers/` 下运行
`sha256sum -c SHA256SUMS` 验证。若仓库将公开，推送 PDF 前还需逐篇确认再分发许可；保守做法是只提交本索引和官方链接。

## 面试前先只盯两篇

1. **先读 `Your Agent is More Brittle Than You Think`（《你的智能体比你想象中更脆弱》）§3.3 与
   Table 3–4（表 3–4）。**
   它是最接近本项目主线的方法参考：比较 tool-input/function-call phase（工具输入/函数调用阶段）的
   hidden state（隐藏状态）、logistic probe（逻辑回归探针）、danger-direction cosine（危险方向余弦）和
   pre-action circuit breaker（动作前熔断器）。本项目需要自行把这些概念操作化到
   AgentDojo（智能体安全评测框架）/Hugging Face（模型库）接口；论文没有公开本仓库所采用的精确
   token capture（词元捕获）约定。
2. **再读 MELON（掩码重执行检测方法）§3、Algorithm 1（算法 1）。**它是本项目的顶会复现线，负责
   masked re-execution（掩码重执行）与动作提交前的 tool-call comparison（工具调用比较）。

AgentDojo 已经看过时，把它当 benchmark（基准测试）/API（应用程序编程接口）手册按需查；
Refusal Direction（拒答方向方法）、PVDetector（投影向量检测器）和 Task Shield（任务护盾）暂不
通读。这样两篇就足够理解当前实现并准备第一轮面试叙事。

## 方法资料索引

下列编号用于资料管理，不代表面试前阅读顺序；优先级以上方“先只盯两篇”为准。

### 1. AgentDojo — 评测底座

- 本地 PDF：[`core/2406.13352_AgentDojo_NeurIPS2024.pdf`](core/2406.13352_AgentDojo_NeurIPS2024.pdf)
- 出处：NeurIPS（神经信息处理系统大会）2024 Datasets and Benchmarks（数据集与基准测试）；
  [arXiv（开放论文预印本平台）](https://arxiv.org/abs/2406.13352)；[官方代码](https://github.com/ethz-spylab/agentdojo)
- 优先阅读：§3.1–3.4、§4.1、§4.3。
- 为什么读：明确任务、攻击、工具环境以及 deterministic utility/security checks（确定性任务可用性/安全性检查），
  避免自造一个只能演示、不能比较的 benchmark。
- 决定：任务/攻击子集、指标分母、失败桶，以及所有防御共用的实验协议。

### 2. MELON — 顶会核心算法对照切片

- 本地 PDF：[`core/2502.05174_MELON_ICML2025.pdf`](core/2502.05174_MELON_ICML2025.pdf)
- 出处：ICML（国际机器学习大会）2025；[arXiv](https://arxiv.org/abs/2502.05174)；
  [官方代码](https://github.com/kaijiezhu11/MELON)
- 优先阅读：§3.1–3.3、Algorithm 1、§4.1–4.5、Appendix（附录）A.1–A.3。
- 为什么读：它与本项目最接近的顶会主线是“AgentDojo + 多步工具调用 + 动作执行前检测”。核心是
  masked re-execution、tool-call cache（工具调用缓存）和工具调用相似度比较。
- 决定：作为 `no defense / representation probe`（无防御/表示探针）主线之外的顶会核心算法切片对照，不阻塞简历主项目闭环。
- 注意：官方实现规模很小且面向旧接口，未见明确 LICENSE（许可证文件）；按论文独立重实现最小算法，不整段复制源码。

### 3. PVDetector — 可选表示扩展

- 本地 PDF：[`frontier/2607.12624_PVDetector_Hidden_State_PI_Detection.pdf`](frontier/2607.12624_PVDetector_Hidden_State_PI_Detection.pdf)
- 出处：[arXiv v1](https://arxiv.org/abs/2607.12624)；作者稿标注 ACM MM（美国计算机协会国际多媒体会议）2026，但当前 PDF 的
  DOI（数字对象标识符）仍为占位符；[官方代码](https://github.com/Claresigle/PVDetector)
- 优先阅读：§4.2–4.3、§5、§6.1.6、§6.2.3、§6.2.5–6.2.7。
- 为什么读：它从 policy-compliant / policy-violating（符合/违反策略）对比样本的最后 token 隐藏状态提取
  difference-in-means（均值差）向量，再在关键层聚合投影分数，正好提供比泛化的 refusal direction 更贴近“策略违规”的
  表示级方案。
- 决定：作为面试追问时的可选表示扩展；当前不再临时追加到已冻结矩阵。
- 限制：论文面向 purpose-specific agent（特定用途智能体）的输入级 PI（提示注入）检测，不等同于 AgentDojo 的
  indirect PI（间接提示注入）多步轨迹；迁移是否成立必须实验验证。

## P1（第一优先级）：读方法与撞车边界

### 4. Your Agent is More Brittle Than You Think — 表示主方法与撞车检查

- 本地 PDF：[`frontier/2604.03870_Agent_More_Brittle_RepE_Circuit_Breaker.pdf`](frontier/2604.03870_Agent_More_Brittle_RepE_Circuit_Breaker.pdf)
- 出处：[arXiv v1](https://arxiv.org/abs/2604.03870)，当前版本标注为 under review（评审中），论文中未给出官方代码仓库。
- 优先阅读：§3.1、§3.3、Table 3–4。
- 为什么读：它已经在动态工具 Agent（智能体）场景中比较 logistic probe 与 danger-direction cosine score（危险方向余弦分数），
  并对 tool-input / function-call 两个位置做了消融。
- 决定：将 §3.3 作为 tool-input logistic probe 的工程化迁移目标，并明确记录本项目自己的 token position（词元位置）、
  hook（钩子）和 artifact（工件）约定；同时不能把“AgentDojo + hidden-state probe（隐藏状态探针）+
  pre-action circuit breaker”描述为
  新颖贡献，更诚实的定位是独立实现、跨框架适配和同协议比较。

### 5. Refusal in Language Models Is Mediated by a Single Direction（语言模型的拒答行为由单一方向介导）— 表示工程基础

- 本地 PDF：[`core/2406.11717_Refusal_Single_Direction_NeurIPS2024.pdf`](core/2406.11717_Refusal_Single_Direction_NeurIPS2024.pdf)
- 出处：NeurIPS 2024；[arXiv](https://arxiv.org/abs/2406.11717)；[官方代码](https://github.com/andyrdt/refusal_direction)
- 优先阅读：§2.3、§3、§7 和 limitations（局限性）。
- 为什么读：理解 difference-in-means 方向、投影和因果干预，但“拒绝有害请求”不等于“识别间接注入或越权工具调用”。
- 决定：只作为基础方法或 sanity-check baseline（合理性检查基线），不再作为项目唯一理论依据。

## P2（第二优先级）：有余力再读

### 6. Task Shield — 动作对齐备选

- 本地 PDF：[`alternatives/2412.16682_Task_Shield_ACL2025.pdf`](alternatives/2412.16682_Task_Shield_ACL2025.pdf)
- 出处：ACL（计算语言学协会年会）2025 Long Paper（长文）；[arXiv](https://arxiv.org/abs/2412.16682)
- 优先阅读：§3、§4、§5。
- 为什么读：它把安全问题改写成“每条指令和工具调用是否服务于用户目标”，与运行时动作边界高度相关。
- 决定：只作为动作对齐方向的设计讨论，不纳入当前实现。

## 阅读后应形成的项目结论

1. **固定问题：** indirect prompt injection（间接提示注入）通过不可信 tool output（工具输出）劫持多步 Agent，
   并最终形成越权工具调用。
2. **固定执行边界：**检测信号可以来自重执行或隐藏状态，但真正的 allow/block（放行/阻断）必须发生在候选工具调用
   产生后、工具尚未执行时。
3. **当前实现：**同一 AgentDojo 协议下已有 `none`、`repeat_user_prompt`、`direction`、
   `activation_probe` 和 MELON slice（MELON 算法切片），并完成首轮冻结 held-out（留出测试）矩阵。
4. **当前选择：**function-call 位置与 PVDetector projection（投影）保留为可选扩展，不影响面试主线，也不根据已观察
   test（测试集）临时追加。
5. **贡献表述：**这是受前沿工作启发的独立工程化迁移与可复现实验框架，不声称发明 hidden-state Agent 防御，
   也不把小型 8B held-out 结果称为跨场景验证。当前可讨论的增量是统一动作边界、可审计适配、同协议对照和
   诚实的失败分析。

## 面试前范围控制

- 不再扩读六篇论文；先把顶部两篇与本仓库实现逐点对齐。
- 不依据当前 held-out 结果回调 layer（层）、position（位置）、threshold（阈值）或 artifact。
- 不现场重跑 8B 矩阵；展示去敏报告，并用 synthetic demo（合成演示）讲控制流。
- 若面试官追问 PVDetector、Task Shield 或 refusal direction，再按对应小节回答它们与当前主线的关系。
