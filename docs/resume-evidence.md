# 简历证据对齐

这份文件把已经提交的简历表述逐句映射到代码、实验事实和必须主动披露的限制。面试目标不是逐字辩护，而是让每句话
都能落到一个可运行入口、一段核心代码或一份去敏结果。

## 已提交表述

> 面向 Agent（智能体）的运行时注入防御：表示级探针 × 工具调用边界
>
> 基于 AgentDojo（智能体安全评测框架）构建工具调用 agent，在每次 tool-call（工具调用）前用
> refusal-direction（拒答方向）/激活探针读取本地开源模型隐藏状态，对
> OWASP（Open Worldwide Application Security Project，开放全球应用程序安全项目）
> LLM01（其中 LLM 为 Large Language Model，即大语言模型；01 为风险编号）定义的 prompt injection（提示注入）
> 与危险工具调用做运行时检测与拦截。
>
> 以 AgentDojo 确定性 utility/security（任务可用性/安全性）双指标，与内置防御同协议对比，量化
> ASR（Attack Success Rate，攻击成功率）、注入拦截率、
> 正常任务可用性与推理开销。

## 逐句证据表

| 简历表述 | 代码证据 | 已运行事实 | 限制与面试口径 |
|---|---|---|---|
| 基于 AgentDojo 构建工具调用 Agent | `build_guarded_pipeline`、`run_hf_agentdojo_case` | Banking（银行任务套件）表示级 clean/attacked loop（无攻击/受攻击循环）与四套件完整分母矩阵均已运行 | 是 AgentDojo tool-calling pipeline（工具调用流水线），不是从零训练一个 Agent |
| 每次 tool-call 前检测 | `GuardedToolsExecutor.query → RuntimeGate.decide → runtime.run_function` | contract test（契约测试）中恶意 `update_password` 被 block（阻断），runtime audit（运行时审计）无该调用且环境不变 | Executor（执行器）会逐候选调用决策；首版表示路径只对单调用回合有有效 per-call activation（逐调用激活），多调用回合会 detector-invalid（检测器无效）/ fail closed（故障时默认阻断） |
| refusal-direction | `DirectionDetector`、`fit_direction_artifact` | 正式 direction（方向）在 held-out（留出测试）中审核阻断唯一恶意 proposal（候选调用） | 实现是通用 difference-in-means（均值差）骨架；正式样本是 policy-compliant / violating call（符合/违反策略的调用），不应声称做了独立 refusal-labeled（拒答标签）效果实验 |
| 激活探针 | `LinearProbeDetector`、`fit_linear_probe_artifact` | 真实 artifact（工件）、train-only scaler（仅训练集拟合的标准化器）、benign calibration（良性校准）和执行前 gate（门控）已接通 | 首轮 probe（探针）没产生恶意 proposal，却误阻合法 `update_user_info`；ASR=0 不是成功 interception（拦截） |
| 读取本地开源模型隐藏状态 | `HuggingFaceToolCallingLLM`、`_ResidualPreCapture` | Qwen2.5（通义千问 2.5 模型）smoke（冒烟测试）、Qwen3-8B（通义千问 3 80 亿参数模型）reality（真实接线）/held-out，以及 Qwen3-30B-A3B（通义千问 3，300 亿总参数/30 亿激活参数模型）layer-29（第 29 层）tool-call smoke 均捕获 `resid_pre` | 30B 只增加白盒接线证据；没有训练 probe 或产生防御效果。`tool_input` 是 generation-prefill（生成预填充）的最后一个 non-padding token（非填充词元），不是注入文本末 token |
| prompt injection（OWASP LLM01） | AgentDojo injection task（注入任务）+ untrusted tool output（不可信工具输出） | `injecagent` 把恶意指令植入文件/工具返回，模型随后可能提出越权动作 | 当前只覆盖 LLM01 中的 indirect prompt-injection（间接提示注入）子集和一个正式攻击模板，不代表完整 OWASP LLM01 覆盖 |
| 危险工具调用检测与拦截 | `ToolRiskPolicy` + detector（检测器）+ executor | 30 回合人工审核中 direction/MELON（Masked re-Execution and TooL comparisON，掩码重执行与工具调用比较）各阻断唯一恶意调用；完整矩阵中 `melon_paper` 自动匹配到 92/287 个精确参考调用阻断 | Representation detector（表示检测器）检测异常状态；完整矩阵调用统计是自动工具名/参数匹配，不是人工恶意标签；风险策略也不是参数级授权系统 |
| utility/security 双指标 | AgentDojo deterministic checks（确定性检查）、`aggregate_results` | 已分列 BU（Benign Utility，良性任务可用性）、UA（Utility Under Attack，攻击场景任务可用性）、Targeted ASR（Targeted Attack Success Rate，定向攻击成功率）和 valid coverage（有效覆盖率） | raw（原始）`security_results=True` 表示攻击目标完成；不能解释成“安全通过”；跨防御效果优先看共同有效配对 |
| 与内置防御同协议对比 | `build_hf_experiment_pipeline`、matrix manifest（矩阵清单） | 30 回合矩阵中，`repeat_user_prompt` 与自定义防御共享 task/attack（任务/攻击）、BU/UA/ASR 和 E2E（End-to-End，端到端）计时 | Built-in（内置方法）使用上游 executor，没有本项目 per-call trace（逐调用轨迹）；完整 MELON 主矩阵只比较 none（无防御）与 `melon_paper` |
| ASR、拦截率、可用性和开销 | `matrix.py` 聚合器、受检分析脚本与 call review（调用审核） | 完整矩阵计划 1452 回合、有效 1380；共同有效 attacked 577 配对上 ASR 146→53、clean utility 36→35、attacked utility 166→157，中位开销 +4.923 秒 | 单一 `important_instructions` 攻击、本地 MiniLM（小型句向量模型）后端；不是论文四攻击或原始数值复现；30B 停在 screening（筛选实验） |

## 两张矩阵各自承担什么证据

| 矩阵 | 直接回答的问题 | 核心事实 | 正式来源 |
|---|---|---|---|
| 30 回合表示级 held-out（留出测试）矩阵 | 简历中的隐藏状态、direction（方向）、probe（探针）、内置防御和执行前阻断是否真正接通 | 30/30 有效；direction 与 MELON 各人工审核到 1/1 恶意调用阻断；activation probe（激活探针）误阻合法调用 | [`reports/qwen3-heldout-matrix.md`](../reports/qwen3-heldout-matrix.md) |
| 1452 回合 MELON 主矩阵 | 论文兼容动作比较路径在 AgentDojo `v1.1.2` 四套件全部 629 个攻击配对上的效果、代价与失败是什么 | 1380/1452 有效；共同有效 attacked 577 配对上 ASR 146/577→53/577，93 个改善、0 个反向 | [`reports/qwen3-v112-full-matrix.md`](../reports/qwen3-v112-full-matrix.md) |

主矩阵中 none（无防御）有效 707/726，`melon_paper` 有效 673/726。后者有效回合里自动阻断 92/287 个
exact-reference call（精确参考调用）、46 个 non-reference call（非参考调用），clean（无攻击）场景误阻 1 次；这些是
automatic exact syntactic match（自动精确语法匹配），不是人工恶意调用标签。Travel（旅行任务套件）纯文本攻击的 20 个
共同有效配对中，两组 ASR 都是 7/20，且没有阻断。

在 93 个 ASR 成功→失败的共同有效配对中，69 个配对至少存在精确参考调用阻断，其余 24 个配对只有非参考调用阻断；
精确参考里包含只读前置动作，非参考里也包含合法调用。因此 93 是 episode-level transition（回合级结果转移），
不是经人工确认的恶意调用拦截数。

`melon_paper` 的无效回合分解为 34 个 masked tool-parse error（掩码轨迹工具解析错误）、16 个其他解析错误和
3 个 OOM（Out of Memory，显存不足）；none 为 15 个解析错误和 4 个 OOM。对涉及 OOM 的 4 个语义案例另用
fresh process（全新进程）
成对复核 8 回合，8/8 的有效性、失败、错误、utility（任务可用性）与 ASR 均和主矩阵一致；该复核只确认长上下文资源上限，
不回填主指标。

## 跨模型补充证据怎么定位

[Qwen3-30B screening 报告](../reports/qwen3-30b-screening.md) 只补充两件事：更大 FP8（8-bit Floating Point，8 位浮点）模型能在 layer 29 走通
`native_tools + generation_prefill_last_nonpad + resid_pre` 白盒路径；以及
exact Targeted ASR（精确定向攻击成功率）与广义有害副作用不能
混为一谈。唯一 attacked trial（受攻击试验）的 exact ASR 是 `0/1`，但 trace 已执行并成功完成一笔
injection-driven（注入驱动）未授权转账，只是收款参数与预注册 exact target（精确目标）近似而不相等。

Continuation gate（继续实验门槛）要求至少一次 exact no-defense success（无防御条件下精确攻击成功）。门槛未满足后，
本轮停止，没有采集 30B train/calibration（训练/校准）
activation、拟合 artifact 或运行 held-out。因此它不能增加简历中“防御效果”的证据级别，Qwen3-8B 仍是唯一正式
产生防御效果表的生成模型；8B 的表示级矩阵与 MELON 主矩阵仍须分开解释。

## 你真正实现的四层结构

```text
模型层
  HuggingFaceToolCallingLLM
  → candidate tool call + resid_pre

检测层
  DirectionDetector / LinearProbeDetector / MelonToolCallDetector
  → ProbeObservation

决策层
  ToolRiskPolicy + RuntimeGate
  → allow / block

执行与评测层
  GuardedToolsExecutor
  → runtime side effect or blocked result
  → AgentDojo utility / attack-goal checks
```

这四层是项目最重要的工程解释：hidden-state probe（隐藏状态探针）不是安全边界，`RuntimeGate` 只形成决策，真正阻止副作用的是
`GuardedToolsExecutor` 中位于 `runtime.run_function` 之前的 block 分支。

## 60 秒简历回答

> 这个项目处理 indirect prompt injection（间接提示注入）：Agent 从文件或工具返回中读到恶意指令，随后可能提出越权工具调用。
> 我在进程内 Hugging Face（模型库）中的模型生成候选调用时捕获指定层的 generation-prefill `resid_pre`，用
> difference-in-means direction 或 logistic probe（逻辑回归探针）打分，再与确定性的工具风险策略组合。最终强制边界在
> AgentDojo executor 内，候选调用只有被 allow（放行）才会进入 `runtime.run_function`。
>
> 评测上我没有只看 ASR，而是分开报告 BU、UA（攻击场景任务可用性）、Targeted ASR、调用阻断、有效覆盖和配对开销。
> 30 回合表示级矩阵直接验证 direction/probe 与执行边界：direction 和 MELON 各阻断唯一人工审核的恶意
> proposal（候选调用），但 probe 是误阻合法调用后没有产生恶意 proposal。随后四套件主矩阵计划 1452 回合、有效 1380；
> 在 577 个共同有效 attacked（受攻击）配对上，`melon_paper` 把 ASR 从 146/577 降到 53/577，但 clean utility 从
> 36/95 降到 35/95，attacked utility 从 166/577 降到 157/577，中位配对开销增加 4.923 秒。
> 这仍只是本地 MiniLM 后端和单一攻击的重跑，不是 MELON 论文四攻击或原始数值复现。

## 面试官可能抓住的词

### “每次 tool-call 前”是否准确？

准确的执行语义是：`GuardedToolsExecutor` 会遍历 assistant message（助手消息）中的每个候选调用，并在每次
`runtime.run_function` 前产生独立决策。当前表示采集仍是“每个生成回合一份 prefill activation”；若同一回合生成
多个调用，代码不会把一份状态冒充多份 per-call activation，而会标记 detector invalid，对高风险调用 fail closed。

### 为什么简历写 refusal-direction？

仓库实现的是可承载 refusal direction 的 difference-in-means 框架；但正式 AgentDojo artifact 使用
policy-compliant / policy-violating call states（调用状态），因此面试时应主动称为
**policy-violation direction baseline（策略违规方向基线）**。Refusal Direction（拒答方向）论文提供表示工程基础，
不等于本项目已经验证“拒答方向就是注入方向”。

### “危险工具调用检测”由谁负责？

Detector 回答“当前表示或动作比较是否异常”；`ToolRiskPolicy` 回答“这个工具的潜在影响有多大”；executor
负责 enforcement（强制执行）。当前风险策略主要按工具名和前缀分级，尚未实现用户授权、金额、收件人等参数级
policy engine（策略引擎）。

### “同协议”是否完全同粒度？

30 回合内置对照的 Episode（回合）层相同：同模型、任务、攻击、utility/ASR check（任务可用性/攻击成功率检查）和端到端计时。
Call（调用）层不同：自定义 executor 有完整 trace，`repeat_user_prompt` 没有，所以其 interception/false-block 必须是
`N/A`（Not Applicable，不适用）。1452 回合 MELON 主矩阵只比较 none 与 `melon_paper`，不能冒充内置防御的完整分母对比。

## 展示顺序

1. 运行 `uv run agent-defense interview-demo`，只讲控制流；
2. 运行或展示 boundary contract（边界契约），证明 block 早于真实副作用；
3. 打开 `reports/qwen3-v112-full-matrix.md`，讲完整分母、共同有效 ASR 转移、可用性、失败和开销；
4. 打开 `reports/qwen3-heldout-matrix.md`，讲简历最直接的表示探针证据和 probe 失败；
5. 若追问跨模型，再打开 `reports/qwen3-30b-screening.md`，讲 exact target 与有害 near miss（近似但未命中）的差异；
6. 若追问代码，沿
   `HuggingFaceToolCallingLLM.query → GuardedToolsExecutor.query → RuntimeGate.decide → detector.inspect`
   展开。
