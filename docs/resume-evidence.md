# 简历证据对齐

这份文件把已经提交的简历表述逐句映射到代码、实验事实和必须主动披露的限制。面试目标不是逐字辩护，而是让每句话
都能落到一个可运行入口、一段核心代码或一份去敏结果。

## 已提交表述

> 面向 Agent（智能体）的运行时注入防御：表示级探针 × 工具调用边界
>
> 基于 AgentDojo（智能体安全评测框架）构建工具调用 agent，在每次 tool-call（工具调用）前用
> refusal-direction（拒答方向）/激活探针读取本地开源模型隐藏状态，对
> OWASP（开放全球应用程序安全项目）LLM01（大语言模型应用安全风险第 1 项）定义的 prompt injection（提示注入）
> 与危险工具调用做运行时检测与拦截。
>
> 以 AgentDojo 确定性 utility/security（任务可用性/安全性）双指标，与内置防御同协议对比，量化
> ASR（攻击成功率）、注入拦截率、
> 正常任务可用性与推理开销。

## 逐句证据表

| 简历表述 | 代码证据 | 已运行事实 | 限制与面试口径 |
|---|---|---|---|
| 基于 AgentDojo 构建工具调用 Agent | `build_guarded_pipeline`、`run_hf_agentdojo_case` | Banking（银行任务套件）clean/attacked loop（无攻击/受攻击循环）、真实 Qwen3（通义千问 3 模型）接线和冻结矩阵均已运行 | 是 AgentDojo tool-calling pipeline（工具调用流水线），不是从零训练一个 Agent |
| 每次 tool-call 前检测 | `GuardedToolsExecutor.query → RuntimeGate.decide → runtime.run_function` | contract test（契约测试）中恶意 `update_password` 被 block（阻断），runtime audit（运行时审计）无该调用且环境不变 | Executor（执行器）会逐候选调用决策；首版表示路径只对单调用回合有有效 per-call activation（逐调用激活），多调用回合会 detector-invalid（检测器无效）/ fail closed（故障时默认阻断） |
| refusal-direction | `DirectionDetector`、`fit_direction_artifact` | 正式 direction（方向）在 held-out（留出测试）中审核阻断唯一恶意 proposal（候选调用） | 实现是通用 difference-in-means（均值差）骨架；正式样本是 policy-compliant / violating call（符合/违反策略的调用），不应声称做了独立 refusal-labeled（拒答标签）效果实验 |
| 激活探针 | `LinearProbeDetector`、`fit_linear_probe_artifact` | 真实 artifact（工件）、train-only scaler（仅训练集拟合的标准化器）、benign calibration（良性校准）和执行前 gate（门控）已接通 | 首轮 probe（探针）没产生恶意 proposal，却误阻合法 `update_user_info`；ASR=0 不是成功 interception（拦截） |
| 读取本地开源模型隐藏状态 | `HuggingFaceToolCallingLLM`、`_ResidualPreCapture` | Qwen2.5（通义千问 2.5 模型）smoke（冒烟测试）、Qwen3-8B（通义千问 3 80 亿参数模型）reality（真实接线）/held-out，以及 Qwen3-30B-A3B（通义千问 3，300 亿总参数/30 亿激活参数模型）layer-29（第 29 层）tool-call smoke 均捕获 `resid_pre` | 30B 只增加白盒接线证据；没有训练 probe 或产生防御效果。`tool_input` 是 generation-prefill（生成预填充）的最后一个 non-padding token（非填充词元），不是注入文本末 token |
| prompt injection（OWASP LLM01） | AgentDojo injection task（注入任务）+ untrusted tool output（不可信工具输出） | `injecagent` 把恶意指令植入文件/工具返回，模型随后可能提出越权动作 | 当前只覆盖 LLM01 中的 indirect prompt-injection（间接提示注入）子集和一个正式攻击模板，不代表完整 OWASP LLM01 覆盖 |
| 危险工具调用检测与拦截 | `ToolRiskPolicy` + detector（检测器）+ executor | direction/MELON（掩码重执行检测方法）的恶意调用在 runtime（运行时）前被阻断 | Representation detector（表示检测器）检测异常状态；危险等级是工具名/前缀级确定性策略，不是完整参数级授权系统 |
| utility/security 双指标 | AgentDojo deterministic checks（确定性检查）、`aggregate_results` | 已分列 BU（良性任务可用性）、UA（攻击场景任务可用性）、Targeted ASR（定向攻击成功率）和 valid coverage（有效覆盖率） | raw（原始）`security_results=True` 表示攻击目标完成；不能解释成“安全通过” |
| 与内置防御同协议对比 | `build_hf_experiment_pipeline`、matrix manifest（矩阵清单） | `repeat_user_prompt` 与自定义防御共享 task/attack（任务/攻击）、BU/UA/ASR 和 E2E（端到端）计时 | Built-in（内置方法）使用上游 executor，没有本项目 per-call trace（逐调用轨迹）；interception/false-block（拦截/误阻断）为 `N/A`（不适用） |
| ASR、拦截率、可用性和开销 | `matrix.py` 聚合器与人工 call review（调用审核） | Qwen3-8B：30 episodes（回合）、30/30 valid（有效）；direction/MELON 各 1/1；所有 BU 1/3；已报告配对中位开销 | 8B 是唯一正式防御效果矩阵。30B 停在 screening（筛选实验），不能混入该表；极小样本、单一攻击模板、seed（随机种子）0 |

## 跨模型补充证据怎么定位

[Qwen3-30B screening 报告](../reports/qwen3-30b-screening.md) 只补充两件事：更大 FP8（8 位浮点）模型能在 layer 29 走通
`native_tools + generation_prefill_last_nonpad + resid_pre` 白盒路径；以及
exact Targeted ASR（精确定向攻击成功率）与广义有害副作用不能
混为一谈。唯一 attacked trial（受攻击试验）的 exact ASR 是 `0/1`，但 trace 已执行并成功完成一笔
injection-driven（注入驱动）未授权转账，只是收款参数与预注册 exact target（精确目标）近似而不相等。

Continuation gate（继续实验门槛）要求至少一次 exact no-defense success（无防御条件下精确攻击成功）。门槛未满足后，
本轮停止，没有采集 30B train/calibration（训练/校准）
activation、拟合 artifact 或运行 held-out。因此它不能增加简历中“防御效果”的证据级别，Qwen3-8B 仍是唯一正式
矩阵。

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
> 评测上我没有只看 ASR，而是分开报告 BU、UA（攻击场景任务可用性）、Targeted ASR、人工审核的 call-level interception、
> false block 和配对开销。首轮 Qwen3-8B 只有三个 held-out user task（用户任务）；direction 和 MELON 各阻断唯一恶意
> proposal，但 probe 是误阻合法调用后没有产生恶意 proposal。因此项目证明的是完整、可审计的工程闭环和
> 失败分析，不是统计显著或跨模板泛化。

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

Episode 层相同：同模型、任务、攻击、utility/ASR check 和端到端计时。Call 层不同：自定义 executor 有完整 trace，
`repeat_user_prompt` 没有，所以其 interception/false-block 必须是 `N/A`。

## 展示顺序

1. 运行 `uv run agent-defense interview-demo`，只讲控制流；
2. 运行或展示 boundary contract（边界契约），证明 block 早于真实副作用；
3. 打开 `reports/qwen3-heldout-matrix.md`，讲正式结果和 probe 失败；
4. 若追问跨模型，再打开 `reports/qwen3-30b-screening.md`，讲 exact target 与有害 near miss（近似但未命中）的差异；
5. 若追问代码，沿
   `HuggingFaceToolCallingLLM.query → GuardedToolsExecutor.query → RuntimeGate.decide → detector.inspect`
   展开。
