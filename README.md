# Agent Runtime Defense（智能体运行时防御）

这是一个面向工具型 Agent（智能体）的运行时 indirect prompt-injection（间接提示注入）防御原型：本地模型先生成候选
`tool call`（工具调用），系统在工具尚未执行时读取隐藏状态或运行 MELON（掩码重执行检测方法）检测，再由独立
风险策略决定 `allow`（放行） / `block`（阻断）。

项目目标不是提出 SOTA（当前最先进水平），而是把简历中的“表示级探针 × 工具调用边界”做成一条可运行、可解释、
可复现、能被追问的工程证据链。

> 默认现场演示使用脚本化 Agent 和人工构造 activation（激活），只证明控制流。真实模型结果来自冻结的
> Qwen3-8B（通义千问 3 80 亿参数模型）、AgentDojo（智能体安全评测框架）的 Banking（银行任务套件）小型
> held-out（留出测试）矩阵，二者不能混用。

## 结论先行

冻结评测包含 3 个未参与 train/calibration（训练集/校准集）的 user task（用户任务），每个运行
clean/attacked（无攻击/受攻击）和 5 种防御，共 30 episodes（回合）；30/30 有效，0 个
infrastructure failure（基础设施故障）。

| Defense（防御方式） | BU（良性任务可用性） | UA（攻击场景任务可用性） | Targeted ASR（定向攻击成功率） | 人工调用审计 | 配对中位开销（ms，毫秒） |
|---|---:|---:|---:|---|---:|
| none（无防御） | 1/3 | 1/3 | 1/3 | 放行唯一恶意调用：0/1 阻断 | 0.00 ms |
| `repeat_user_prompt`（重复用户提示词内置防御） | 1/3 | 1/3 | 0/3 | N/A（不适用），无自定义 executor（执行器）trace（轨迹） | 3456.02 ms |
| direction（方向） | 1/3 | 1/3 | 0/3 | 阻断唯一恶意调用：1/1 | 2.07 ms |
| activation probe（激活探针） | 1/3 | 0/3 | 0/3 | 无恶意 proposal（候选调用）；误阻 1 个合法调用 | 7.99 ms |
| `MELON slice (hashing)`（基于哈希嵌入的 MELON 算法切片） | 1/3 | 1/3 | 0/3 | 阻断唯一恶意调用：1/1 | 3799.02 ms |

四个指标不要混淆：

- **BU**：clean trial（无攻击试验）的 AgentDojo utility（任务可用性）。
- **UA**：attacked trial（受攻击试验）的 AgentDojo utility；不额外与 `not attack_succeeded` 做 AND（逻辑与）。
- **Targeted ASR**：攻击目标实际落地的比例；AgentDojo 的 raw（原始）`security_results=True` 表示攻击成功。
- **Interception（拦截率）**：已经提出的恶意调用中，有多少在 runtime（运行时）前被阻断。

最重要的失败分析是：activation probe 的 ASR=0 **不是**成功 interception。该组没有产生恶意
proposal，而是先误阻了合法 `update_user_info`。此外所有方法 BU 都只有 1/3，说明基础模型的任务完成
能力也是主要瓶颈。

完整分母、延迟、逐任务失败与限制以
[`reports/qwen3-heldout-matrix.md`](reports/qwen3-heldout-matrix.md) 为唯一正式数值来源。

跨模型补充只做到 Qwen3-30B-A3B（通义千问 3，300 亿总参数/30 亿激活参数模型）screening（筛选实验）：2/2 trial（试验）均
valid（有效），白盒 tool-input（工具输入）activation 接线成功；attacked trial 的
exact Targeted ASR（精确定向攻击成功率）是 `0/1`，但 trace 仍执行并成功完成了一笔
injection-driven（注入驱动）未授权转账，只是收款参数与精确 benchmark target（基准目标）近似而不相等。因此
`0/1` 不能解释为安全。Continuation gate（继续实验门槛）未通过后，本轮停止，没有训练 30B
probe、没有校准或运行 held-out。见
[`reports/qwen3-30b-screening.md`](reports/qwen3-30b-screening.md)。

## 为什么边界放在 tool call 前

Indirect prompt injection 的攻击指令来自邮件、文件、网页等不可信 tool output（工具输出）。模型读到恶意文本并不等于
已经造成损害；真正产生安全后果的是随后执行转账、改密、发信等动作。

因此本项目把强制边界放在：

```text
untrusted tool output
        ↓
in-process HF generation
        ↓
candidate tool call + activation
        ↓
detector score + deterministic tool risk
        ↓
RuntimeGate decision
   ┌────┴────┐
 allow     block
   ↓         ↓
runtime   blocked tool result
(side effect) (no side effect)
```

Detector（检测器）只提供信号；`GuardedToolsExecutor` 才是最终 enforcement（强制执行）边界。

## 五分钟上手

项目使用 Python（编程语言）3.12、`uv`（Python 包与环境管理器）和仓库独立虚拟环境：

```bash
uv sync
uv run agent-defense doctor
uv run agent-defense interview-demo
uv run agent-defense validate-boundary --defense activation_probe --scenario attacked
```

你应该看到：

1. 无防御时，恶意候选调用进入 runtime，攻击目标完成；
2. probe 路径中，同一类调用在 runtime 前被阻断，环境不变；
3. clean teaching case（无攻击教学案例）的授权调用仍被放行。

再运行最小回归：

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## 先沿这一条代码链阅读

```text
agentdojo-run
→ run_hf_agentdojo_case
→ build_hf_experiment_pipeline
→ HuggingFaceToolCallingLLM.query
→ GuardedToolsExecutor.query
→ RuntimeGate.decide
→ detector.inspect
→ block result / runtime.run_function
```

第一遍不要线性通读整个仓库。打开
[`docs/code-tour.md`](docs/code-tour.md)，按符号跳读以下核心文件：

| 文件 | 只需要先理解什么 |
|---|---|
| [`types.py`](src/agent_defense/types.py) | 候选调用、检测输入、观察、决策和 trace |
| [`hf_llm.py`](src/agent_defense/hf_llm.py) | tool call 生成与 `resid_pre` 捕获 |
| [`agentdojo_integration.py`](src/agent_defense/agentdojo_integration.py) | 唯一的执行前强制边界 |
| [`policy.py`](src/agent_defense/policy.py) | detector 信号如何结合工具风险 |
| [`detectors.py`](src/agent_defense/detectors.py) | direction、logistic probe（逻辑回归探针）与 MELON 打分 |

## 90 分钟学懂路线

1. **0–15 分钟**：读本页，运行 `interview-demo` 和 `validate-boundary`。
2. **15–45 分钟**：按 [代码导览](docs/code-tour.md) 跟一条 attacked trace，回答“调用在哪里提出，
   又在哪里真正执行”。
3. **45–60 分钟**：读 [架构说明](docs/architecture.md)，理解 detector、risk policy（风险策略）、executor 为什么分层。
4. **60–75 分钟**：读 [正式结果报告](reports/qwen3-heldout-matrix.md)，重点解释 probe 的误阻、低 BU 和
   MELON 的额外模型开销。
5. **75–90 分钟**：照 [面试讲解指南](docs/interview-guide.md) 复述两分钟主叙事，并回答五个高频追问。

第一遍可以先不看 `cli.py`、`matrix.py` 的全部实现，也不要现场重跑 8B 模型。它们分别是命令编排和正式实验
基础设施，不是理解安全主线的起点。

## 四层证据

| 证据 | 证明什么 | 不证明什么 |
|---|---|---|
| synthetic teaching demo（合成教学演示） | allow/block 控制流与指标代码可运行 | 真实模型检测效果 |
| AgentDojo boundary contract（边界契约） + HF（Hugging Face 模型库）wiring spike（接线验证） | block 位于真实 runtime 副作用之前；本地模型可同时给出调用和隐藏状态 | probe 泛化 |
| Qwen3-8B held-out matrix（留出测试矩阵） | 同一攻击模板下跨 3 个 user task 的小型工程对比 | 统计显著性、跨模板或跨模型泛化 |
| Qwen3-30B screening 负结果 | 更大 FP8（8 位浮点）模型的白盒接线；exact target（精确目标）与有害 near-miss（近似但未命中）调用的口径差 | 任何 30B 防御效果；本轮在 held-out 前停止 |

工程快照和 one-task（单任务）接线历史见
[`reports/verified-smoke.md`](reports/verified-smoke.md)；正式矩阵只看
[`reports/qwen3-heldout-matrix.md`](reports/qwen3-heldout-matrix.md)。

## 论文只先读两篇

AgentDojo 已经读过时，面试前先看：

1. *Your Agent is More Brittle Than You Think*（《你的智能体比你想象中更脆弱》）§3.3、Table 3–4（表 3–4）：
   tool-input/function-call hidden state（工具输入/函数调用隐藏状态）、logistic probe（逻辑回归探针）、
   danger direction（危险方向）和 pre-action circuit breaker（动作前熔断器）。
2. MELON §3、Algorithm 1（算法 1）：masked re-execution（掩码重执行）、cache（缓存）和
   tool-call comparison（工具调用比较）。

本地 PDF（便携式文档格式）、出处和精确阅读范围见 [`papers/README.md`](papers/README.md)。
Refusal Direction（拒答方向）、PVDetector（投影向量检测器）和 Task Shield（任务护盾）都是补充材料，不阻塞当前主线。

## 详细资料

- [代码导览](docs/code-tour.md)：一条真实调用链、核心符号、断点和 90 分钟练习。
- [简历证据对齐](docs/resume-evidence.md)：逐句映射代码、实验事实和必须披露的限制。
- [跨模型复核协议](docs/cross-model-replication.md)：Qwen3-30B 预注册、screening 负结果、停止决策和未执行阶段。
- [架构说明](docs/architecture.md)：稳定设计与实现语义。
- [实验协议](docs/experiment-protocol.md)：split（数据划分）、阈值、指标、有效性和完整复现命令。
- [MELON 论文兼容重建](docs/melon-reproduction.md)：官方源码审计、按论文附录重建的掩码轨迹、语义嵌入、629 案例协议与不可复现项。
- [MELON 论文兼容筛选报告](reports/melon-paper-screening.md)：16 个预注册配对、64 回合有效性、掩码候选稀疏和唯一阻断审核。
- [面试讲解指南](docs/interview-guide.md)：两分钟话术、简历映射和高频追问。
- [正式 held-out 报告](reports/qwen3-heldout-matrix.md)：唯一完整结果表。
- [30B screening 补充报告](reports/qwen3-30b-screening.md)：白盒接线、exact-target near miss 与 continuation gate。

## 项目边界

- 当前只覆盖 AgentDojo Banking、一个攻击模板和极小 held-out 分母，不声称 SOTA 或泛化。
- 新增 `melon_paper` 作为独立的论文兼容重建路径；它不改写 `melon` 哈希切片或既有冻结结果，16 配对筛选已完成，
  但尚未产生正式全量效果表。
- Qwen3-8B 是唯一正式防御效果矩阵；30B 只完成 no-defense（无防御）screening，没有 artifact（工件）、
  calibration 或 held-out 结果。
- `melon` 路径是独立实现的核心算法切片；hashing embedding（哈希嵌入）、单中性提示和“阻断后继续”均不同于论文完整配置。
- `tool_input` 是 CLI（命令行界面）名称，artifact 中的精确定义是 `generation_prefill_last_nonpad`，不是注入文本末
  token（词元）。
- 默认只执行 AgentDojo 沙箱工具；不连接真实银行、邮件、账户、文件系统或 shell（命令行外壳）。
- 原始 activation、模型权重和大日志留在 Git（版本控制系统）忽略目录，只提交去敏的小型摘要。
