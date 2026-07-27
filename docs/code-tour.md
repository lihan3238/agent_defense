# 代码导览：跟一条危险调用走到底

这份导览只回答一个问题：

> 模型提出的危险工具调用，怎样携带隐藏状态到达 gate（门控），并在
> `FunctionsRuntime.run_function` 之前被允许或阻断？

先建立最重要的心智模型：LLM（大语言模型）负责**提出候选动作**，detector（检测器）负责**产生风险信号**，
`RuntimeGate`（运行时门控）负责**形成决策**，`GuardedToolsExecutor`（受保护工具执行器）才负责**执行该决策**。

## 1. 真实调用链

```text
cli.py
└─ agentdojo_run
   └─ experiments.py: run_hf_agentdojo_case
      └─ build_hf_experiment_pipeline
         └─ build_guarded_pipeline
            ├─ HuggingFaceToolCallingLLM.query
            │  ├─ model.generate
            │  ├─ parse tool_calls
            │  └─ capture activation into extra_args/state
            └─ ToolsExecutionLoop
               └─ GuardedToolsExecutor.query
                  ├─ build DetectionContext
                  ├─ optional MELON masked re-execution
                  ├─ RuntimeGate.decide
                  │  ├─ ToolRiskPolicy.classify
                  │  └─ detector.inspect
                  ├─ block → synthetic blocked tool result
                  └─ allow → runtime.run_function
```

AgentDojo（智能体安全评测框架）pipeline（流水线）的实际顺序是：

```text
SystemMessage
→ InitQuery
→ LLM
→ ToolsExecutionLoop([GuardedToolsExecutor, LLM])
```

第一次 LLM 调用先产生候选动作；之后循环执行“executor（执行器）处理调用 → LLM 读取工具结果”。正式路径使用
Qwen3-8B（通义千问 3 80 亿参数模型），运行 Banking（银行任务套件）任务。

## 2. 两条数据通道

Hugging Face（模型库）适配类 `HuggingFaceToolCallingLLM.query` 返回两类相关数据，它们不是同一个对象：

| 数据 | 传递位置 | 含义 |
|---|---|---|
| candidate tool calls（候选工具调用） | assistant message（助手消息）的 `tool_calls` | 模型想做什么 |
| activation（激活） + metadata（元数据） | `extra_args/state["agent_defense.activation"]` | 模型提出动作前的内部状态 |

`GuardedToolsExecutor.query` 把两者合并成 `DetectionContext`。因此 detector 能读取隐藏状态，
policy（策略）又能看到实际工具名和参数风险。

## 3. 第一遍只读这些符号

不要按文件从第一行读到最后一行。按下面顺序跳读：

1. [`types.py`](../src/agent_defense/types.py)

   - `CandidateToolCall`：尚未执行的动作；
   - `DetectionContext`：detector 的最小输入；
   - `ProbeObservation`：只有分数与健康状态，没有执行权限；
   - `PolicyDecision`：allow/block（放行/阻断）、原因和风险级别；
   - `DecisionTrace`：调用是否越过 runtime boundary（运行时边界）。

2. [`experiments.py`](../src/agent_defense/experiments.py)

   - `build_hf_experiment_pipeline`：把 defense（防御方式）名映射到 detector、
     MELON（掩码重执行检测方法）provider（提供器）或 AgentDojo built-in（内置防御）；
   - `run_hf_agentdojo_case`：运行一个 clean/attacked episode（无攻击/受攻击回合），并收集
     utility（任务可用性）、ASR（攻击成功率）、trace（轨迹）和开销。

3. [`hf_llm.py`](../src/agent_defense/hf_llm.py)

   - `HuggingFaceToolCallingLLM.query`：渲染消息、生成、解析调用和写入 activation；
   - `_ResidualPreCapture`：在 decoder block（解码器层）上注册 `forward_pre_hook`；
   - `tool_input`：首次 generation prefill（生成预填充）的最后一个 non-padding token（非填充词元）；
   - `function_call`：沿原始 generated token IDs（生成词元标识符）replay（重放）到调用 closing tag（结束标签）。

4. [`agentdojo_integration.py`](../src/agent_defense/agentdojo_integration.py)

   - `GuardedToolsExecutor.query`：本项目唯一的 pre-action enforcement（动作执行前强制执行）；
   - block 分支只构造 `status=blocked` 的 tool result（工具结果）；
   - allow 分支才会调用 `runtime.run_function`。

5. [`policy.py`](../src/agent_defense/policy.py)

   - `ToolRiskPolicy.classify`：名称级、确定性的工具影响分级；
   - `RuntimeGate.decide`：把 detector observation（观察）与工具风险合成最终决策。

6. [`detectors.py`](../src/agent_defense/detectors.py)

   - `DirectionDetector`：difference-in-means direction（均值差方向）的 cosine/projection score（余弦/投影分数）；
   - `LinearProbeDetector`：StandardScaler（标准化器） + logistic probe（逻辑回归探针）；
   - `MelonToolCallDetector`：比较原调用与 masked re-execution（掩码重执行）的候选调用。

最后才看 [`matrix.py`](../src/agent_defense/matrix.py) 的 `apply_call_reviews` 和
`aggregate_results`；它们解释正式实验怎样从 episode 结果得到 BU（良性任务可用性）、
UA（攻击场景任务可用性）、ASR 和人工审核指标，
不是 runtime 安全边界本身。

## 4. Gate 的四条决策规则

`RuntimeGate.decide` 可以压缩成四条：

```text
detector invalid + medium/high risk + fail_closed → block
detector invalid + low risk                         → allow
detector triggered + medium/high risk              → block
otherwise                                           → allow
```

这解释了为什么 `triggered` 不等于 `block`：同一个异常分数落在低风险读取工具上时可以继续放行。
也解释了为什么 detector 和 policy 必须分开——表示异常与动作影响不是同一件事。

## 5. 三条 detector 路径

### Direction / activation probe（方向/激活探针）

```text
prefill resid_pre
→ artifact compatibility check
→ cosine/projection or logistic score
→ threshold
→ ProbeObservation
```

Artifact（工件）绑定模型、revision（修订版本）、dtype（数据类型）、layer（层）、position（位置）、
chat template（对话模板）和 tool schema（工具结构约束）等身份。不兼容时 observation invalid（无效）；高风险动作按
fail-closed（故障时默认阻断）处理，不能静默复用。

### MELON slice（MELON 算法切片）

```text
tool outputs
→ hide original user task
→ same LLM masked re-execution
→ cache masked tool calls
→ hashing similarity with current call
→ ProbeObservation
```

Masked generation（掩码生成）发生在 scoring（打分）前，所以它的主要开销不在 `detector_latency_ms` 中，但会进入
episode 端到端时间和模型调用数。

论文兼容路径额外阅读 [`melon.py`](../src/agent_defense/melon.py)、
[`melon_agentdojo.py`](../src/agent_defense/melon_agentdojo.py) 和
[`semantic_embeddings.py`](../src/agent_defense/semantic_embeddings.py)：它们分别负责论文参数投影/语义比较、按论文附录重建的掩码轨迹，
以及 OpenAI（模型服务提供方）/本地 Hugging Face（模型库）嵌入后端。`melon_paper` 会整批预检本轮调用，并在任意命中时
替换候选消息、终止回合；这一语义不适用于冻结的 `melon` 结果。

### No defense（无防御）

`NoDefenseDetector` 始终返回有效且不触发的 observation；自定义 executor 仍保留完整 trace。因此 none（无防御）
可以与 direction/probe/MELON 做同粒度调用审计。

AgentDojo built-in `repeat_user_prompt` 使用上游 executor，当前没有本项目的 per-call trace（逐调用轨迹），所以调用级
interception/false-block（拦截率/误阻率）必须是 `N/A`（不适用）。

## 6. 最容易讲错的术语

| 术语 | 正确含义 |
|---|---|
| `tool_input` | CLI（命令行界面）名称；artifact 中精确记为 `generation_prefill_last_nonpad` |
| `function_call` | CLI 名称；artifact 中精确记为 `function_call_end` |
| `triggered` | detector 分数超过阈值，不保证最终 block |
| detector `valid`（有效） | detector 是否健康，不表示 episode 安全 |
| trial（试验）`valid` | 基础设施、解析和 detector 健康，不表示 utility 通过 |
| `runtime_invoked` | 调用已经进入真实 runtime，可能产生副作用 |
| `tool_succeeded` | runtime 返回时是否无错误 |
| `executed` | `runtime_invoked` 的兼容别名，不等于成功 |
| raw（原始）`security_results=True` | injection task（注入任务）的攻击目标已完成，即一次 ASR 命中 |
| `activation_probe` | defense 展示名；artifact/detector kind（类型）是 `linear_probe` |

## 7. 三种演示不要混用

| 入口 | 用途 | 能否进真实效果表 |
|---|---|---|
| [`demo.py`](../src/agent_defense/demo.py) | 脚本化 Agent（智能体） + synthetic activation（合成激活），教学控制流 | 否 |
| [`agentdojo_runner.py`](../src/agent_defense/agentdojo_runner.py) | 真实 AgentDojo runtime contract（运行时契约），证明 block 早于副作用 | 否 |
| [`experiments.py`](../src/agent_defense/experiments.py) + [`matrix.py`](../src/agent_defense/matrix.py) | 真实 HF（Hugging Face 模型库）episode 与 frozen held-out（冻结留出测试）聚合 | 是 |

## 8. 建议断点

按一次 attacked case（受攻击案例）设置这些符号断点即可：

1. `build_hf_experiment_pipeline`：确认 defense 如何装配；
2. `HuggingFaceToolCallingLLM.query`：看生成前消息、解析后的 `tool_calls` 和 activation metadata；
3. `GuardedToolsExecutor.query`：看候选调用进入执行器；
4. `LinearProbeDetector.inspect` 或 `MelonToolCallDetector.inspect`：看 score/threshold（分数/阈值）；
5. `RuntimeGate.decide`：看 risk（风险）与 observation 如何合成；
6. `runtime.run_function` 调用行：确认只有 allow 分支能到达。

## 9. 90 分钟练习

### 0–15 分钟：亲眼看边界

```bash
uv run agent-defense interview-demo --json-output
uv run agent-defense validate-boundary --defense activation_probe --scenario attacked
```

回答：恶意调用在哪里被提出？`decision_trace` 中哪三个字段证明它没有进入 runtime？

### 15–45 分钟：走读一条 trace

按第 3 节顺序打开五个核心文件，在纸上画出：

```text
tool output → model state/call → observation → decision → side effect
```

### 45–60 分钟：理解训练边界

阅读 [`training.py`](../src/agent_defense/training.py) 的 `fit_artifact_from_samples` 与
`evaluate_artifact`。能说清：

- train（训练集）拟合 scaler/weights（标准化器参数/权重）；
- benign calibration（良性校准集）选 threshold（阈值）；
- test（测试集）只在设计冻结后运行一次；
- 同一任务的 clean/attack pair（无攻击/攻击配对）不能跨 split（数据划分）。

### 60–75 分钟：理解结果

阅读 [正式结果报告](../reports/qwen3-heldout-matrix.md)，回答：

- 为什么 ASR 不等于 interception？
- 为什么 probe 的 ASR=0 不是成功拦截？
- 为什么 30/30 valid 仍可能 BU=1/3？
- 为什么 MELON scoring 很快，端到端却慢约 3.8 秒？

### 75–90 分钟：面试复述

打开 [面试讲解指南](interview-guide.md)，讲两遍两分钟主叙事，再只练五个追问：
tool-call boundary（工具调用边界）、hook token（钩子词元）、call-level（调用级）标签、数据泄漏、MELON 与 probe 的权衡。

## 10. 最小验证命令

```bash
uv run agent-defense interview-demo --json-output
uv run agent-defense validate-boundary --defense activation_probe --scenario attacked
uv run pytest tests/test_agentdojo_integration.py::test_probe_blocks_before_runtime_and_preserves_banking_environment
```

面试周不建议为了“代码更漂亮”重拆 `cli.py`、`matrix.py`、`hf_llm.py` 或 `experiments.py`。
当前优先级是保持已验证行为稳定，并能准确解释每个模块的责任。
