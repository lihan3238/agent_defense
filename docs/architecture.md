# 架构说明

## 1. 要解决的问题

本项目关注 indirect prompt injection：攻击指令不直接出现在用户请求中，而是藏在邮件、文件、网页或
其他工具返回值里。工具型 Agent 读取这些不可信内容后，可能把它转化为转账、修改账户或外传数据等调用。

风险真正落地的时刻不是“模型读到了恶意文本”，而是：

> 模型已经提出候选 tool call，执行器尚未调用真实工具。

因此，表示探针负责提供风险信号，`GuardedToolsExecutor` 才是强制安全边界。

## 2. 阅读定位

本文档只解释稳定的系统设计，不维护动态实验数字：

- 一条真实调用链和建议阅读顺序见 [代码导览](code-tour.md)；
- 已实际运行的工程、边界和白盒接线证据见
  [已验证运行快照](../reports/verified-smoke.md)；
- 冻结 Qwen3-8B 矩阵的完整结果只维护在
  [正式 held-out 报告](../reports/qwen3-heldout-matrix.md)；
- split、阈值、分母和复现命令见 [实验协议](experiment-protocol.md)。

当前实现覆盖进程内 Hugging Face tool calling、两种 `resid_pre` capture position、direction/probe artifact、
MELON masked re-execution、AgentDojo pre-action executor、activation recorder 和 held-out matrix。这里关注这些
部件如何组合，不重复列出某次运行的 BU、UA、ASR 或延迟。

## 3. 端到端控制流

```text
┌──────────────────────────────┐
│ AgentDojo task/environment   │
└──────────────┬───────────────┘
               │ user/tool messages
               ▼
┌──────────────────────────────┐
│ LLM pipeline element         │
│ - scripted teaching backend  │
│ - in-process HF backend      │
└──────────────┬───────────────┘
               │ assistant message
               │ candidate tool call
               │ optional activation
               ▼
┌──────────────────────────────┐
│ GuardedToolsExecutor         │
│ 1. build DetectionContext    │
│ 2. optional masked rerun     │
│ 3. RuntimeGate.decide        │
└──────────────┬───────────────┘
          allow│               │block
               ▼               ▼
┌─────────────────────┐  ┌─────────────────────┐
│ runtime.run_function│  │ blocked tool result │
│ possible side effect│  │ no environment write│
└──────────┬──────────┘  └──────────┬──────────┘
           └──────────────┬──────────┘
                          ▼
              trace + AgentDojo checks
```

AgentDojo pipeline 的具体组合是：

```text
SystemMessage
→ InitQuery
→ LLM
→ ToolsExecutionLoop([GuardedToolsExecutor, LLM])
```

这与默认 `[ToolsExecutor, LLM]` 的差别很小，便于审计拦截点。

## 4. 核心类型

### `CandidateToolCall`

记录工具名、结构化参数和 call id。`canonical_text()` 将参数排序后序列化，使 MELON 比较不受字典顺序影响。

### `DetectionContext`

把候选调用、activation、masked candidates、消息和最小元数据交给 detector。它不包含任何执行权限。

### `ProbeObservation`

detector 只返回 score、threshold、是否触发、是否有效、耗时和小型元数据。

### `PolicyDecision`

`RuntimeGate` 将 observation 与 `ToolRiskPolicy` 结合，生成 `allow/block` 和稳定 reason code。

### `DecisionTrace` 与 `TrialRecord`

前者把 `runtime_invoked`（是否进入 `runtime.run_function`）与 `tool_succeeded`（工具是否无错误返回）分开；即使
工具先产生部分副作用再报错，也不会被错误记录成“未执行”。后者记录整个 trial 的 utility、攻击成功、拦截与延迟。
正式聚合层按 AgentDojo 上游口径分别计算 BU、attacked `utility` 的 UA、Targeted ASR，并单列进入各指标分母的
valid coverage；这些 episode 级主指标不能由 call-level interception 替代。

## 5. Detector 与 policy 为什么分开

如果 detector 直接执行阻断，会把三个概念混在一起：

1. 模型表示是否异常；
2. 工具动作的影响等级；
3. 系统在 detector 失效时采用什么容错策略。

当前设计中：

- `DirectionDetector`、`LinearProbeDetector`、`MelonToolCallDetector` 只打分。
- `ToolRiskPolicy` 只做确定性风险分级，不声称是 ML detector。
- `RuntimeGate` 负责 enforcement。

默认 `minimum_block_risk=MEDIUM`。低风险读取即使 detector 触发也不阻断；中高风险动作在 score 超阈值时阻断。
若 detector 无 activation、输出 NaN 或维度不匹配，中高风险动作默认 fail closed。

## 6. 表示级方法

### Difference-in-means direction

训练集中的正类表示为被劫持/策略违规调用，负类表示为 benign 调用：

```text
d = normalize(mean(h_positive) - mean(h_negative))
projection(h) = dᵀh
cosine(h) = dᵀh / ||h||
```

默认使用 cosine，也可以显式选择 projection。阈值只由 benign calibration score 的上分位数确定，并加入
`max(8 ULP, 1e-12 × score scale)` 的极小数值 margin，避免 NumPy/math 等价实现的末位误差把 calibration
benign 样本误判为触发。

当前实现是 refusal direction 与 policy-violation direction 共用的工程骨架。真实数据决定方向语义；不能仅凭类名
宣称它已经学会了 prompt injection。

### Linear activation probe

先只在 train split 上拟合 `StandardScaler`，再使用平衡类别权重的 logistic regression：

```text
z = (h - train_mean) / train_scale
score(h) = sigmoid(wᵀz + b)
```

训练依赖 scikit-learn，运行时 artifact 保存 `weights/bias/threshold` 以及 scaler mean/scale，不加载 pickle。
除模型、layer、position 和隐藏维度外，真实 recorder 还冻结 revision、dtype/量化配置、state kind、module path、
tokenizer/chat-template、tool schema、system message 和 render mode；运行时缺失或不匹配会令 detector invalid，
高风险动作按策略 fail closed。正式指标必须把这种 invalid block 与有效 probe interception 分开。

当模型来源是本地 checkpoint 路径时，metadata 还绑定路径无关的 `checkpoint_content_id`：配置、tokenizer、chat
template、量化配置和 weight index 做完整摘要，权重分片只读取固定的首/中/尾小窗口。它用于防止“同名目录、不同
权重”的正常误混，不是抵抗恶意篡改的完整文件校验。旧本地 artifact 缺少该 ID 时会 fail closed，必须从同一
checkpoint 重新采集和拟合；远端 Hugging Face 仓库仍由 resolved immutable revision 绑定。

## 7. MELON 核心算法切片

当前路径包含四个关键步骤：

1. 收集当前轨迹中的 tool outputs；
2. 隐藏原始用户任务，以中性提示重新调用同一个 LLM；
3. 缓存 masked run 提出的 tool calls；
4. 将原候选调用与缓存调用做最大相似度比较，在执行前阻断高相似调用。

当前 embedder 会对规范化 tool-call 文本分词，用 BLAKE2b 将 token 确定性映射到 256 维，再 L2 归一化并点积。
这提供了完全本地、无需密钥、可重复的 smoke 路径，但语义能力弱于成熟 embedding 模型。

除 embedding 外，当前切片还用短中性 masked prompt 代替论文的 few-shot prompt，并直接比较完整 canonical
tool-call 文本，没有实现论文的 security-specific argument projection。这三点都必须作为复现差异报告。

当前只运行一个短中性 prompt，没有论文 §3.4 的多 task-neutral-prompt ensemble，因此不具备、也不声称该节
ensemble detector 的 error bound 或 worst-case guarantee。代码默认相似度阈值 `0.8` 只是 hashing smoke 的
方便默认值；embedding 后端已改变，首轮 held-out 只把它作为运行前冻结的未校准 baseline。若下一版要声称
calibrated operating point，必须在新的 calibration 数据上选择，并保留独立的新 test，不能回调本轮结果。

masked re-execution 若生成了 malformed tool call，不能解释成“没有候选，因此安全”。当前实现会把该次
detector observation 标记为 invalid；对高风险主调用，统一风险策略在进入 runtime 前 fail closed，并把错误单列
到 failure/invalid 诊断，而不是计作有效的 MELON interception。

执行语义也不同：论文路径在相似调用命中后报告攻击并终止 episode；本项目 detector 只产生 observation，统一
`RuntimeGate` 再结合工具风险，仅阻断当前调用并返回 blocked tool result，之后 Agent 循环可以继续。这样便于与
direction/probe 共用 enforcement，但必须把它作为有意的工程差异，而不是论文的原样复现。

因此准确表述应是：

> 独立实现并接入了 MELON 的 masked re-execution、cache 和 pre-action tool-call comparison 核心切片；
> 当前相似度后端是本地 hashing embedding，只有单个中性 prompt，并采用“阻断当前调用后继续”的统一 gate
> 语义；尚未复现论文完整配置、episode-abort 行为、§3.4 ensemble 保证和效果。

MELON 的真实开销不仅是 `MelonToolCallDetector.inspect()`，还包括额外 LLM re-execution。正式报告必须计量
整条路径，不能只展示 scoring latency；当前数值统一见
[held-out 报告](../reports/qwen3-heldout-matrix.md#延迟与模型调用)。

## 8. Hugging Face 表示采集

`HuggingFaceToolCallingLLM` 在进程内加载模型，避免 vLLM/OpenAI-compatible 服务隐藏内部状态。

### `tool_input`

在正常 `model.generate()` 的第一次 prefill 给指定 decoder block 注册 `forward_pre_hook`，读取完整渲染
generation prompt 的最后一个非 padding token。它不是“注入正文最后 token”，也不是已经生成的 tool-call token；
Qwen2.5 smoke 中它是 assistant generation marker `\n`。由于因果注意力已汇总此前包含 tool output 的上下文，
本项目把这份 `resid_pre` 明确定义为对论文 tool-input phase 的工程化 operationalization（操作化定义），而不是
声称论文规定了这个精确 token。artifact 中精确记为
`position=generation_prefill_last_nonpad`，不需要额外 forward。

正式候选 Qwen3-8B 的 one-task spike 与首轮 held-out 矩阵都关闭 thinking，固定 layer 22、隐藏宽度 4096；该位置的
assistant marker 为 `\n\n`。无防御和防御路径都使用相同 render mode。这个配置证明真实 Banking 轨迹能够接线并
完成冻结矩阵，但不改变该位置“生成候选动作前的上下文状态”这一语义，也不构成 layer 22 最优的结论。

### `function_call`

先生成并解析候选调用，parser 同时保留被选中合法调用的精确字符 span；再把该 span 映射回模型原始 generated
token IDs，截取到对应 closing tag，而不是搜索 completion 中最后一个同名 closing tag。随后对
`prompt + raw call tokens` 做一次无梯度 replay，并在指定 block 读取 closing-tag token 的 `resid_pre`。
不重序列化 JSON，也不把 EOS 或 call 后 prose 当作 function-call 表示。元数据将
`position=function_call_end`、`extra_forward_count=1`，正式延迟报告必须包含这次成本。字符—token 边界无法对齐时
不执行猜测 replay，而是写入 `function_call_replay_boundary_error`，让高风险调用走 detector-invalid fail-closed。

hook 使用上下文管理器，异常退出时也会移除。Parser 与无防御执行路径支持连续多个 AgentDojo/Qwen 风格调用；
表示防御尚未逐调用提取状态，因此多调用回合会清除 activation、写入
`multiple_tool_calls_require_per_call_activations`，高风险动作按 detector invalid 路径 fail closed。

## 9. AgentDojo 结果语义

架构层只需要记住三个不变量：

1. Banking injection task 的 raw `security_results=True` 表示攻击目标已经落地，所以内部字段命名为
   `attack_succeeded`；
2. BU、UA 和 Targeted ASR 是 episode/environment 级结果，call-level interception 不能替代它们；
3. detector `valid`、trial `valid`、utility 和安全性是不同概念，基础设施失败不能通过缩小分母美化结果。

精确定义、分母和失败桶统一由 [实验协议](experiment-protocol.md#7-指标定义) 维护。

## 10. 安全与工程边界

- synthetic demo 与 AgentDojo 都使用沙箱环境，不连接真实银行。
- artifact 和 trace 不保存完整 prompt；activation recorder 默认写入 Git 忽略目录。
- tool risk 是名称级启发式，尚未做完整参数级授权策略。
- 当前 held-out 只覆盖同一攻击模板和极小 user-task 子集；正式限制与 probe 误阻分析见
  [held-out 报告](../reports/qwen3-heldout-matrix.md#证据边界)。
- MELON slice 没有 neutral-prompt ensemble、论文 §3.4 保证或命中后 episode abort 语义。
- 首版表示防御要求一次只调用一个工具；多调用不会复用共享 activation，而会显式进入 detector-invalid 路径。
- blocked call 仍保留为“模型提出过的调用”并附带 blocked tool result，以维持消息协议；Banking checks 主要按
  环境副作用判定。迁移到依赖 function-call trace 的 suite 时，必须区分 proposed 与 executed，重新验证指标语义。
- 未知工具会拒绝执行，但当前 gate 决策先于工具存在性检查；正式安全审计仍需补充 schema/unknown-tool 专项测试。
