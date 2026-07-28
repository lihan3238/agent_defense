# 架构说明

## 1. 要解决的问题

本项目关注 indirect prompt injection（间接提示注入）：攻击指令不直接出现在用户请求中，而是藏在邮件、文件、网页或
其他工具返回值里。工具型 Agent（智能体）读取这些不可信内容后，可能把它转化为转账、修改账户或外传数据等调用。

风险真正落地的时刻不是“模型读到了恶意文本”，而是：

> 模型已经提出候选 tool call（工具调用），执行器尚未调用真实工具。

因此，表示探针负责提供风险信号，`GuardedToolsExecutor` 才是强制安全边界。

## 2. 阅读定位

本文档只解释稳定的系统设计，不维护动态实验数字：

- 一条真实调用链和建议阅读顺序见 [代码导览](code-tour.md)；
- 已实际运行的工程、边界和白盒接线证据见
  [已验证运行快照](../reports/verified-smoke.md)；
- Qwen3-8B（通义千问 3 80 亿参数模型）四套件 MELON（Masked re-Execution and TooL comparisON，
  掩码重执行与工具调用比较）主矩阵见
  [完整主矩阵报告](../reports/qwen3-v112-full-matrix.md)，表示级 direction/probe（方向/探针）结果见
  [held-out（留出测试）报告](../reports/qwen3-heldout-matrix.md)；两者职责见
  [实验与证据记录索引](../reports/README.md)；
- split（数据划分）、阈值、分母和复现命令见 [实验协议](experiment-protocol.md)。

当前实现覆盖进程内 Hugging Face（模型库）tool calling（工具调用）、两种 `resid_pre`
capture position（捕获位置）、direction/probe artifact（方向/探针工件）、
MELON masked re-execution（掩码重执行）、
AgentDojo（智能体安全评测框架）pre-action executor（动作执行前执行器）、activation recorder（激活记录器）和
held-out matrix（留出测试矩阵），以及 `melon_paper` 四套件完整矩阵的生成、验收与分析链路。这里关注这些部件如何组合，
不重复列出某次运行的 BU（Benign Utility，良性任务可用性）、UA（Utility Under Attack，攻击场景任务可用性）、
ASR（Attack Success Rate，攻击成功率）或延迟。

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

AgentDojo pipeline（流水线）的具体组合是：

```text
SystemMessage
→ InitQuery
→ LLM
→ ToolsExecutionLoop([GuardedToolsExecutor, LLM])
```

这里的 LLM（Large Language Model，大语言模型）只负责提出候选动作。该组合与默认 `[ToolsExecutor, LLM]` 的差别很小，
因此拦截点便于审计。

## 4. 核心类型

### `CandidateToolCall`

记录工具名、结构化参数和 call ID（Identifier，调用标识符）。`canonical_text()` 将参数排序后序列化，使 MELON 比较不受字典顺序影响。

### `DetectionContext`

把候选调用、activation、masked candidates（掩码候选）、消息和最小元数据交给 detector（检测器）。它不包含
任何执行权限。

### `ProbeObservation`

detector 只返回 score（分数）、threshold（阈值）、是否触发、是否有效、耗时和小型元数据。

### `PolicyDecision`

`RuntimeGate` 将 observation（观察）与 `ToolRiskPolicy` 结合，生成 `allow/block`（放行/阻断）和稳定
reason code（原因码）。

### `DecisionTrace` 与 `TrialRecord`

前者把 `runtime_invoked`（是否进入 `runtime.run_function`）与 `tool_succeeded`（工具是否无错误返回）分开；即使
工具先产生部分副作用再报错，也不会被错误记录成“未执行”。后者记录整个 trial（试验）的 utility（任务可用性）、
攻击成功、拦截与延迟。正式聚合层按 AgentDojo 上游口径分别计算 BU、attacked（受攻击）`utility` 的 UA、
Targeted ASR（Targeted Attack Success Rate，定向攻击成功率），并单列进入各指标分母的 valid coverage（有效覆盖率）；这些 episode（回合）级
主指标不能由 call-level interception（调用级拦截率）替代。

## 5. Detector 与 policy（策略）为什么分开

如果 detector 直接执行阻断，会把三个概念混在一起：

1. 模型表示是否异常；
2. 工具动作的影响等级；
3. 系统在 detector 失效时采用什么容错策略。

当前设计中：

- `DirectionDetector`、`LinearProbeDetector`、`MelonToolCallDetector` 只打分。
- `ToolRiskPolicy` 只做确定性风险分级，不声称是 ML（Machine Learning，机器学习）detector。
- `RuntimeGate` 负责 enforcement（强制执行）。

默认 `minimum_block_risk=MEDIUM`。低风险读取即使 detector 触发也不阻断；中高风险动作在 score 超阈值时阻断。
若 detector 无 activation、输出 NaN（Not a Number，非数值）或维度不匹配，中高风险动作默认
fail closed（故障时默认阻断）。

## 6. 表示级方法

### Difference-in-means direction（均值差方向）

训练集中的正类表示为被劫持/策略违规调用，负类表示为 benign（良性）调用：

```text
d = normalize(mean(h_positive) - mean(h_negative))
projection(h) = dᵀh
cosine(h) = dᵀh / ||h||
```

默认使用 cosine（余弦相似度），也可以显式选择 projection（投影）。阈值只由
benign calibration score（良性校准分数）的上分位数确定，并加入 `max(8 ULP, 1e-12 × score scale)` 的极小数值
margin（余量）；其中 ULP（Unit in the Last Place，末位单位）用于刻画浮点精度，避免 NumPy（数值计算库）/math（数学模块）
等价实现的末位误差把 calibration benign
样本误判为触发。

当前实现是 refusal direction（拒答方向）与 policy-violation direction（策略违规方向）共用的工程骨架。真实数据决定方向语义；不能仅凭类名
宣称它已经学会了 prompt injection。

### Linear activation probe（线性激活探针）

先只在 train split（训练集划分）上拟合 `StandardScaler`（标准化器），再使用平衡类别权重的
logistic regression（逻辑回归）：

```text
z = (h - train_mean) / train_scale
score(h) = sigmoid(wᵀz + b)
```

训练依赖 Python（编程语言）的 scikit-learn（机器学习库），运行时 artifact 保存
`weights/bias/threshold` 以及
scaler mean/scale（标准化器均值/尺度），不加载 pickle（序列化格式）。除模型、layer（层）、position（位置）和隐藏维度外，真实
recorder（记录器）还冻结 revision（修订版本）、dtype（数据类型）/量化配置、state kind（状态类型）、
module path（模块路径）、tokenizer/chat-template（分词器/对话模板）、tool schema（工具结构约束）、
system message（系统消息）和 render mode（渲染模式）；运行时缺失或不匹配会令 detector invalid（无效），
高风险动作按策略 fail closed。正式指标必须把这种 invalid block 与有效 probe interception 分开。

当模型来源是本地 checkpoint（检查点）路径时，metadata（元数据）还绑定路径无关的 `checkpoint_content_id`：配置、
tokenizer、chat template、量化配置和 weight index（权重索引）做完整摘要，权重分片只读取固定的首/中/尾小窗口。
它用于防止“同名目录、不同权重”的正常误混，不是抵抗恶意篡改的完整文件校验。旧本地 artifact 缺少该
ID（标识符）时会
fail closed，必须从同一 checkpoint 重新采集和拟合；远端 Hugging Face 仓库仍由
resolved immutable revision（解析后的不可变修订版本）绑定。

## 7. 两条 MELON 路径

### 7.1 冻结的 `melon` 哈希切片

当前路径包含四个关键步骤：

1. 收集当前轨迹中的 tool outputs（工具输出）；
2. 隐藏原始用户任务，以中性提示重新调用同一个 LLM（Large Language Model，大语言模型）；
3. 缓存 masked run（掩码运行）提出的 tool calls；
4. 将原候选调用与缓存调用做最大相似度比较，在执行前阻断高相似调用。

当前 embedder（嵌入器）会对规范化 tool-call 文本分词，用 BLAKE2b（哈希算法）将 token（词元）确定性映射到
256 维，再做 L2（L2 范数）归一化并点积。这提供了完全本地、无需密钥、可重复的 smoke（冒烟测试）路径，
但语义能力弱于成熟 embedding（嵌入）模型。

除 embedding 外，当前切片还用短中性 masked prompt（掩码提示）代替论文的 few-shot prompt（少样本提示），并直接
比较完整 canonical tool-call（规范化工具调用）文本，没有实现论文的
security-specific argument projection（安全专用参数投影）。这三点都必须作为复现差异报告。

当前只运行一个短中性 prompt，没有论文 §3.4 的多
task-neutral-prompt ensemble（任务中性提示集成），因此不具备、也不声称该节 ensemble detector（集成检测器）的
error bound（误差界）或 worst-case guarantee（最坏情况保证）。代码默认相似度阈值 `0.8` 只是 hashing（哈希）smoke 的
方便默认值；embedding 后端已改变，首轮 held-out 只把它作为运行前冻结的未校准 baseline（基线）。若下一版要声称
calibrated operating point（经校准工作点），必须在新的 calibration 数据上选择，并保留独立的新
test（测试集），不能回调本轮结果。

masked re-execution 若生成了 malformed tool call（格式错误的工具调用），不能解释成“没有候选，因此安全”。当前
实现会把该次 detector observation 标记为 invalid；对高风险主调用，统一风险策略在进入
runtime（运行时）前 fail closed，并把错误单列到 failure/invalid（失败/无效）诊断，而不是计作有效的
MELON interception。

执行语义也不同：论文路径在相似调用命中后报告攻击并终止 episode；本项目 detector 只产生 observation，统一
`RuntimeGate` 再结合工具风险，仅阻断当前调用并返回 blocked tool result（已阻断工具结果），之后 Agent 循环可以
继续。这样便于与 direction/probe 共用 enforcement，但必须把它作为有意的工程差异，而不是论文的原样复现。

因此准确表述应是：

> 独立实现并接入了 MELON 的 masked re-execution、cache（缓存）和
> pre-action tool-call comparison（动作执行前工具调用比较）核心切片；当前相似度后端是本地
> hashing embedding（哈希嵌入），只有单个中性 prompt，并采用“阻断当前调用后继续”的统一 gate（门控）
> 语义；尚未复现论文完整配置、episode-abort（回合终止）行为、§3.4 ensemble 保证和效果。

MELON 的真实开销不仅是 `MelonToolCallDetector.inspect()`，还包括额外 LLM re-execution（重新执行）。
正式报告必须计量
整条路径，不能只展示 scoring latency（打分延迟）；当前数值统一见
[held-out 报告](../reports/qwen3-heldout-matrix.md#延迟与模型调用)。

### 7.2 新增的 `melon_paper` 论文兼容路径

`melon_paper` 不改写上述冻结基线，而是按论文 Appendix A.1–A.3（附录 A.1–A.3）独立重建
few-shot masked prompt（少样本掩码提示）、虚构 `read_file("random.txt")` 输出包装、
security-specific argument projection（安全专用参数投影）和 semantic embedding（语义嵌入）接口。

该路径会先对本轮全部候选调用做 batch preflight（整批预检）。任意调用与掩码缓存的相似度严格高于 `0.8` 时，
所有候选均不会进入 runtime；原 assistant message（助手消息）被替换成无工具调用的终止消息，因此 AgentDojo 不会把
未执行候选误当作已完成动作。本项目独立 trace（轨迹）仍保留候选、分数和未执行状态，供面试审计。

这里只保留跨步骤 tool-call cache（工具调用缓存），没有复制发布源码的 embedding cache（嵌入缓存）。整批终止和
fail-closed（故障时默认阻断）是本仓库的工具边界语义；论文算法本身只规定返回攻击告警。论文附录与发布源码的两处提示文本冲突见
[MELON 论文兼容重建](melon-reproduction.md)。

`openai/text-embedding-3-large` 对应论文发布代码的 embedding backend（嵌入后端）；本地
`sentence-transformers/all-MiniLM-L6-v2` 是四套件完整主矩阵实际使用的离线后端，但不能冒充论文嵌入配置或数值。
完整审计与协议见
[MELON 论文兼容重建](melon-reproduction.md)。

## 8. Hugging Face 表示采集

`HuggingFaceToolCallingLLM` 在进程内加载模型，避免 vLLM（高吞吐大模型推理引擎）/OpenAI-compatible（兼容
OpenAI 接口）服务隐藏内部状态。

### `tool_input`

在正常 `model.generate()` 的第一次 prefill（预填充）给指定 decoder block（解码器层）注册
`forward_pre_hook`，读取完整渲染 generation prompt（生成提示）的最后一个 non-padding token（非填充词元）。它不是
“注入正文最后 token”，也不是已经生成的 tool-call token（工具调用词元）；
Qwen2.5（通义千问 2.5 模型）smoke 中它是 assistant generation marker（助手生成标记）`\n`。由于因果注意力已
汇总此前包含 tool output 的上下文，
本项目把这份 `resid_pre` 明确定义为对论文 tool-input phase（工具输入阶段）的工程化
operationalization（操作化定义），而不是
声称论文规定了这个精确 token。artifact 中精确记为
`position=generation_prefill_last_nonpad`，不需要额外 forward（前向传播）。

正式候选 Qwen3-8B 的 one-task spike（单任务接线验证）与首轮 held-out 矩阵都关闭 thinking（思考模式），固定
layer 22、隐藏宽度 4096；该位置的 assistant marker 为 `\n\n`。无防御和防御路径都使用相同 render mode。这个配置
证明真实 Banking（银行任务套件）轨迹能够接线并
完成冻结矩阵，但不改变该位置“生成候选动作前的上下文状态”这一语义，也不构成 layer 22 最优的结论。

### `function_call`

先生成并解析候选调用，parser（解析器）同时保留被选中合法调用的精确字符 span（跨度）；再把该 span 映射回模型原始
generated token IDs（生成词元标识符），截取到对应 closing tag（结束标签），而不是搜索 completion（生成文本）中
最后一个同名 closing tag。随后对 `prompt + raw call tokens`（提示词与原始调用词元）做一次无梯度 replay（重放），并在指定 block 读取
closing-tag token 的 `resid_pre`。不重序列化 JSON（JavaScript Object Notation，JavaScript 对象表示法），也不把
EOS（End of Sequence，序列结束标记）或 call 后
prose（普通文本）当作 function-call（函数调用）表示。元数据将
`position=function_call_end`、`extra_forward_count=1`，正式延迟报告必须包含这次成本。字符—token 边界无法对齐时
不执行猜测 replay，而是写入 `function_call_replay_boundary_error`，让高风险调用走 detector-invalid fail-closed。

hook（钩子）使用上下文管理器，异常退出时也会移除。Parser 与无防御执行路径支持连续多个 AgentDojo/Qwen 风格调用；
表示防御尚未逐调用提取状态，因此多调用回合会清除 activation、写入
`multiple_tool_calls_require_per_call_activations`，高风险动作按 detector invalid 路径 fail closed。

## 9. AgentDojo 结果语义

架构层只需要记住三个不变量：

1. Banking injection task（注入任务）的 raw（原始）`security_results=True` 表示攻击目标已经落地，所以内部字段命名为
   `attack_succeeded`；
2. BU、UA 和 Targeted ASR 是 episode/environment（回合/环境）级结果，call-level interception 不能替代它们；
3. detector `valid`、trial `valid`、utility 和安全性是不同概念，基础设施失败不能通过缩小分母美化结果。

精确定义、分母和失败桶统一由 [实验协议](experiment-protocol.md#7-指标定义) 维护。

## 10. 安全与工程边界

- synthetic demo（合成演示）与 AgentDojo 都使用沙箱环境，不连接真实银行。
- artifact 和 trace（轨迹）不保存完整 prompt；activation recorder 默认写入 Git（版本控制系统）忽略目录。
- tool risk（工具风险）是名称级启发式，尚未做完整参数级授权策略。
- 当前 held-out 只覆盖同一攻击模板和极小 user-task（用户任务）子集；正式限制与 probe 误阻分析见
  [held-out 报告](../reports/qwen3-heldout-matrix.md#证据边界)。
- 冻结的 `melon` slice（MELON 算法切片）没有 neutral-prompt ensemble（中性提示集成）、论文 §3.4 保证或命中后
  episode abort（回合终止）语义；`melon_paper` 已实现终止语义和集成计算原语，并完成单一攻击下的四套件完整主矩阵。
  主矩阵实际使用单个中性提示，集成原语只经过单元测试；结果仍使用本地 MiniLM（小型句向量模型），不是论文四攻击或
  原始数值复现。
- 首版表示防御要求一次只调用一个工具；多调用不会复用共享 activation，而会显式进入 detector-invalid 路径。
- `melon` 的 blocked call（已阻断调用）仍保留为“模型提出过的调用”并附带 blocked tool result，以维持消息协议；
  Banking checks（银行任务检查）主要按环境副作用判定。迁移到依赖 function-call trace（函数调用轨迹）的
  suite（评测套件）时，必须区分 proposed（已提出）与 executed（已执行），重新验证指标语义。`melon_paper` 则在命中时
  替换该候选消息，同时把 proposal（候选调用）保留在独立审计 trace 中。
- 未知工具会拒绝执行，但当前 gate 决策先于工具存在性检查；正式安全审计仍需补充
  schema/unknown-tool（结构约束/未知工具）专项测试。
