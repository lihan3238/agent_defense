# 面试讲解指南

## 先记住这句定位

> 我完成了一个 AgentDojo（智能体安全评测框架）上的执行前 tool-call gate（工具调用门控）、进程内隐藏状态采集、
> direction/probe（方向/探针）与 MELON（掩码重执行检测方法）对照，以及 Qwen3-8B（通义千问 3 80 亿参数模型）的
> 首轮小型 held-out（留出测试）工程评测；价值是可审计闭环和失败分析，不是统计显著或
> 跨攻击模板泛化。

面试中只维护一个数字来源：打开
[`reports/qwen3-heldout-matrix.md`](../reports/qwen3-heldout-matrix.md) 展示正式结果。One-task（单任务）接线历史和
synthetic（合成）/真实证据边界见
[`reports/verified-smoke.md`](../reports/verified-smoke.md)。不要凭记忆另造一张结果表。

Qwen3-30B-A3B（通义千问 3，300 亿总参数/30 亿激活参数模型）只有
[screening（筛选实验）补充证据](../reports/qwen3-30b-screening.md)，不进入正式防御效果表。

## 两分钟主叙事

可以直接按下面的顺序讲：

> 我做的是工具型 Agent（智能体）的运行时 prompt-injection（提示注入）防御。威胁不是用户直接越狱，而是 Agent 从文件、
> 邮件或网页里读到攻击指令，随后产生越权工具调用。
>
> 我把强制边界放在候选 tool call 已生成、但工具尚未执行的位置。`tool_input` 信号取完整
> generation prefill（生成预填充）最后一个 token（词元）的隐藏状态，候选调用生成后再把它和实际动作风险送入 gate。模型适配层输出调用和
> 隐藏状态，detector（检测器）用 difference-in-means（均值差）direction 或
> logistic activation probe（逻辑回归激活探针）打分；独立的 policy（策略）再结合工具风险，决定 allow（放行）或
> block（阻断）。这样探针即使失效，也不会
> 绕过执行器的 schema（结构约束）和风险策略。
>
> 评测底座用 AgentDojo，因为它提供真实的多步工具环境以及确定性的 utility（任务可用性）和攻击目标检查。我也实现了
> MELON 的 masked re-execution（掩码重执行）、tool-call cache（工具调用缓存）和动作比较切片，当前用本地
> hashing embedding（哈希嵌入）保证离线可审计。
>
> 工程边界先在 deterministic（确定性）synthetic 闭环中验证；随后我固定 Qwen3-8B、layer（层）22 和
> tool-input position（工具输入位置），先跑通一个真实 Banking（银行任务套件）clean/attack pair（无攻击/受攻击配对），
> 再冻结 `user_task_1/11/13` 做 30-episode（30 回合）held-out 矩阵。30 个 trial（试验）全部有效。无防御
> Targeted ASR（定向攻击成功率）是 1/3，direction、activation probe、MELON slice（MELON 算法切片）和
> `repeat_user_prompt` 都是 0/3；但所有方法 BU（良性任务可用性）都只有 1/3，activation probe 的
> UA（攻击场景任务可用性）还是 0/3。
>
> 所以我没有只看 ASR。逐调用人工审计发现 direction 和 MELON 确实各阻断了唯一一条恶意 proposal（候选调用）；activation
> probe 根本没遇到恶意 proposal，而是提前误阻合法 `update_user_info`，不能把它算成成功拦截。配对中位开销上，
> direction 约 2 ms（毫秒），probe 约 8 ms，MELON 约 3.8 s（秒），`repeat_user_prompt` 约 3.46 s。
>
> 这只能说明同一攻击模板下跨三个 user task（用户任务）的小型工程结果。如果继续扩展，就要换预先冻结的新攻击模板和
> 新 test groups（测试集分组），不能根据这次 test（测试集）结果回头调阈值再重报。

这里可以补一句工程细节：矩阵在正式计时前先运行并丢弃一次 warm-up（预热），随后复用同一个进程内模型；每个 trial 立即写
JSONL（逐行 JSON，JSON 为 JavaScript 对象表示法），异常进入 failure bucket（失败桶）。调用级 interception（拦截率）/
false-block（误阻率）只有人工审核标签完整时才出数，否则是 `N/A`（不适用）。

## 两分钟现场演示

现场只运行一个单进程命令，依次展示无防御攻击、probe 阻断攻击、probe 放行正常任务：

```bash
uv run agent-defense interview-demo
```

依次指出：

```text
no_defense_attacked: attacker recipient, decision=allow, executed=true, attack_goal_achieved=true
probe_attacked:      same recipient, decision=block, executed=false, attack_goal_achieved=false
probe_clean:         authorized recipient, decision=allow, executed=true, utility=true
```

若时间够，再展示矩阵：

```bash
uv run agent-defense eval-demo
```

真实 held-out 不在现场重新加载 8B 模型。用 manifest（运行清单）命令展示实验规模，再直接打开已保存的去敏报告：

```bash
uv run agent-defense matrix-plan examples/qwen3-heldout-matrix.example.json
```

重点展示 `reports/qwen3-heldout-matrix.md`，而不是现场重跑 30 episodes。

开场必须主动说一句：“这里是 deterministic synthetic teaching demo（确定性合成教学演示），展示控制流，不把数值当
真实模型实验结果。”

## 真实模型证据怎么展示

不要现场重新跑 8B 模型。打开
[`reports/qwen3-heldout-matrix.md`](../reports/qwen3-heldout-matrix.md)，按下面顺序指表：

1. 先看 Scheduled/Valid（计划数/有效数）：30/30 有效，说明分母没有被基础设施失败缩小；
2. 再看 BU/UA/Targeted ASR：安全结果和任务完成结果必须分列；
3. 再看人工调用审计：direction/MELON 的 1/1 与 probe 的 0 个恶意 proposal 是完全不同的证据；
4. 最后看 paired overhead（配对额外开销）：MELON 的主要成本来自额外 masked generation（掩码生成），不是相似度
   dot product（点积）。

若面试官追问开发过程，再打开
[`reports/verified-smoke.md`](../reports/verified-smoke.md) 解释证据阶梯：

- one-task spike（单任务接线验证）证明真实 tool call、hidden state（隐藏状态）和 gate 接线；
- held-out matrix（留出测试矩阵）才是独立 user-task 的首轮结果；
- 两者都只覆盖一个攻击模板，不能支持统计泛化。

最值得主动讲的失败是 calibration coverage（校准覆盖）：只用只读 benign（良性）调用会误阻合法 `send_money`；加入合法高风险
付款后，one-task spike 能放行正常付款，但 held-out 又暴露 probe 会误阻合法 `update_user_info`。结论不是某个
阈值“最优”，而是 calibration 必须覆盖多类正常高风险动作，而且观察过的 test 不能拿来反调本轮阈值。

### 跨模型 30B 复核目前怎么说？

[跨模型复核协议](cross-model-replication.md) 固定了 Qwen3-30B-A3B 的 revision（修订版本）、layer、
task groups（任务组）和 continuation gate（继续实验门槛）。白盒 smoke（冒烟测试）
成功捕获 layer 29 的 `float32[2048]` 状态，clean/attacked（无攻击/受攻击）两个 screening trial 也都 valid（有效）、无
parse/runtime failure（解析/运行时故障）。Attacked trial 的 exact Targeted ASR（精确定向攻击成功率）是 `0/1`，
但人工 trace（轨迹）审核发现
模型仍执行并成功完成了一笔 injection-driven（注入驱动）未授权转账，只是收款参数与预注册 exact target（精确目标）
近似而不相等。

最值得讲的不是“30B ASR 为 0”，而是 **exact-target metric（精确目标指标）与真实有害副作用可能分离**。项目
continuation gate 要求至少一次 exact no-defense success（无防御条件下精确攻击成功）；门槛未满足后，本轮停止，没有采集
30B train/calibration（训练/校准）activation、拟合
probe 或打开 held-out。正式防御效果仍只看 Qwen3-8B 矩阵。

## 简历逐句对应

完整的四列证据映射见 [简历证据对齐](resume-evidence.md)。下面只保留现场速查：

| 简历表述 | 仓库证据 | 当前诚实状态 |
|---|---|---|
| 基于 AgentDojo 构建工具调用 Agent | Banking suite（银行任务套件）、pipeline（流水线）和 deterministic checks（确定性检查） | 边界、Qwen3 接线和 30-episode held-out 已跑通 |
| 每次 tool-call 前检测 | `GuardedToolsExecutor` 在 `runtime.run_function` 前调用 gate | 单调用表示路径已验证；多调用回合没有独立 per-call（逐调用）activation，会 invalid（无效）/fail closed（故障时默认阻断） |
| refusal-direction（拒答方向）/激活探针 | direction 与 linear probe（线性探针）artifact（工件）/detector | 正式 direction 是 policy-violation direction（策略违规方向）；probe 暴露合法调用误阻；样本仍很小 |
| 读取本地开源模型隐藏状态 | 进程内 HF（Hugging Face 模型库）adapter（适配器）和 `resid_pre` hook（钩子） | 0.5B smoke、Qwen3 spike 与 held-out 都已验证 |
| OWASP（开放全球应用程序安全项目）LLM01（大语言模型应用安全风险第 1 项）prompt injection | AgentDojo untrusted tool-output injection（不可信工具输出注入） | 覆盖 indirect-injection（间接注入）子集和一个正式攻击模板，不是完整 LLM01 |
| 检测与拦截危险工具调用 | detector + name-level risk policy（工具名级风险策略）+ executor trace（执行器轨迹） | direction/MELON 有执行前审计；尚不是参数级授权系统 |
| utility/security（任务可用性/安全性）双指标与内置对照 | AgentDojo checks + matrix aggregator（矩阵聚合器） | 已报告 BU、UA、ASR；built-in（内置方法）的 call-level（调用级）指标为 N/A |
| ASR、拦截率、utility、开销 | matrix 聚合与人工 call review（调用审核） | 已保留原始计数；小样本、seed（随机种子）0、模型加载不进入 episode 计时 |

## 高频追问

### 为什么选 tool-call boundary（工具调用边界）？

因为 prompt injection 本身只是输入，真正的安全后果发生在动作执行。这个边界同时满足两个条件：上下文足够完整，
而副作用尚未发生。相比只改 system prompt（系统提示词），它还是一个独立的强制执行点。

### 为什么用 AgentDojo，而不是自造 demo（演示）或 LLM judge（大语言模型评审器）？

它同时提供多步 indirect prompt injection（间接提示注入）、沙箱工具环境和确定性的 utility/attack-goal checks
（任务可用性/攻击目标检查）。LLM judge 可以做
补充分析，但不适合替代可重复的主指标；自造 demo 只保留作控制流回归，不能支撑效果结论。

### 如何证明 block 真的发生在副作用之前？

边界测试 instrument（插桩）了 `runtime.run_function`，同时比较执行前后环境，并把候选调用拆成 `proposed` 与 `executed`。
无防御时恶意 `update_password` 出现在 runtime audit（运行时审计）且环境改变；probe 时它仍在
decision trace（决策轨迹），但
`executed=false`、runtime audit 中不存在该调用、环境不变。

### 为什么 detector 和 policy 要分开？

detector 只能说明表示是否异常，不能自动说明工具影响。读取余额和转账即使得到相同分数，也不应采用完全相同的
故障策略。分层后可以单测打分、风险分类和 enforcement（强制执行），也可以在 detector 不可用时对高风险动作
fail closed（故障时默认阻断）。

### `tool_input` 到底 hook 哪个 token、在什么时候？

它是 `model.generate()` 首次 prefill 时，完整 chat template（对话模板）加 generation prompt（生成提示词）后的最后一个
non-padding token（非填充词元）；Qwen smoke 中是 assistant marker（助手标记）`\n`。它通过因果注意力汇总了此前包含
不可信 tool output（工具输出）的上下文，但尚未包含生成出来的 call token（调用词元）。候选调用生成并解析后，gate 再把
这份 context state（上下文状态）与实际 tool risk（工具风险）结合。`function_call_end` 则沿原始 generated IDs
（生成词元编号），并 replay（重放）到 closing tag（结束标签），多一次 forward（前向传播）。

正式 Qwen3-8B spike 关闭 thinking（思考模式），固定 layer 22、宽度 4096；对应 generation marker 为 `\n\n`。token 的意义仍是
“生成前上下文汇聚位置”，不是“注入文本末 token”。

### 最大工程难点是什么？

AgentDojo 默认本地模型走 vLLM（高吞吐大语言模型推理引擎）/OpenAI-compatible（OpenAI 接口兼容）服务，服务边界拿不到
隐藏状态，所以增加了进程内 HF adapter。真正难点不是写一个 dot product，而是让 tool call、activation、chat template、
model revision（模型修订版本）、layer/position 和执行 trace 可对齐；artifact 因此绑定这些身份，运行前
preflight（预检），运行时仍做逐调用校验。

### 正负标签到底是什么？

标签是 call-level 的 aligned/benign（对齐/良性）与 hijacked-or-policy-violating（被劫持或违反策略）状态，不是把整条
attacked trajectory（受攻击轨迹）全标为 1。攻击轨迹里的正常检索和注入前步骤可能仍是负类，所以 attacked
recorder（记录器）先写 `label=null`，episode 后按
`activation_sample_id` 逐调用复核。

### Direction 与 logistic probe 有什么区别？

direction 是正负均值差，参数少、可解释、样本需求小，但只能表达一个线性方向。logistic probe 同样是线性的，
却能监督学习权重和 bias（偏置），通常更灵活，也更容易过拟合。因此阈值必须只在 calibration split（校准划分）上选。

### 这是 refusal direction 吗？

代码实现的是通用 difference-in-means direction 骨架。用 refusal 数据训练时可以叫 refusal direction；用
AgentDojo policy-compliant/violating（符合/违反策略）轨迹训练时更准确地叫 policy-violation direction。拒答倾向不等于注入检测能力。

### MELON 复现到了什么程度？

仓库现在保留两条路径。`melon` 是冻结结果使用的核心切片：短中性 prompt（提示词）、完整参数、本地
hashing embedding（哈希嵌入）和阻断后继续。新增的 `melon_paper` 按论文附录独立重建 few-shot masked prompt
（少样本掩码提示词）、虚构 `read_file` 输出包装、security-specific argument projection（安全特定参数投影）、
semantic embedding（语义嵌入）接口、跨步骤 cache（缓存）、整批执行前检查和命中后 episode abort（回合终止）。

这叫“算法与协议兼容重建”，还不能叫“论文效果复现”：官方源码没有许可证、版本锁、完整模型配置或原始日志，目前也尚未
完成论文三个模型、97/629 全量和四攻击的正式效果表。
这里的缓存只指跨步骤工具调用缓存，不包含官方源码的 embedding cache（嵌入缓存）；整批终止和 fail-closed（故障时默认阻断）
是本仓库的工程化边界。论文附录与发布源码在中性提示开头和最后一条少样本工具输出上不一致，本实现以论文附录为准。

最新 16 配对筛选共计划 64 回合，其中 60 个有效；有效 attacked trial（受攻击试验）中，`none` 的 Targeted ASR 为
`6/15`，`melon_paper` 为 `5/15`。唯一阻断在 runtime（运行时）前生效，但不匹配精确攻击参考调用，所以面试时应把它讲成
“控制流验证 + 负结果”，不能讲成已验证恶意调用拦截。唯一正式分母来源是
[`reports/melon-paper-screening.md`](../reports/melon-paper-screening.md)。

### MELON 的 `0.8` 是校准阈值吗？有论文的理论保证吗？

对 `melon` 都不是。`0.8` 是 hashing smoke 的默认值，来源于论文主实验的相似度阈值；换成 hashing embedding 后分数分布不同。
首轮 held-out 把它明确当作运行前冻结的未校准 baseline（基线）；若下一版要校准，必须使用新的 calibration 和独立 test，
不能拿本轮结果反调。`melon_paper` 的 `0.8` 与发布代码一致，并实现了 §3.4 的多 task-neutral-prompt ensemble
（任务中性提示词集成）计算原语，但尚未验证均值可分性和独立性前提，因此仍不能宣称继承 error bound（误差界）或
worst-case guarantee（最坏情况保证）。

### MELON 与表示探针的主要差异？

MELON 通过第二条轨迹问“去掉原任务后，不可信内容是否仍驱动相同动作”，语义直观但通常多一次模型调用。
表示探针只读一次隐藏状态，潜在开销低，但依赖模型、层、位置和训练分布，跨模型迁移更弱。

### 为什么不直接禁止转账？

正常用户任务也需要转账。工具风险决定 enforcement 强度，探针判断当前上下文是否异常。demo 中用户授权账单仍被允许，
攻击者账户的转账才被阻断。

### AgentDojo 的 security result（安全结果）怎么解释？

对 injection task（注入任务），`True` 表示攻击目标完成。因此代码将它改名为 `attack_succeeded`，ASR 直接统计 True 的比例。
只有在分母与无效 trial 规则固定后，才能另外报告 `1-ASR` 作为安全率。

### 为什么 30B Targeted ASR 是 `0/1`，仍然不能说安全？

该 injection task 的 deterministic check 要求工具参数与预注册 ground truth（基准真值）精确匹配。30B attacked trace 中，
`send_money(amount=1810)` 已经进入 runtime 并执行成功，但去敏后的收款参数只与 exact target 近似、不相等，所以
`attack_goal_achieved=false`。这说明 `0/1` 只回答“精确目标是否落地”，不能否定其他未授权有害副作用。

人工 near-miss（近似但未命中）审核是失败分析，不应事后包装成新的 generalized ASR（广义攻击成功率）。若要正式量化，
必须在下一轮预先定义参数级授权
或语义等价规则，并使用独立 test。

### 正式结果为什么要分 BU、UA、Targeted ASR？

BU 是无攻击时 AgentDojo utility check 通过的比例；UA 是有攻击的 security cases（受攻击安全案例）中同一个
utility check 通过的
比例；Targeted ASR 是攻击者指定目标真正落地的比例。上游按 attacked `utility` 直接聚合 UA，不再手工与
`not attack_succeeded` 做 AND（逻辑与）；如需“既有用又安全”的联合率，应另起名字。三者回答不同问题，不能用一个泛化的
`utility` 列代替。
旁边还要单列 clean/attack valid coverage（无攻击/受攻击有效覆盖率），避免 parse error（解析错误）、timeout（超时）或
detector invalid（检测器无效）缩小分母后美化结果。

### ASR 与 interception 为什么不同？

ASR 是 episode/environment（回合/环境）级：攻击目标最终是否落地。Interception（拦截率）是 call（调用）级：已经提出的恶意候选中有多少被
gate 阻断。模型根本没提出恶意调用不能算 probe 命中；parse error、timeout 或 detector invalid 也不能伪装成
防御成功，必须进入独立分母或失败桶。

### 如何避免数据泄漏？

不对同一任务的相邻轨迹做随机拆分。真实实验按 user task 或攻击模板分组，train 用于拟合，calibration 只选阈值，
test 只做一次最终报告。真实 artifact 固定模型、revision、dtype（数据类型）/量化配置、layer、position、模板、工具 schema
和 render mode（渲染模式）；不兼容时 detector invalid，而不是静默复用方向。

### 当前最大限制是什么？

首轮 held-out 只有 3 个 user tasks，而且统一使用 `injection_task_5 + injecagent`；人工审核后只有一条恶意
proposal，远不足以估计稳定拦截率。BU 只有 1/3，说明基础模型的任务完成和精确回答能力也是主要瓶颈。Activation
probe 在 attacked task 上误阻合法写操作，冻结 MELON 的 hashing embedding 也可能漏掉语义等价调用。新
`melon_paper` 尚未完成全量模型实验。当前价值是完整、
可审计的工程闭环和诚实的小型 held-out 失败分析，不是跨模板泛化结论。

## 可以说与不能说

可以说：

- “我实现了运行时 tool-call gate，并在 AgentDojo synthetic Banking 闭环中验证了执行前阻断。”
- “我实现了 direction/probe 的训练、校准、artifact 和推理路径。”
- “我独立实现了 MELON 的核心算法切片，当前用本地 hashing embedding。”
- “我另外按论文附录独立重建了 `melon_paper`：少样本掩码轨迹、参数投影、语义嵌入、跨步骤工具调用缓存、整批预检和命中后回合终止；当前尚未
  把它包装成论文原始数值复现。”
- “`melon_paper` 的 16 配对筛选已完成；链路稳定，但掩码候选稀疏，唯一阻断不是精确攻击参考调用，因此我主动停止在筛选阶段。”
- “Qwen3-8B 的首轮 30-episode held-out 已完成；direction 和 MELON 各审核到 1/1 恶意调用阻断，probe 则暴露了
  合法调用误阻。”
- “Qwen3-30B 白盒与 no-defense screening 已跑通，但 exact positive gate（精确正例门槛）未满足，所以我在训练 probe 和打开
  held-out 之前停止。”

不能说：

- “已经在 8B 模型上显著降低 ASR”。当前虽有 `1/3 → 0/3` 的首轮计数，但分母极小、攻击模板单一，不能称显著。
- “activation probe 成功拦截了攻击”。该组没有恶意 proposal，ASR=0 同时伴随合法 `update_user_info` 被提前阻断。
- “首次提出 Agent hidden-state circuit breaker（隐藏状态熔断器）”。已有相关工作。
- “完整复现了 MELON 官方效果”。论文模型环境和全量实验尚未重建，当前筛选也不支持这一说法。
- “当前 MELON 具有论文 §3.4 的理论保证”。虽已实现集成计算原语，但尚未验证该界成立所需的统计前提。
- “security_results=True 表示安全”。它表示攻击成功。
- “30B Targeted ASR 是 0/1，所以模型安全”。该 trial 已执行一笔人工审核为 injection-driven 的未授权转账。
- “已经在 30B 上复现防御效果”。本轮没有 30B artifact、calibration 或 held-out。

## 面试前 30 分钟排练

1. **5 分钟**：运行 `interview-demo`，用 proposed（已提出）/ blocked（已阻断）/ runtime-invoked（已进入运行时）
   三个词解释现场输出。
2. **5 分钟**：打开正式 held-out 报告，只讲一张主表和 activation probe 的失败案例。
3. **10 分钟**：不看稿讲两遍“两分钟主叙事”，第二遍必须主动说出样本量和攻击模板限制。
4. **10 分钟**：随机回答五题：为什么选 tool-call boundary、hook 哪个 token、标签如何做、如何防数据泄漏、
   MELON 与 probe 的成本差异。

验收标准：断网、无 GPU（图形处理器）时仍能完成 synthetic demo；不看代码也能画出
`tool output → model state/call → observation → decision → side effect`，并准确解释 probe 的 ASR=0 为什么
不是成功 interception。
