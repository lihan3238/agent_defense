# AGENTS.md

## 作用域

本文件适用于整个仓库。更深目录若新增 `AGENTS.md`，以离目标文件最近的规则为准。

## 项目使命

构建一个可复现、可解释、可现场演示的 Agent 运行时注入防御原型：本地开源模型生成候选工具调用后、工具真正执行前，读取模型隐藏状态并用 activation probe / refusal direction 打分，再由独立策略层决定允许或阻断；使用确定性 utility/security 检查评估安全性、正常任务可用性与开销。

本仓库首先服务于一周内可讲清、可运行、可验证的面试项目，不建设通用 Agent 平台，也不以堆砌框架或岗位关键词为目标。

## 事实与证据优先级

1. 当前用户请求和最近的 `AGENTS.md`。
2. 仓库中的代码、测试、配置、锁文件和可复现实验产物。
3. 已固定版本或 commit 的上游官方文档、源码与论文。
4. 外部冲刺笔记和简历只说明目标与叙事，不证明功能已经实现或指标已经获得。

遵守以下规则：

- 项目证据拥有当前状态。代码、测试或实验不支持的内容必须写成“计划”“假设”或“待验证”。
- 简历中的每个技术声明最终都应能指向一条命令、一个测试或一份带元数据的实验摘要。
- 不把私人笔记、简历、个人路径、联系方式或机器清单复制进仓库。
- 不把外部笔记中的数字或趋势当作本项目实验结果。
- 当 `pyproject.toml` 尚不存在时，视为 bootstrap 阶段；先建立最小可测骨架，不假装已有运行命令。

## 研究与开源边界

- 只实现和公开已发表或已公开的方法，例如 refusal direction、CAA、SafeSteer 与 AgentDojo 已公开协议。
- 不读取、复制或反向移植其他私有研究仓库中的未发表实现、数据与中间产物。
- 任何未发表研究的具体命名、假设、公式、分解方式、泛化技巧、实现细节及其等价表达，都不进入代码、配置、注释、issue、测试数据或公开文档。
- 引用或改写上游实现时记录来源、版本/commit 和许可证；能依赖成熟上游包时不复制源码。
- 新方法、新数据管线或训练任务只有在重复出现确定性缺口、替换边界足够小且有可测收益时才引入。

## 威胁模型与信任边界

最小威胁模型如下：

- 攻击者可控制 Agent 读取的邮件、网页、文档或工具返回值，并在其中放置 direct/indirect prompt injection。
- 攻击目标是诱导 Agent 产生违反用户意图或工具策略的调用，例如越权读取、数据外传、非预期写入或危险参数组合。
- 模型输出、候选工具名、候选参数和外部内容全部是不可信数据。
- 真正的强制边界是工具执行器，而不是 system prompt、自然语言拒绝或探针标签本身。
- 探针提供检测信号；策略层把分数、工具风险和参数校验转成 `allow` / `block`。两者必须可独立测试。

首版不声称解决通用越狱、模型权重投毒、训练数据投毒、主机入侵或任意多智能体串谋。新增攻击面前先更新 threat model 和验收条件。

## 硬性安全约束

- 默认只提供 2–3 个确定性、可回滚、无真实副作用的 sandbox 工具。
- 测试和演示默认 `dry-run`；不得让模型输出直接进入 shell、网络、邮件、账户、真实文件写入或其他不可逆操作。
- 工具必须显式注册、使用严格 schema、拒绝未知字段，并在执行前完成类型、范围、路径和权限校验。
- 高风险工具在探针不可用、超时、产生 NaN 或配置不匹配时 fail closed；低风险模拟工具是否继续必须由显式配置决定，禁止静默放行。
- 工具参数校验、最小权限和 allowlist 不得被 activation probe 取代。
- 日志默认不保存完整 prompt、私人文档、密钥、原始隐藏状态或可还原敏感内容；只记录评测所需的最小结构化字段。
- 不提交密钥、token、模型权重、数据集副本、缓存、完整隐藏状态或大体积原始 trace。

## 近期优先级

严格按以下顺序推进；后项不得阻塞前项：

1. **P0：最小 Agent 闭环。** 本地模型、确定性工具、候选 tool call、观察结果和终止条件全部跑通。
2. **P1：运行时边界。** 在每次真实执行前调用探针，展示同一任务 `defense=none` 与 `defense=activation_probe` 的 before/after。
3. **P2：可信评测。** 固定任务、攻击、模型、seed 和配置，计算 utility/security、ASR、拦截率、误阻率与延迟。
4. **P3：AgentDojo 集成。** 适配当前固定版本的接口，并在同协议下比较无防御和至少一个经核实存在的上游基线。
5. **P4：展示材料。** README、架构图、一页结果摘要和两分钟内可完成的稳定 demo。
6. **P5：扩展研究。** indirect/multi-turn/adaptive attacks、MCP、planner/executor、memory、多智能体或 RL。

面试前不得为了匹配 JD 而提前扩展 RL、多智能体、Repo 级轨迹或分布式训练。没有端到端证据时，只能把它们作为下一步设计讨论。

## 目标代码结构

按需创建，禁止只为匹配目录树而添加空文件：

```text
pyproject.toml
src/agent_defense/
  agent/          # 最小 loop、消息状态、停止条件
  models/         # 本地模型适配与隐藏状态采集
  tools/          # schema、registry、risk policy、sandbox executor
  defenses/       # detector 接口、no-defense、activation probe、决策策略
  eval/           # 任务适配、确定性检查、指标聚合
  telemetry/      # 结构化 trace 与计时，不记录敏感原文
  cli.py
tests/
  unit/
  integration/
configs/
  demo/
  eval/
docs/
  threat-model.md
  architecture.md
reports/          # 小型、可审计、可复现的聚合结果
```

核心数据流保持单向且清晰：

```text
untrusted observation -> model -> candidate tool call
                      -> probe observation -> policy decision
                      -> validated sandbox executor -> tool result
```

- Agent loop 不直接执行工具。
- 模型适配层不决定安全策略。
- 探针只产出结构化观测，不修改工具参数。
- 策略层不依赖具体模型类。
- AgentDojo 逻辑放在适配层，不侵入核心 loop。

## 稳定接口

优先用小型 typed dataclass、Protocol 或现有上游类型表达以下概念，避免字典在层间自由漂移：

- `CandidateToolCall`：工具名、结构化参数、call id。
- `ProbeObservation`：标量分数、层、token/pooling 位置、方向版本、耗时和有效性状态；默认不携带完整激活。
- `PolicyDecision`：`allow` / `block`、稳定 reason code、阈值、工具风险级别和决策耗时。
- `ToolResult`：成功状态、最小观察值和显式错误类型。
- `TrialRecord`：任务、攻击、配置指纹、seed、决策、确定性检查结果和计时。

reason code 应稳定、可聚合，例如 `score_above_threshold`、`invalid_tool_schema`、`detector_unavailable`；不要让测试依赖易变的自然语言解释。

## 模型与探针实现约束

- 需要探针的主路径必须使用进程内、可读取隐藏状态的本地开源模型；闭源 API 只能作为不带表示级探针的对照或 loop 学习样例。
- 模型 ID、revision、tokenizer revision、dtype、device、层号、token 位置/pooling、chat template 和生成参数都进入配置或运行清单，禁止散落成魔法常量。
- 不硬编码个人模型路径。通过配置或命名环境变量接收本地路径；默认测试不得隐式联网下载模型。
- forward hook 必须封装成上下文管理器或等价生命周期对象，在异常路径也能移除。测试重复调用和并发/串行复用，防止 hook 泄漏或重复注册。
- 明确区分 residual stream、attention 输出及其他激活位置；名称、论文定义和代码实际张量必须一致。
- 明确 token 选择与 pooling 规则。禁止用“最后一个 token”掩盖 padding、chat template 或生成阶段索引差异。
- 方向向量 artifact 至少记录：兼容模型/revision、层、提取方法、归一化方式、训练/校准数据摘要或 hash、创建代码版本。
- 阈值只在 calibration split 上选择；测试集只能做最终评估。不得按目标测试结果手调阈值。
- 分数符号、归一化和判定方向必须有合成单元测试，避免把高风险/低风险方向写反。

## Agent 与工具调用约束

- 首版 loop 应保持可读，显式展示 `reason/plan -> candidate action -> observation -> next step`；不为少量功能引入大型 Agent 框架。
- 如果模型不稳定地产生合法 tool call，先用严格解析、有限重试和清晰错误修复，不用隐藏 prompt patch 掩盖失败率。
- 每轮设最大步数、最大重试数和总超时；达到上限时返回可区分的终止原因。
- tool choice accuracy、schema validity 和任务完成率分别统计，不把“生成了 JSON”当作任务成功。
- 注入文本始终作为不可信数据进入上下文；测试夹具中清楚标记 trusted instruction 与 untrusted content 的来源。
- 工具风险策略至少区分只读/模拟写入/高影响动作。探针分数相同不意味着不同风险工具应有完全相同的处置。

## 依赖与上游策略

- Python 项目使用 `pyproject.toml` 和单一锁文件；优先采用成熟工具链。若采用 `uv`，CI/复现实验使用 frozen lock。
- 固定 AgentDojo、Transformers、PyTorch 及相关模型 revision；升级依赖必须单独变更，并重跑协议/回归测试。
- 先查看当前固定版本的官方 API 与源码，再编写 adapter；不凭笔记中的旧类名或旧函数签名编码。
- 优先复用 AgentDojo 的任务、攻击和确定性检查；核心 probe 与 policy 保持独立，避免被某个 benchmark 锁死。
- 不默认引入 LangChain、LangGraph、MCP SDK、向量数据库、Web UI 或分布式训练栈。只有明确验收条件证明现有最小实现存在重复缺口时才增加。
- 默认测试必须离线、CPU 可运行，并使用 fake model/fixture；真实 8B 模型与 GPU 评测标为显式 integration/slow 测试。

## 评测协议

所有防御比较必须保持模型、任务、攻击输入、生成参数、seed、最大步数和工具实现一致，只改变被研究的防御变量。

至少报告原始计数和以下指标：

- `utility = 正常任务通过确定性 utility check 的数量 / 正常任务总数`
- `ASR = 攻击试验中恶意目标通过 deterministic security outcome check 的数量 / 有效攻击试验总数`
- `interception_rate = 在工具执行前被正确阻断的注入试验数 / 应阻断的注入试验总数`
- `false_block_rate = 被阻断的正常试验数 / 正常试验总数`
- `tool_schema_valid_rate = schema 合法的候选调用数 / 所有候选调用数`
- `latency_overhead`：同配置下防御开启相对关闭的额外延迟，同时报告绝对值；经过 warm-up 后至少给出中位数，样本足够时给 p95。

规则：

- AgentDojo 的原生 utility/security deterministic checks 是主结果；LLM judge 只能作为补充分析，不能成为唯一证据。
- 不自动假设 `ASR = 1 - security`；先核对两者分母和无效试验处理是否一致。
- 报告总试验数、成功/失败/无效数。小样本先诚实给原始计数，不用小数点制造精度幻觉。
- 至少包含 `no_defense` 基线。加入 spotlighting、transformer detector 或其他内置基线前，先核实当前固定版本确实提供且协议可比。
- 失败、解析错误、模型拒答、超时和 detector error 必须分桶，禁止全部归为“安全”。
- 不挑选最好 seed、最好层或最好阈值作为唯一结果。搜索空间、选择规则和 calibration/test 边界必须留痕。
- 性能比较使用同一运行环境、精度、batch 和计时范围；若无法严格控制，明确标注限制。

## 实验与产物管理

- 每次正式运行保存不可变配置快照和短 manifest：代码 commit、依赖锁 hash、模型/revision、方向 artifact id、suite、攻击、seed、阈值与时间。
- 原始运行数据写入 gitignored 的 `artifacts/` 或等价目录；只提交体积小、已去敏、能追溯 manifest 的 CSV/JSON 和 Markdown 聚合摘要。
- trace 默认只记录 task/sample id、候选工具、分数、阈值、decision、reason code、utility/security outcome 和阶段耗时，不记录完整私人输入。
- 图表必须由已提交的聚合数据和可复现脚本生成；禁止手工修改图中数字。
- 结果表同时列出 defense-on/off，避免只展示成功案例。
- 无法完成 GPU 或全量 benchmark 时，提交 CPU 单元测试和小型 smoke evidence，并在文档中明确尚未验证的部分；不得伪造或估算结果。

## 测试要求

每个行为变更至少覆盖与风险相称的测试：

- 纯函数单元测试：分数投影、归一化、阈值方向、指标分母、reason code。
- hook 生命周期测试：正常、异常、重复调用后均无残留 hook。
- tool boundary 测试：允许正常调用、阻断攻击调用、拒绝未知工具/字段/越界参数。
- failure-path 测试：解析失败、超时、NaN、模型/方向不兼容、detector unavailable。
- 对照测试：关闭防御时不偷偷执行探针；开启防御时每个实际 tool call 都经过且仅经过一次策略判断。
- 集成测试：使用极小公开模型或可控 fake backend 跑完整 loop；AgentDojo/8B/GPU 测试显式标记并按需运行。
- 回归测试：修复实验协议或指标 bug 时，先加入能复现旧错误的测试。

bootstrap 后的默认开发入口应收敛为少量命令，并在 README 与 CI 保持一致，例如：

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

只有实际配置了相应工具后，才把命令写成“可用”。不要在文档中保留不存在的命令。

## 编码规范

- Python API、符号、配置键和结构化日志使用英文；面试说明和面向用户的文档可使用中文。
- 公共接口与安全关键函数提供类型标注和简短 docstring，说明输入信任级别、单位、shape 与失败行为。
- 优先小函数、显式依赖和纯计算；避免全局可变状态、隐式单例、导入时加载模型或导入时联网。
- 不用宽泛 `except Exception` 把安全故障转成允许；若在顶层收敛异常，必须记录稳定错误类型并遵守 fail policy。
- 不用字符串拼接生成可执行命令，不对模型输出使用 `eval`/`exec`，不反序列化不可信 pickle。
- 数值 tensor 的 device/dtype/shape 转换必须显式；热路径避免无意的 CPU/GPU 往返。
- 注释解释“为什么”和协议边界，不复述代码，不写未经证实的宣传性结论。
- 只改当前任务需要的文件；保持变更小而可审查，不顺手重构无关区域。

## 开发前五道门

编码前快速检查：

1. **Eligibility：** 是否直接改善 MVP、可信指标、可复现性或面试演示？
2. **Collision：** 当前仓库或成熟上游是否已有同等能力？
3. **Reality：** 是否已查看真实文件、固定版本 API 和现有测试，而不是按记忆假设？
4. **Clarity：** 是否能写出一个可观察的验收条件和失败条件？
5. **Scope：** 是否是满足验收条件的最小改动？

任一项不通过，先缩小或重新定义任务。探索性代码要有明确退出条件，不能永久进入主路径。

## 文档与面试交付

README 最终至少包含：一句话定位、威胁模型、架构图、安装与模型配置、最小 demo、评测命令、指标定义、结果表、局限和复现元数据。

文档必须明确区分：

- `Implemented and verified`
- `Implemented but not fully evaluated`
- `Planned`

推荐让首个可用版本收敛到稳定 CLI，而不是依赖 notebook 手工执行：

```bash
python -m agent_defense.cli demo --config configs/demo/minimal.yaml --defense none
python -m agent_defense.cli demo --config configs/demo/minimal.yaml --defense activation_probe
python -m agent_defense.cli eval --config configs/eval/smoke.yaml
```

这些是目标接口；在实现和测试存在前不得宣称可运行。Notebook 仅用于分析，不作为唯一 demo 或唯一指标来源。

对华为 Agent 技术岗位的叙事重点应来自真实工程证据：agent loop、上下文中的信任边界、工具 schema、失败重试、运行时决策、确定性 eval、延迟与泛化限制。MCP、规划、记忆、RL 和多智能体只在核心闭环完成后作为扩展设计讨论。

## 完成定义

一次变更只有同时满足以下条件才算完成：

- 行为与请求一致，且没有越过研究、隐私和真实副作用边界。
- 新行为有自动化测试；相关既有测试通过。
- 能运行最小 smoke path，或明确说明因何未运行以及未验证范围。
- 指标与实验变更记录配置、分母、失败桶和可复现元数据。
- README/配置/类型接口在需要时同步更新，不把计划写成事实。
- 没有提交秘密、私人路径、模型权重、缓存、大型日志或原始隐藏状态。
- 最终汇报说明做了什么、如何验证、剩余限制；不得用推测结果填补未完成实验。
