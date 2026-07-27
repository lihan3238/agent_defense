# AGENTS.md

## 目标

本仓库服务于 Agent（智能体）技术面试：把简历中的“表示级探针 × 工具调用边界”维护成可运行、可解释、可复现的
工程证据，不追求临时发明新方法或刷 SOTA（当前最先进水平）。

核心威胁是 indirect prompt injection（间接提示注入）：不可信 tool output（工具输出）进入上下文，模型随后提出
越权 tool call（工具调用）。真正的安全边界位于候选动作已生成、`FunctionsRuntime.run_function` 尚未执行之时。

## 稳定主线

```text
untrusted tool output
→ in-process HF generation
→ candidate tool call + resid_pre
→ direction / logistic probe / MELON signal
→ deterministic tool-risk policy
→ GuardedToolsExecutor allow or block
→ AgentDojo utility and attack-goal checks
```

- 基线固定为 AgentDojo（智能体安全评测框架）`agentdojo==0.1.35`、Banking（银行任务套件）`v1.2.2` 和
  Qwen3-8B（通义千问 3 80 亿参数模型）冻结配置。
- 表示方法参考 *Your Agent is More Brittle Than You Think*（《你的智能体比你想象中更脆弱》）§3.3；本仓库自行
  操作化 token（词元）、hook（钩子）、artifact（工件）和执行边界，不声称复现论文未公开的精确约定。
- Direction（方向）是可解释基线；refusal direction（拒答方向）不等同于
  prompt-injection detector（提示注入检测器）。
- 冻结结果中的 `melon` 只称 MELON（掩码重执行检测方法）核心算法切片：已实现 masked re-execution（掩码重执行）、
  cache（缓存）和 pre-action comparison（动作执行前比较）；hashing embedding（哈希嵌入）、单中性提示和
  block-then-continue（阻断后继续）都不同于论文完整配置。
- 新增 `melon_paper` 是依据论文附录独立实现的兼容重建：按附录重建的少样本掩码轨迹、参数投影、语义嵌入接口、整批预检和
  命中后回合终止；只有 `text-embedding-3-large` 后端对应论文发布代码，且尚无正式全量效果结果。
- Detector（检测器）只打分，risk policy（风险策略）形成决策，executor（执行器）才执行
  allow/block（放行/阻断）。

## 已冻结证据

- AgentDojo contract test（契约测试）已证明 probe（探针）能在 runtime（运行时）前阻断恶意调用，环境无副作用。
- Qwen2.5-0.5B（通义千问 2.5 5 亿参数模型）只作为本地 Hugging Face（模型库）白盒 hook
  smoke（冒烟测试），不进入效果表。
- Qwen3-8B one-task spike（单任务接线验证）只证明真实模型接线，不证明 probe 泛化。
- 30-episode（30 回合）held-out（留出测试）矩阵已完成并人工审核；正式数字只维护在
  `reports/qwen3-heldout-matrix.md`。
- `melon_paper` 的 16 配对 screening（筛选实验）共计划 64 回合，60 个有效；唯一阻断发生在 runtime（运行时）前，
  但不匹配精确攻击参考调用，不能计为已验证恶意调用拦截。该筛选数字只维护在
  `reports/melon-paper-screening.md`。
- Qwen3-30B-A3B（通义千问 3，300 亿总参数/30 亿激活参数模型）FP8（8 位浮点）已通过
  layer-29 hidden-state（第 29 层隐藏状态）与
  native tool-call（原生工具调用）smoke。其预注册 screening（筛选实验）中 clean utility（无攻击任务可用性）通过；
  attacked run（受攻击运行）执行了 injection-driven（注入驱动）未授权转账，但因目标账户抄错未命中 AgentDojo
  exact ASR（精确攻击成功率）。continuation gate（继续实验门槛）因此停止，未训练 30B artifact，也未运行
  30B held-out。
- Activation probe（激活探针）的 ASR=0 伴随合法调用误阻，不能表述为成功 interception（拦截）。
- AgentDojo injection task（注入任务）的 `security_results=True` 表示攻击目标完成，即 ASR 命中。
- Synthetic activation（合成激活）/demo（演示）只能证明控制流，不能写入真实模型效果表。

## 当前阶段

冻结现有 test（测试集）、artifact、layer（层）、position（位置）和 threshold（阈值）。除非用户明确要求新实验，
当前只做：

1. 降低理解成本，保持 README、代码导览、架构、协议、报告各自只有一个职责；
2. 修正文档、命名和命令不一致；
3. 增加不改变行为的小型测试或注释；
4. 运行全量回归，保持现有证据稳定。

不要依据已观察的 `user_task_1/11/13` 回调参数后重报同一 test。

## 开发规则

- 使用 Python（编程语言）3.12、`uv`（Python 包与环境管理器）和仓库独立 `.venv`；不要安装到
  base（基础环境）。
- 改动保持小而可测；完成态陈述必须指向代码、测试、命令或去敏结果摘要。
- 模型、revision（修订版本）、dtype（数据类型）、layer、position 或模板身份不兼容时
  fail closed（故障时默认阻断），不静默复用 artifact。
- Function-call（函数调用）表示必须 replay（重放）模型原始生成 token；不得重序列化
  JSON（JavaScript 对象表示法）冒充原表示。
- 默认只运行 AgentDojo 沙箱工具；不连接真实邮件、银行、账户、shell（命令行外壳）或不可逆文件操作。
- 不提交密钥、私人输入、机器地址、GPU（图形处理器）编号、缓存路径、权重、原始隐藏状态或大日志。
- 不扩展 RL（强化学习）、多智能体、MCP（模型上下文协议）、GUI（图形用户界面）、Web UI（网页用户界面）或
  分布式训练。
- 不复制无明确许可证的 MELON 或 PVDetector（投影向量检测器）源文件；按论文独立实现并披露差异。

开始工作先读 `README.md`。理解代码优先读 `docs/code-tour.md`；稳定设计见
`docs/architecture.md`，实验规则见 `docs/experiment-protocol.md`，面试话术见
`docs/interview-guide.md`。
