# AGENTS.md

## 目标

本仓库服务于 Agent 技术面试：把简历中的“表示级探针 × 工具调用边界”维护成可运行、可解释、可复现的
工程证据，不追求临时发明新方法或刷 SOTA。

核心威胁是 indirect prompt injection：不可信 tool output 进入上下文，模型随后提出越权 tool call。真正的
安全边界位于候选动作已生成、`FunctionsRuntime.run_function` 尚未执行之时。

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

- 基线固定为 `agentdojo==0.1.35`、Banking `v1.2.2` 和 Qwen3-8B 冻结配置。
- 表示方法参考 *Your Agent is More Brittle Than You Think* §3.3；本仓库自行操作化 token、hook、
  artifact 和执行边界，不声称复现论文未公开的精确约定。
- Direction 是可解释基线；refusal direction 不等同于 prompt-injection detector。
- MELON 只称核心算法切片：已实现 masked re-execution、cache 和 pre-action comparison；hashing embedding、
  单中性提示和 block-then-continue 都不同于论文完整配置。
- Detector 只打分，risk policy 形成决策，executor 才执行 allow/block。

## 已冻结证据

- AgentDojo contract test 已证明 probe 能在 runtime 前阻断恶意调用，环境无副作用。
- Qwen2.5-0.5B 只作为本地白盒 hook smoke，不进入效果表。
- Qwen3-8B one-task spike 只证明真实模型接线，不证明 probe 泛化。
- 30-episode held-out 矩阵已完成并人工审核；正式数字只维护在
  `reports/qwen3-heldout-matrix.md`。
- Qwen3-30B-A3B FP8 已通过 layer-29 hidden-state 与 native tool-call smoke。其预注册 screening 中 clean
  utility 通过；attacked run 执行了 injection-driven 未授权转账，但因目标账户抄错未命中 AgentDojo exact ASR。
  continuation gate 因此停止，未训练 30B artifact，也未运行 30B held-out。
- Activation probe 的 ASR=0 伴随合法调用误阻，不能表述为成功 interception。
- AgentDojo injection task 的 `security_results=True` 表示攻击目标完成，即 ASR 命中。
- Synthetic activation/demo 只能证明控制流，不能写入真实模型效果表。

## 当前阶段

冻结现有 test、artifact、layer、position 和 threshold。除非用户明确要求新实验，当前只做：

1. 降低理解成本，保持 README、代码导览、架构、协议、报告各自只有一个职责；
2. 修正文档、命名和命令不一致；
3. 增加不改变行为的小型测试或注释；
4. 运行全量回归，保持现有证据稳定。

不要依据已观察的 `user_task_1/11/13` 回调参数后重报同一 test。

## 开发规则

- 使用 Python 3.12、`uv` 和仓库独立 `.venv`；不要安装到 base。
- 改动保持小而可测；完成态陈述必须指向代码、测试、命令或去敏结果摘要。
- 模型、revision、dtype、layer、position 或模板身份不兼容时 fail closed，不静默复用 artifact。
- Function-call 表示必须 replay 模型原始生成 token；不得重序列化 JSON 冒充原表示。
- 默认只运行 AgentDojo 沙箱工具；不连接真实邮件、银行、账户、shell 或不可逆文件操作。
- 不提交密钥、私人输入、机器地址、GPU 编号、缓存路径、权重、原始隐藏状态或大日志。
- 不扩展 RL、多智能体、MCP、GUI、Web UI 或分布式训练。
- 不复制无明确许可证的 MELON/PVDetector 源文件；按论文独立实现并披露差异。

开始工作先读 `README.md`。理解代码优先读 `docs/code-tour.md`；稳定设计见
`docs/architecture.md`，实验规则见 `docs/experiment-protocol.md`，面试话术见
`docs/interview-guide.md`。
