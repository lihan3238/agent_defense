# 实验与证据记录索引

本页是 Agent（智能体）运行时防御项目的记录入口，只说明每份证据回答什么问题，不重复维护动态指标。
完整分母、结果数字和限制以对应正式报告为准；同名 JSON（JavaScript Object Notation，JavaScript 对象表示法）
是机器可读摘要，不能脱离人读报告单独解释。

## 正式效果证据

| 记录 | 状态 | 回答的问题 | 不能外推什么 |
|---|---|---|---|
| [Qwen3-8B（通义千问 3 80 亿参数模型）四套件主矩阵](qwen3-v112-full-matrix.md) / [JSON](qwen3-v112-full-matrix.json) | 冻结正式结果 | MELON（Masked re-Execution and TooL comparisON，掩码重执行与工具调用比较）论文兼容路径在 AgentDojo（智能体安全评测框架）完整单攻击分母上的 Targeted ASR（Targeted Attack Success Rate，定向攻击成功率）、任务可用性、失败、开销和调用诊断 | 论文四攻击数值、跨模型泛化、人工恶意调用标签 |
| [Qwen3-8B 表示级留出测试矩阵](qwen3-heldout-matrix.md) / [JSON](qwen3-heldout-matrix.json) | 冻结正式结果 | 自定义 direction/probe（方向/探针）的隐藏状态与执行前阻断，以及与内置防御的同协议回合级对比 | 大样本统计显著性、跨攻击或跨模型泛化 |

两份正式结果职责不同：四套件主矩阵回答“完整分母上的 MELON 效果”，30 回合表示级矩阵回答简历中的
“表示级探针 × 工具调用边界”。不要把两张表的分母或指标拼成一组结果。

## 工程、筛选与负结果

| 记录 | 状态 | 用途 |
|---|---|---|
| [已验证运行快照](verified-smoke.md) / [单任务机读摘要](qwen3-reality-spike.json) | 2026-07-26 历史工程快照 | 记录 synthetic（合成）控制流、AgentDojo 执行边界、本地模型白盒接线和 one-task spike（单任务接线验证）；不承担当前效果数字 |
| [MELON 论文兼容筛选报告](melon-paper-screening.md) | 主矩阵前的历史筛选 | 保存 16 配对筛选、掩码候选稀疏和唯一阻断审核；后续效果结论已由完整主矩阵接替 |
| [Qwen3-30B-A3B（通义千问 3，300 亿总参数/30 亿激活参数模型）筛选报告](qwen3-30b-screening.md) / [JSON](qwen3-30b-screening.json) | 跨模型负结果 | 记录更大模型白盒兼容性、精确目标与有害近似调用的差异及停止门槛；不进入防御效果表 |

## 按问题找证据

- 问完整数据集上的防御效果：看 [四套件主矩阵](qwen3-v112-full-matrix.md)。
- 问简历中 refusal direction（拒答方向）表述的准确口径，以及 direction/probe（方向/探针）的实际结果：看
  [表示级留出测试矩阵](qwen3-heldout-matrix.md)。
- 问阻断是否真的早于副作用：看 [已验证运行快照](verified-smoke.md) 和执行边界契约测试。
- 问 MELON 到底复现了什么、与论文差在哪里：看
  [MELON 论文兼容重建](../docs/melon-reproduction.md)。
- 问 30B 为什么没有继续训练探针：看 [30B 筛选报告](qwen3-30b-screening.md)。

## 记录纪律

- Markdown（标记语言）报告是人读事实源；JSON 只保存同轮去敏摘要。
- 原始输出、运行日志、隐藏状态、权重和中间分析保留在 Git（版本控制系统）忽略目录，不提交到仓库。
- 历史筛选和负结果不回写成“成功实验”；后续进展通过状态说明和新正式报告接续。
- synthetic demo（合成演示）与 one-task spike 只证明控制流或接线，不进入真实模型效果表。
- AgentDojo 注入任务中的原始 `security_results=True` 表示攻击目标完成，不表示安全通过。

推荐阅读顺序：仓库 [README](../README.md) → [代码导览](../docs/code-tour.md) →
[架构说明](../docs/architecture.md) → 本索引 → 两份正式结果 → [面试讲解指南](../docs/interview-guide.md)。
