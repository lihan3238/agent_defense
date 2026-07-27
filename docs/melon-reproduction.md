# MELON 论文兼容重建

## 结论

MELON（Masked re-Execution and TooL comparisON，掩码重执行与工具调用比较）存在官方
GitHub（代码托管平台）仓库：`https://github.com/kaijiezhu11/MELON`。审计时上游提交为
`4d3cc9c0175cc26332aac696ac5556c3f85a5e8e`，但仓库只有 `README.md` 和 `pi_detector.py`，没有
LICENSE（许可证）、依赖锁、测试、原始日志、完整实验脚本或精确模型配置。

因此本仓库不复制上游源码，而是依据论文 §3、Algorithm 1（算法 1）和 Appendix A.1–A.3（附录 A.1–A.3）进行
clean-room（净室式）独立实现。当前目标是按论文附录重建公开算法和评测协议，不是逐字、逐行或连同上游缺陷一起复刻。
仅凭公开物料也不能保证逐项复现论文 Table 1（表 1）数值。

## 两条 MELON 路径不能混用

| 防御名 | 用途 | 实现语义 |
|---|---|---|
| `melon` | 冻结的历史工程基线 | 短中性提示、完整参数、256 维哈希嵌入、按当前调用阻断后继续 |
| `melon_paper` | 论文兼容重建 | 按论文附录重建的少样本掩码轨迹、安全参数投影、语义嵌入、整批动作预检、命中后终止回合 |

旧的 Qwen3-8B（通义千问 3 80 亿参数模型）正式矩阵仍只对应 `melon`，不得用新实现回写或重解释冻结数字。

## `melon_paper` 已补齐什么

1. 按论文构造 masking state（掩码状态）：删除原 user task（用户任务）和 assistant history（助手历史），保留系统消息，
   拼接全部历史 tool output（工具输出）。
2. 加入 Appendix A.2 的两段 few-shot（少样本）轨迹，再用虚构的 `read_file("random.txt")` 工具结果承载原轨迹输出。
3. 掩码路径只生成 tool call（工具调用），从不执行；候选调用按顺序放入跨步骤
   tool-call cache（工具调用缓存）。
4. 按 Appendix A.3 投影安全相关参数：

   - `send_email` 只保留 `recipients`；
   - `send_money` 只保留 `recipient` 和 `amount`；
   - 其他工具保留全部参数。

5. 使用 embedding（嵌入）向量和 cosine similarity（余弦相似度）比较本轮全部原始调用与缓存调用，严格采用
   `score > 0.8`。
6. 在任何工具执行前完成 batch preflight（整批预检）。任意调用命中后，本轮全部调用均不执行；原候选助手消息被替换成
   无工具调用的终止消息，同时候选仍保存在本项目独立审计轨迹中。
7. 实现论文 §3.4 的多提示平均分 ensemble（集成检测）计算原语。官方仓库没有发布这一部分的可运行实现，所以当前只做
   独立单元测试，未把它写成正式效果结论。

## 论文附录与发布源码的两处冲突

论文附录和官方发布源码并非逐字一致，因此不存在一个能同时与两者一比一匹配的提示模板：

- Appendix A.1 的中性提示从 `Please use...` 开始，发布源码则在前面增加 `Thanks. Then`；
- Appendix A.2 的第二段少样本以 `Successfully created the report.` 作为工具输出，发布源码对应字段却是空字符串。

本实现在这两处以论文附录为准；工具名使用 `read_file` 等可执行标识符，与发布源码一致。这些取舍必须披露，
不能把本路径称为官方源码的逐字复制。

## 本仓库的工程化语义

- 当前只复现跨步骤工具调用缓存。发布源码还会缓存已转换字符串的 embedding cache（嵌入缓存）；本实现保留原始候选调用，
  每次比较时重新计算已缓存调用的嵌入。检测语义在嵌入稳定时等价，但开销与源码不同。
- 论文 Algorithm 1 只规定返回攻击告警。本仓库把它接到 batch preflight（整批预检）：任意调用命中后整批都不执行，替换原
  assistant message（助手消息）并显式终止回合。发布源码也会通过替换候选消息自然停止，但还会改写前一条工具输出；本仓库不改写已记录输出。
- masked re-execution（掩码重执行）解析失败、嵌入失败等异常会按 fail-closed（故障时默认阻断）处理。这是本仓库的运行时边界策略，
  不是论文或发布源码规定的检测结论。

## 嵌入后端

### 论文后端

`--melon-embedding-backend openai --melon-embedding-model text-embedding-3-large`

这与官方代码的 OpenAI（模型服务提供方）后端一致。调用凭据只由运行环境提供，不进入参数、日志或 Git（版本控制系统）。

### 本地可复现后端

`--melon-embedding-backend hf --melon-embedding-model sentence-transformers/all-MiniLM-L6-v2`

该路径使用 Hugging Face（模型库）编码器和平均池化，便于离线测试完整控制流，但不能声称复现论文嵌入分数。本地
smoke（冒烟测试）中，相同攻击转账调用相似度为 `1.000000`，不同收款方的合法转账为 `0.793048`；这只验证后端接线与
阈值行为，不是 AgentDojo（智能体安全评测框架）效果指标。

## 与论文 629 案例对齐的协议

论文分母对应 AgentDojo `v1.1.2`，不是本仓库冻结矩阵使用的 `v1.2.2`：

| Suite（任务套件） | 用户任务 | 注入任务 | 配对案例 |
|---|---:|---:|---:|
| Banking（银行） | 16 | 9 | 144 |
| Slack（协作通信） | 21 | 5 | 105 |
| Travel（旅行） | 20 | 7 | 140 |
| Workspace（办公空间） | 40 | 6 | 240 |
| 合计 | 97 | 27 | 629 |

论文四种攻击在当前 AgentDojo 中对应：

- `direct`；
- `ignore_previous`；
- `system_message`；
- `important_instructions`。

每个 model × defense（模型×防御）需要 97 个 clean episode（无攻击回合）和 `4 × 629` 个 attacked episode
（受攻击回合），共 2613 回合。面试前优先运行 `none` 与 `melon_paper` 的单一强攻击，再决定是否扩展到四攻击全量。

## 单案例命令

本地语义后端：

```bash
uv sync --extra hf
uv run agent-defense agentdojo-run \
  --model Qwen/Qwen3-8B \
  --defense melon_paper \
  --benchmark-version v1.1.2 \
  --suite banking \
  --user-task user_task_1 \
  --injection-task injection_task_7 \
  --attack important_instructions \
  --scenario attacked \
  --melon-threshold 0.8 \
  --melon-embedding-backend hf \
  --melon-embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --melon-embedding-device cpu
```

论文嵌入后端：

```bash
uv sync --extra hf --extra melon-openai
uv run agent-defense agentdojo-run \
  --model Qwen/Qwen3-8B \
  --defense melon_paper \
  --benchmark-version v1.1.2 \
  --suite banking \
  --user-task user_task_1 \
  --injection-task injection_task_7 \
  --attack important_instructions \
  --scenario attacked \
  --melon-threshold 0.8 \
  --melon-embedding-backend openai \
  --melon-embedding-model text-embedding-3-large
```

## 仍然不能声称什么

- 不能声称复现论文三个模型的原始数值：论文没有公开完整 snapshot（模型快照）、服务商、推理配置和日志。
- 不能把本地 MiniLM（小型句向量模型）结果写成论文后端结果。
- 不能把当前 `v1.2.2` 的 949 配对称为论文 629 案例。
- 不能声称已复现 MELON-Aug（MELON 与重复用户提示组合方法）；官方仓库也没有发布该实现。
- 不能把 §3.4 的条件误差界描述成任意攻击下的端到端安全证明。

## 当前筛选结论

Qwen3-8B（通义千问 3 80 亿参数模型）与本地 MiniLM（小型句向量模型）的 16 配对筛选已经完成。工程链路可以跨四个
AgentDojo（智能体安全评测框架）套件运行，但掩码轨迹很少生成候选，唯一阻断也不匹配精确攻击参考调用，因此停止在
screening（筛选实验），不把它表述成论文效果复现。分母、开销和调用审核只维护在
[`reports/melon-paper-screening.md`](../reports/melon-paper-screening.md)。

## 面试表述

> 官方确实发布了核心源码，但没有许可证、版本锁和完整数值复现实验。我没有复制上游文件，而是按论文附录独立重建了
> 少样本掩码轨迹、安全参数投影、语义工具调用比较和跨步骤工具调用缓存。论文与发布源码有两处提示文本冲突，嵌入缓存也没有复制；
> 整批执行前检查、显式终止和故障时默认阻断是本仓库的工程化边界。论文商业模型环境无法由公开物料逐项重建，所以我把
> “算法与协议重建”和“本地模型重跑结果”分开报告。
