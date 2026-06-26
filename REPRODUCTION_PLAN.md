# verl-recipe 全流程复现计划

> 硬件约束：8x RTX 4090 (24GB VRAM, PCIe 4.0, 无 NVLink)

---

## 目录

1. [硬件约束与可行性分析](#1-硬件约束与可行性分析)
2. [总体学习路径](#2-总体学习路径)
3. [Phase 0: 环境搭建与验证](#3-phase-0-环境搭建与验证)
4. [Phase 1: 入门 — char_count (1 GPU)](#4-phase-1-入门--char_count-1-gpu)
5. [Phase 2: 评估实战 — r1 (1-2 GPU)](#5-phase-2-评估实战--r1-1-2-gpu)
6. [Phase 3: 算法扩展 — gvpo (2-4 GPU)](#6-phase-3-算法扩展--gvpo-2-4-gpu)
7. [Phase 4: GenRM + 多任务 — genrm_remote (4 GPU)](#7-phase-4-genrm--多任务--genrm_remote-4-gpu)
8. [Phase 5: Multi-Turn Agent — langgraph_agent (4-8 GPU)](#8-phase-5-multi-turn-agent--langgraph_agent-4-8-gpu)
9. [Phase 6: 生产级 RL — dapo 7B (8 GPU)](#9-phase-6-生产级-rl--dapo-7b-8-gpu)
10. [Phase 7: Agent + Tool-Use — retool 7B (8 GPU)](#10-phase-7-agent--tool-use--retool-7b-8-gpu)
11. [Phase 8: 异步训练 — async_flow (8 GPU)](#11-phase-8-异步训练--async_flow-8-gpu)
12. [各阶段时间估算](#12-各阶段时间估算)
13. [关键超参数速查表](#13-关键超参数速查表)
14. [故障排查速查表](#14-故障排查速查表)

---

## 1. 硬件约束与可行性分析

### 1.1 RTX 4090 vs H100 对比

| 规格 | RTX 4090 | H100 (参考) | 影响 |
|------|----------|-------------|------|
| VRAM | 24 GB | 80 GB | 模型大小受限，必须用 offloading |
| 显存带宽 | 1008 GB/s | 3352 GB/s | 训练/推理速度慢 ~3x |
| BF16 TFLOPS | ~82.6 | ~989 | 计算慢 ~12x |
| 互联 | PCIe 4.0 (~32 GB/s) | NVLink (900 GB/s) | 多卡通信慢 ~28x |
| TMA | 不支持 (sm_89) | 支持 (sm_90) | 部分 Triton kernel 降级 |

### 1.2 核心约束

1. **显存约束**：7B 模型 BF16 权重 ~14.5GB，单卡 24GB 装不下模型+KV Cache+训练状态
2. **通信约束**：PCIe 4.0 带宽有限，TP/SP 会显著变慢
3. **计算约束**：4090 的 BF16 算力约为 H100 的 1/12，训练时间会更长

### 1.3 各 Recipe 可行性总览

| Recipe | 模型 | 最少 GPU | 可行性 | 需要的修改 |
|--------|------|---------|--------|-----------|
| **char_count** | 135M | **1** | ✅ 完全可行 | 无 |
| **r1** (eval only) | 1.5B | **1** | ✅ 完全可行 | 无 |
| **gvpo** | 7B | **2-4** | ✅ 可行 | 开启 offload，减小 batch |
| **genrm_remote** | 3B | **4** | ✅ 可行 | 无 |
| **langgraph_agent** | 3B | **4** | ✅ 可行 | 开启 offload |
| **dapo** | 7B | **8** | ✅ 可行 | offload+减 batch+减 n_resp |
| **retool** | 7B | **8** | ⚠️ 紧张 | offload+减 seq_len+TP=4 |
| **async_flow** | 0.5B/7B | **1-8** | ✅ 可行 | 0.5B 轻松，7B 需 offload |
| **open_math_reasoning** | 8B SFT | **8** | ⚠️ 紧张 | 减 SP/FSDP/seq_len |
| **dapo** (32B) | 32B | 8 | ❌ 不可行 | 显存不足 |

### 1.4 4090 专用优化策略

```bash
# 必须开启的选项（7B+ 模型）
param_offload=True          # 参数 offload 到 CPU
optimizer_offload=True      # 优化器状态 offload 到 CPU
enable_gradient_checkpointing=True  # 梯度检查点

# vLLM 配置
gpu_memory_utilization=0.5-0.6  # 保守分配（H100 用 0.85）
enforce_eager=True              # 禁用 CUDA Graph（避免已知 bug）
tensor_model_parallel_size=2-4  # 7B 模型需要 TP>=2

# 通信配置
NCCL_P2P_DISABLE=1              # 4090 可能需要
```

---

## 2. 总体学习路径

```
Phase 0: 环境搭建与验证 (Day 1)
  │
Phase 1: char_count — 理解全流程 (Day 1-2)
  │   学习：数据格式 → SFT → Reward → GRPO → 评估
  │
Phase 2: r1 — 评估实战 (Day 2-3)
  │   学习：多数据集评估、Reward 路由、best@N/maj@N
  │
Phase 3: gvpo — 自定义算法 (Day 3-5)
  │   学习：自定义 loss、自定义 trainer、算法对比实验
  │
Phase 4: genrm_remote — GenRM + 多任务 (Day 5-7)
  │   学习：远程 reward model、API 服务、多任务 reward
  │
Phase 5: langgraph_agent — Multi-Turn Agent (Day 7-10)
  │   学习：Agent Loop、Tool 调用、response_mask、状态机
  │
Phase 6: dapo 7B — 生产级 RL (Day 10-14)
  │   学习：Dynamic Sampling、Clip-Higher、Overlong Penalty
  │
Phase 7: retool 7B — Agent + RL (Day 14-18)
  │   学习：代码执行 Sandbox、多轮 RL 训练
  │
Phase 8: async_flow — 异步训练 (Day 18-22)
      学习：TransferQueue、异步流水线、吞吐优化
```

---

## 3. Phase 0: 环境搭建与验证

### 3.1 安装 verl

```bash
# 方案 A：Docker（推荐，避免依赖冲突）
docker pull verlai/verl:vllm011.latest
docker run --gpus all -it -v /data/home/yizhou:/workspace verlai/verl:vllm011.latest

# 方案 B：从源码安装
cd /data/home/yizhou/verl
pip install --no-deps -e .
pip install -e ".[vllm,gpu,math]"
```

### 3.2 验证安装

```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_mem/1e9:.1f}GB')

import verl
print(f'verl: {verl.__version__}')

import vllm
print(f'vllm: {vllm.__version__}')
"
```

### 3.3 设置环境变量

```bash
# 添加到 ~/.bashrc 或训练脚本
export NCCL_P2P_DISABLE=1           # 4090 可能需要
export VLLM_USE_V1=1                # vLLM V1 引擎
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HYDRA_FULL_ERROR=1            # 完整错误信息
export RAY_LOGGING_LEVEL=DEBUG       # Ray 调试日志
```

### 3.4 验证 GPU 通信

```bash
# 测试 NCCL
python3 -c "
import torch
import torch.distributed as dist
dist.init_process_group('nccl')
print(f'Rank {dist.get_rank()}: NCCL OK')
dist.destroy_process_group()
"
```

---

## 4. Phase 1: 入门 — char_count (1 GPU)

### 4.1 目标
- 理解 verl 的完整 RL 流程：数据 → SFT → Reward → GRPO → 评估
- 理解 Hydra 配置系统
- 理解 DataProto 数据格式

### 4.2 安装

```bash
cd /data/home/yizhou/verl-recipe
pip install verl==0.6.1  # char_count 要求的版本
```

### 4.3 数据生成

```bash
cd /data/home/yizhou/verl-recipe/char_count
python3 create_dataset.py
# 生成：
# ~/data/char_count/sft/{train,test}.parquet
# ~/data/char_count/rl/{train,test}.parquet
```

### 4.4 SFT 训练

```bash
BACKEND=fsdp bash train_sft.sh
# 模型：SmolLM2-135M-Instruct (135M 参数，~270MB BF16)
# GPU：1 张 4090 足够
# 预期：~140 步，验证 score ~0.435
```

### 4.5 Checkpoint 合并

```bash
export CKPT_PATH=$HOME/experiments/char_count/models/sft/fsdp/global_step_140
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CKPT_PATH \
    --target_dir $CKPT_PATH/huggingface/
```

### 4.6 GRPO 训练

```bash
# 修改 train_grpo.sh 中的模型路径为 SFT checkpoint
# 然后运行：
bash train_grpo.sh
# 预期：5 epochs，验证 score ~0.6
```

### 4.7 学习要点

| 要点 | 对应代码/文件 | 关键理解 |
|------|-------------|---------|
| RL 数据格式 | `create_dataset.py` | prompt/reward_model/data_source 三列 |
| SFT 训练 | `train_sft.sh` | torchrun + FSDP + gradient checkpointing |
| Reward 函数接口 | `reward_function.py` | `(data_source, solution_str, ground_truth) -> float` |
| GRPO 启动 | `train_grpo.sh` | `python3 -m verl.trainer.main_ppo + Hydra overrides` |
| Hydra 配置 | `verl/trainer/config/ppo_trainer.yaml` | defaults 组合 + 命令行 override |
| Checkpoint 管理 | `verl/model_merger` | FSDP shards → HuggingFace 格式 |

### 4.8 动手实验

1. **修改 reward 函数**：把 binary reward (0/1) 改为 partial reward（答案接近给 0.5）
2. **调整 n_resp_per_prompt**：从 8 改为 4 或 16，观察训练曲线变化
3. **关闭 KL**：设置 `use_kl_loss=False`，观察是否 reward hacking
4. **添加 wandb**：修改 `trainer.logger='["console","wandb"]'`，可视化训练

---

## 5. Phase 2: 评估实战 — r1 (1-2 GPU)

### 5.1 目标
- 学习评估流程：批量生成 → 评分 → 统计
- 理解 best@N、maj@N 评估指标
- 学习多数据集 reward 路由

### 5.2 数据准备

```bash
cd /data/home/yizhou/verl-recipe/r1
python3 data_process.py
# 下载 AIME 2024, GPQA Diamond, LiveCodeBench, CNMO 2024
```

### 5.3 运行评估

```bash
# DeepSeek-R1-Distill-Qwen-1.5B，1-2 GPU 即可
bash run_r1_distill_qwen.sh
# 生成 → 评分 → 输出结果表
```

### 5.4 学习要点

| 要点 | 关键理解 |
|------|---------|
| 评估 vs 训练 | `verl.trainer.main_generation` 只做生成，不做训练 |
| Reward 路由 | `reward_func` 根据 `data_source` 分发到不同评分器 |
| Bootstrap 统计 | `bootstrap_metric()` 计算置信区间 |
| best@N | 生成 N 个 response，取最高分 |
| maj@N | 生成 N 个 response，取多数投票 |

---

## 6. Phase 3: 算法扩展 — gvpo (2-4 GPU)

### 6.1 目标
- 学习如何自定义算法（替换 loss 函数）
- 理解 GRPO vs GVPO 的核心区别
- 学习算法对比实验设计

### 6.2 数据准备

```bash
cd /data/home/yizhou/verl
python -m examples.data_preprocess.math_dataset.py
python -m examples.data_preprocess.gsm8k.py
# 生成 ~/data/gsm8k/ 和 ~/data/math/ 的 parquet 文件
```

### 6.3 4090 适配

原始脚本用 2x H800，需要修改：

```bash
# 修改 run_qwen2-7b_math_gvpo.sh
# 关键修改：
actor_rollout_ref.actor.fsdp_config.param_offload=True      # 原来 False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True   # 原来 False
actor_rollout_ref.rollout.gpu_memory_utilization=0.5         # 原来 0.6
actor_rollout_ref.rollout.tensor_model_parallel_size=4       # 原来 2
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4       # 原来 16
actor_rollout_ref.actor.ppo_mini_batch_size=64               # 原来 256
data.train_batch_size=256                                    # 原来 1024
data.max_response_length=512                                 # 原来 1024
trainer.n_gpus_per_node=4                                    # 原来 2
```

### 6.4 对比实验设计

```bash
# 实验 1: GRPO baseline
bash run_grpo_baseline.sh  # loss_mode="vanilla"

# 实验 2: GVPO
bash run_qwen2-7b_math_gvpo.sh  # loss_mode="gvpo"

# 对比指标：
# - val-core/mean@N: 验证准确率
# - critic/score/mean: 训练 reward
# - response_length/mean: 生成长度
```

### 6.5 学习要点

| 要点 | 对应代码 | 关键理解 |
|------|---------|---------|
| 自定义 loss | `gvpo_core_algos.py` | MSE-based loss，无需 importance sampling |
| 自定义 trainer | `gvpo_ray_trainer.py` | 修改 `_balance_batch` 保证同组数据在一起 |
| 自定义 entry | `gvpo_main_ppo.py` | 替换 trainer class |
| 算法配置 | Hydra override | `policy_loss.loss_mode="gvpo"` |

---

## 7. Phase 4: GenRM + 多任务 — genrm_remote (4 GPU)

### 7.1 目标
- 学习远程 Reward Model 的使用
- 理解 API 服务与训练的解耦
- 学习多任务 reward 设计

### 7.2 架构

```
GPU 0-3: vLLM serve GenRM (1.5B model)
GPU 4-7: RL Training (Qwen2.5-3B)
```

### 7.3 运行

```bash
cd /data/home/yizhou/verl-recipe/genrm_remote

# 终端 1: 启动 GenRM 服务
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve verl-team/GenRM-CI-Test-1.5B --served-model-name genrm-demo

# 终端 2: 启动 RL 训练
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_genrm_remote.sh
```

### 7.4 学习要点

| 要点 | 关键理解 |
|------|---------|
| GenRM 架构 | 训练和 reward 计算分离，通过 API 通信 |
| Reward 设计 | 先用 rule-based reward，再用 GenRM 精调 |
| 异步 reward | GenRM 推理是异步的，不阻塞训练 |
| 容错 | `reward_function.py` 有 retry + exponential backoff |

---

## 8. Phase 5: Multi-Turn Agent — langgraph_agent (4-8 GPU)

### 8.1 目标
- 理解 Agent Loop 状态机
- 理解 response_mask（LLM token vs tool token）
- 学习多轮 tool-use 训练

### 8.2 数据准备

```bash
cd /data/home/yizhou/verl-recipe
python recipe/langgraph_agent/example/create_dataset.py
# 生成 data/math_expression_tool/{train,test}.parquet
```

### 8.3 4090 适配

```bash
# 修改 run_qwen2.5_3b.sh
# 关键修改：
GPUS_PER_NODE=4                    # 原来 2
infer_tp=2                         # 保持
train_sp=2                         # 原来 4（减少通信）
offload=true                       # 保持
actor_rollout_ref.rollout.gpu_memory_utilization=0.6  # 原来 0.9
```

### 8.4 运行

```bash
GPUS_PER_NODE=4 bash recipe/langgraph_agent/example/run_qwen2.5_3b.sh
# 预期：~39 步达到 100% 准确率
```

### 8.5 学习要点

| 要点 | 对应代码 | 关键理解 |
|------|---------|---------|
| Agent Loop 接口 | `AgentLoopBase` | `run()` 返回 `AgentLoopOutput` |
| response_mask | `agent_loop.py:97` | LLM token=1, tool token=0 |
| Tool 定义 | `math_expression.py` | `@tool` 装饰器 + `@` 自定义运算符 |
| Agent 配置 | `agent.yaml` | name → class 映射 |
| 状态机 | `tool_agent_loop.py` | PENDING→GENERATING→PROCESSING_TOOLS→... |
| LangGraph 集成 | `react_agent_loop.py` | StateGraph + ReAct 模式 |

### 8.6 动手实验

1. **添加新 tool**：实现一个 `sqrt` tool，让 agent 学会调用
2. **修改 max_turns**：从 8 改为 4，观察 agent 行为变化
3. **分析 response_mask**：打印训练数据，验证 tool token 确实被 mask

---

## 9. Phase 6: 生产级 RL — dapo 7B (8 GPU)

### 9.1 目标
- 学习 DAPO 的三个核心创新
- 理解 Dynamic Sampling 和 Filter Groups
- 学习生产级训练的配置和调优

### 9.2 数据准备

```bash
cd /data/home/yizhou/verl-recipe/dapo
bash prepare_dapo_data.sh
# 下载 dapo-math-17k.parquet 和 aime-2024.parquet
```

### 9.3 4090 适配（关键）

```bash
# 创建 dapo_4090.sh，基于 run_dapo_qwen2.5_7b.sh 修改
# 核心修改：

# 1. 显存优化
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
actor_rollout_ref.rollout.tensor_model_parallel_size=4

# 2. 减小 batch
data.train_batch_size=32           # 原来 512
data.train_prompt_bsz=32           # 原来 512
actor_rollout_ref.actor.ppo_mini_batch_size=8
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1

# 3. 减少 response 数量
actor_rollout_ref.rollout.n=4      # 原来 16

# 4. 缩短序列
data.max_response_length=4096      # 原来 16384

# 5. 禁用 CUDA Graph
actor_rollout_ref.rollout.enforce_eager=True
```

### 9.4 运行

```bash
bash dapo_4090.sh
```

### 9.5 学习要点

| 要点 | 对应代码 | 关键理解 |
|------|---------|---------|
| Clip-Higher | `clip_ratio_low=0.2, clip_ratio_high=0.28` | 非对称 clip 允许更大正向更新 |
| Dynamic Sampling | `dapo_ray_trainer.py:235-294` | 过滤 std=0 的组 |
| Filter Groups | `algorithm.filter_groups.enable=True` | 持续采样直到有足够多样本 |
| Overlong Penalty | `reward_kwargs.overlong_buffer_cfg` | 惩罚超长 response |
| Token-level Loss | `loss_agg_mode=seq-mean-token-sum-norm` | 每个 response 等权 |
| Reward Manager | `dapo_ray_trainer.py` 继承 `RayPPOTrainer` | 自定义 metric 采集 |

---

## 10. Phase 7: Agent + Tool-Use — retool 7B (8 GPU)

### 10.1 目标
- 学习代码执行 Sandbox 集成
- 理解多轮 RL 训练的 reward 设计
- 学习 SFT 冷启动 + RL 的两阶段流程

### 10.2 4090 适配

```bash
# retool 的 7B 脚本已经针对 8 GPU 设计，但需要进一步优化
# 关键修改：
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
actor_rollout_ref.rollout.tensor_model_parallel_size=4
data.max_response_length=8192      # 原来 16384
```

### 10.3 学习要点

| 要点 | 关键理解 |
|------|---------|
| SFT 冷启动 | 先用代码增强的 CoT 数据做 SFT |
| 代码执行 | `daytona_sandbox_tool.py` 集成执行环境 |
| Tool 鼓励 Reward | 对负 reward 的 response，每轮 tool call 给 +0.1 |
| 多轮交互 | model → code block → execute → model → ... |

---

## 11. Phase 8: 异步训练 — async_flow (8 GPU)

### 11.1 目标
- 理解异步训练架构
- 学习 TransferQueue 数据流
- 理解 staleness 控制

### 11.2 先跑 0.5B 版本验证

```bash
cd /data/home/yizhou/verl-recipe/async_flow
# 0.5B 版本，1-2 GPU 即可
bash run/run_grpo_qwen3_0.5b.sh
```

### 11.3 再跑 7B 版本

```bash
# 4090 适配
bash run/run_grpo_qwen2.5_7b.sh
# 需要 offload + 减 batch
```

### 11.4 学习要点

| 要点 | 关键理解 |
|------|---------|
| 架构 | 4 个独立 Worker：ActorForward, ReferenceForward, RewardAdv, ActorTrain |
| TransferQueue | KV-based 异步数据流，支持 streaming |
| Staleness 控制 | `staleness=2` 是甜蜜点，过大性能下降 |
| 性能 | 3.81x 吞吐提升（vs 同步） |
| 背压 | `flow_control_queue` 防止生成过快 |

---

## 12. 各阶段时间估算

| Phase | 内容 | GPU | 预计耗时 | 产出 |
|-------|------|-----|---------|------|
| 0 | 环境搭建 | - | 2-4 小时 | 可用环境 |
| 1 | char_count | 1x 4090 | 2-4 小时 | SFT+GRPO 完整流程 |
| 2 | r1 eval | 1-2x 4090 | 1-2 小时 | 评估结果表 |
| 3 | gvpo | 4x 4090 | 4-8 小时 | GRPO vs GVPO 对比 |
| 4 | genrm_remote | 4x 4090 | 4-8 小时 | GenRM 训练结果 |
| 5 | langgraph_agent | 4x 4090 | 2-4 小时 | Agent 100% 准确率 |
| 6 | dapo 7B | 8x 4090 | 12-24 小时 | DAPO 训练曲线 |
| 7 | retool 7B | 8x 4090 | 12-24 小时 | Tool-use 训练结果 |
| 8 | async_flow | 8x 4090 | 8-16 小时 | 异步 vs 同步对比 |

**总计：约 3-4 周**（每天投入 4-8 小时）

---

## 13. 关键超参数速查表

### 通用参数

| 参数 | 默认值 | 4090 推荐值 | 说明 |
|------|--------|------------|------|
| `learning_rate` | 1e-6 | 1e-6 | 几乎所有 recipe 都用这个值 |
| `temperature` | 1.0 | 1.0 | 生成温度，固定为 1.0 |
| `grad_clip` | 1.0 | 1.0 | 梯度裁剪 |
| `entropy_coeff` | 0 | 0 | 大多数 recipe 不用 entropy bonus |
| `kl_coef` | 0.001 | 0.0 (DAPO) / 0.001 (其他) | KL 正则系数 |

### 显存相关

| 参数 | H100 值 | 4090 推荐值 | 说明 |
|------|---------|------------|------|
| `param_offload` | False | **True** | 参数 offload 到 CPU |
| `optimizer_offload` | False | **True** | 优化器状态 offload 到 CPU |
| `gpu_memory_utilization` | 0.85 | **0.5-0.6** | vLLM 显存占比 |
| `enforce_eager` | False | **True** | 禁用 CUDA Graph |
| `tensor_model_parallel_size` | 1-2 | **2-4** (7B) | vLLM 张量并行 |
| `ppo_micro_batch_size_per_gpu` | 8-16 | **1-4** | 每 GPU 微批大小 |
| `ppo_max_token_len_per_gpu` | 32768 | **4096-8192** | 每 GPU 最大 token 数 |

### 算法相关

| 参数 | GRPO | DAPO | GVPO |
|------|------|------|------|
| `clip_ratio_low` | 0.2 | 0.2 | 0.2 |
| `clip_ratio_high` | 0.2 | **0.28** | 0.2 |
| `loss_agg_mode` | token-mean | **seq-mean-token-sum-norm** | token-mean |
| `filter_groups.enable` | False | **True** | False |
| `n_resp_per_prompt` | 8 | 16 | 4 |
| `norm_adv_by_std_in_grpo` | True | True | **False** |
| `policy_loss.loss_mode` | vanilla | vanilla | **gvpo** |

---

## 14. 故障排查速查表

### OOM (显存不足)

```
症状：CUDA out of memory
排查：nvidia-smi 查看显存使用
解决：
1. 开启 param_offload + optimizer_offload
2. 减小 gpu_memory_utilization (0.5 → 0.4)
3. 增大 TP (2 → 4)
4. 减小 ppo_micro_batch_size_per_gpu (4 → 1)
5. 减小 ppo_max_token_len_per_gpu
6. 减小 max_response_length
7. 减小 n_resp_per_prompt
```

### NCCL 通信超时

```
症状：NCCL timeout after 600 seconds
排查：检查 GPU 间 P2P 是否可用
解决：
1. export NCCL_P2P_DISABLE=1
2. 减小 SP/TP（减少通信量）
3. 检查 PCIe 拓扑：nvidia-smi topo -m
```

### 训练 loss 变 NaN

```
症状：loss 为 NaN 或 inf
排查：
1. 检查 grad_norm（verl 自动记录）
2. 检查 reward 是否有异常值
3. 检查 IS 权重是否爆炸
解决：
1. 减小 learning_rate (1e-6 → 5e-7)
2. 检查 reward 函数是否有除零
3. 检查 max_response_length 是否过短
```

### vLLM 启动失败

```
症状：vLLM server timeout
排查：检查 GPU 显存是否被占用
解决：
1. 减小 gpu_memory_utilization
2. 确保 CUDA 版本 >= 12.8
3. 使用 enforce_eager=True
```

### Reward 全为 0

```
症状：critic/score/mean = 0
排查：
1. 检查 reward 函数是否正确加载
2. 检查 ground_truth 格式
3. 检查 solution_str 的 \boxed{} 提取
解决：
1. 打印 reward 函数输入输出
2. 检查数据格式是否匹配
```

### Dynamic Sampling 超时

```
症状：ValueError: Generated too many batches
排查：数据太简单或太难，所有 response 全对/全错
解决：
1. 增大 max_num_gen_batches
2. 检查数据难度是否匹配模型能力
3. 关闭 filter_groups 先跑通
```

---

## 附录：每阶段的 checkpoint 路径规划

```
~/experiments/verl-recipe/
├── char_count/
│   ├── sft/global_step_140/
│   └── grpo/global_step_xxx/
├── gvpo/
│   └── qwen2.5-7b/global_step_xxx/
├── genrm_remote/
│   └── qwen2.5-3b/global_step_xxx/
├── langgraph_agent/
│   └── qwen2.5-3b/global_step_xxx/
├── dapo/
│   └── qwen2.5-7b/global_step_xxx/
├── retool/
│   ├── sft/global_step_xxx/
│   └── grpo/global_step_xxx/
└── async_flow/
    └── qwen2.5-7b/global_step_xxx/
```

---

*本计划基于 verl v0.6.0+ 和 verl-recipe 最新代码。*
*硬件假设：8x RTX 4090 (24GB), 256GB+ 系统内存, 1TB+ SSD。*
