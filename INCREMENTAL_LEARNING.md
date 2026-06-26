# 增量学习笔记：verl-recipe 复现过程中的深层技术细节

> 基于前两轮分析的增量内容，聚焦工程实现细节、隐藏设计决策、以及复现中会踩的坑

---

## 目录

1. [数据管道的隐藏细节](#1-数据管道的隐藏细节)
2. [SFT 训练器 vs RL 训练器的本质区别](#2-sft-训练器-vs-rl-训练器的本质区别)
3. [Checkpoint 合并的工程细节](#3-checkpoint-合并的工程细节)
4. [动态批处理的数学原理](#4-动态批处理的数学原理)
5. [vLLM 集成的工程细节](#5-vllm-集成的工程细节)
6. [Hydra 配置系统的高级用法](#6-hydra-配置系统的高级用法)
7. [Agent Loop 的完整接口与隐藏机制](#7-agent-loop-的完整接口与隐藏机制)
8. [TransferQueue 的内部架构](#8-transferqueue-的内部架构)
9. [验证流程的隐藏逻辑](#9-验证流程的隐藏逻辑)
10. [4090 复现中的实际踩坑点](#10-4090-复现中的实际踩坑点)

---

## 1. 数据管道的隐藏细节

### 1.1 Chat Template 的延迟应用

**关键发现**：verl 在数据加载时**不**应用 chat template。`RLHFDataset.__getitem__` 只构建 `raw_prompt`（消息列表），实际的 `apply_chat_template` 延迟到 rollout 时由 `AgentLoopBase.apply_chat_template()` 执行。

**设计原因**：同一个数据集需要服务于不同架构的模型（Qwen、Llama、Gemma），它们的 chat template 不同。延迟应用避免了为每个模型重新处理数据集。

**实际影响**：
- 数据集中的 `prompt` 列是原始消息列表，不是 token IDs
- Tokenization 只在 `maybe_filter_out_long_prompts` 中发生一次（用于过滤）
- 训练时的 token 格式取决于所用模型的 tokenizer

### 1.2 多模态数据的正则分割

```python
# rl_dataset.py:299-384
# 用正则分割消息内容，找到 <image>/<video>/<audio> 占位符
parts = re.split("(<image>|<video>|<audio>)", content)
# 替换为结构化 dict：{"type": "image", "image": PIL.Image}
```

**隐藏细节**：每个模态有独立的 offset 计数器，确保多张图片/视频正确对应。所有图片/视频必须被消费，否则 assert 失败。

### 1.3 工具 Schema 的预加载

数据集在加载时就预加载了 tool schema（从 `tool_config_path`/`function_tool_path`），用于在过滤阶段正确估计 prompt 长度——因为 tool schema 会被嵌入到 chat template 中。

### 1.4 序列化优化

`__getstate__` 排除了 `dataframe` 的序列化（除非 `serialize_dataset=True`），允许高效的 checkpoint/resume——从原始 parquet 文件重新下载和 tokenize。

---

## 2. SFT 训练器 vs RL 训练器的本质区别

### 2.1 架构差异

| 组件 | SFT | RL (PPO/GRPO) |
|------|-----|---------------|
| Worker 数量 | 1 (TrainingWorker) | 3+ (Actor, Rollout, Ref, Critic) |
| 数据源 | Ground-truth responses | Model-generated responses |
| Loss | Cross-entropy on response tokens | PPO clip + advantage weighting |
| Reward | 无 | Rule-based 或 Model-based |
| 生成 | 无 | vLLM/SGLang rollout |
| KL 正则 | 无 | 可选 |

### 2.2 SFT 的动态批处理

SFT 支持 `use_dynamic_bsz`，通过 meta_info 中的 `max_token_len_per_gpu` 和 `micro_batch_size_per_gpu` 控制。与 RL 的区别是 SFT 的 batch 来自 DataLoader，而 RL 的 batch 来自 ReplayBuffer。

### 2.3 SFT 的 LoRA 支持

```python
# sft_trainer.py:100-136
def _get_lora_train_meta(config):
    return {
        "lora_adapter_path": config.model.lora_adapter_path,
        "lora_rank": config.model.lora_rank,
        "lora_alpha": config.model.lora_alpha,
        "task_type": config.model.task_type,
    }
```

SFT 训练的 LoRA adapter 在 checkpoint 合并时会被正确提取为 PEFT 兼容格式。

---

## 3. Checkpoint 合并的工程细节

### 3.1 FSDP 合并的 5 个阶段

1. **World Size 检测**：读取 `fsdp_config.json` 获取分片数量
2. **Device Mesh 提取**：检查 rank-0 的第一个 DTensor 获取 mesh 结构
3. **分片配置计算**：支持 `("fsdp",)` 和 `("ddp", "fsdp")`，**不支持** `("tp", "fsdp")`
4. **并行加载合并**：用 `ThreadPoolExecutor(max_workers=min(32, cpu_count))` 并行加载所有分片
5. **验证**：加载参考 HuggingFace 模型，逐 key 比较 shape、dtype、value（atol=1e-6）

### 3.2 LoRA Adapter 提取

```python
# base_model_merger.py:291-388
# 识别 LoRA 参数：name 中包含 "lora_"
# 提取 target_modules：key 结构的 .split(".")[-3]
# 推断 rank：从 weight shape
# 输出：adapter_config.json + adapter_model.safetensors
```

### 3.3 已知限制

- **不支持** `("tp", "fsdp")` mesh（会 raise NotImplementedError）
- 所有 tensor 被 cast 到 bfloat16
- 合并后的验证使用 `atol=1e-6, rtol=1e-6`

---

## 4. 动态批处理的数学原理

### 4.1 工作量估算公式

```python
# seqlen_balancing.py:27-46
# 针对 7B 模型的 FLOPs 校准公式
workload = 24576 * seqlen + seqlen^2
# 近似：12 * hidden_size^2 * seqlen + 2 * hidden_size * seqlen^2
# 其中 hidden_size = 4096
```

### 4.2 Karmarkar-Karp 算法

用于多路数字分割（NP-hard 问题的近似算法）：

1. 按值排序所有 items
2. 对于 `equal_size=True`：每 k 个连续 items 创建初始状态
3. 用最小堆维护状态，每次弹出 spread 最大的两个状态
4. 合并时将最大与最小配对，压回堆
5. 最终得到 k 个近似平衡的分区

### 4.3 Pipeline Bubble 减少

```python
# seqlen_balancing.py:460
# 排序后交错排列：大 batch 放中间，小 batch 放两端
micro_batches = sorted(by_workload, reverse=True)
interleaved = micro_batches[0::2][::-1] + micro_batches[1::2]
```

**直觉**：Pipeline 的 warm-up 和 cool-down 阶段只有部分 GPU 工作，把小 batch 放在这些位置可以减少空闲时间。

### 4.4 UID 分组保持

GRPO 要求同一 prompt 的所有 response 在同一 batch 中。`get_group_balanced_partitions` 先构建连续 group，计算每组工作量，再用 Karmarkar-Karp 分配。

---

## 5. vLLM 集成的工程细节

### 5.1 服务器启动流程

```
launch_server():
  1. 构建 CLI args (dtype, load_format, TP size...)
  2. 配置 data-parallel args (DP master address/port)
  3. 配置 LoRA (max_loras=1, max_lora_rank)
  4. node_rank=0: run_server (完整 HTTP server + uvicorn)
  5. node_rank>0: run_headless (worker-only, 无 API endpoint)
```

### 5.2 权重传输协议

```
BucketedWeightSender:
  1. 分配固定大小 GPU buffer (默认 2GB)
  2. 通过 ZMQ REQ socket 交换 CUDA IPC handle
  3. 将权重打包到 buffer，满了就发送并等 ACK
  4. 超大权重直接通过 reduce_tensor (CUDA IPC)
  5. 清理：关闭 socket，unlink IPC path，调用 ipc_collect()
```

**ZMQ 路径格式**：`ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{local_rank}.sock`

### 5.3 Sleep/Wake 协议

- `sleep(level=2)`：释放 weights + KV cache
- `sleep(level=1)`：只释放 KV cache（保留 weights）
- `wake_up(tags=["kv_cache", "weights"])`：恢复
- 权重更新后：`clear_kv_cache()` 重置 prefix cache、MM cache、encoder cache
- MTP 和 LoRA adapter 用 sleep level 1 保留

### 5.4 请求生成的安全边界

```python
max_tokens = max(1, min(max_tokens, max_possible_tokens))
```

处理 Qwen2.5-VL 的 image token 去重，支持 LoRA 请求，处理 abort 情况（空输出 → stop_reason="aborted"）。

---

## 6. Hydra 配置系统的高级用法

### 6.1 配置组合结构

```yaml
defaults:
  - model_engine: dp                    # 选择 dp/megatron/torchtitan/veomni
  - actor@actor_rollout_ref.actor: ${model_engine}_actor  # 动态映射
  - _self_  # 当前文件覆盖所有上面的
```

**关键模式**：`target@source: config_name` 将配置文件映射到配置树的特定位置。`${model_engine}` 插值允许切换后端。

### 6.2 V1 Trainer 三种模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `sync` | 标准同步 PPO | 基线，调试 |
| `colocate_async` | 异步，训练和推理共享 GPU | 生产环境 |
| `separate_async` | 异步，训练和推理分开 | 大规模训练 |

### 6.3 Replay Buffer 配置

```yaml
trainer:
  v1:
    sampler:
      max_off_policy_threshold: 8  # trajectory 最多跨 8 个模型版本
      max_off_policy_strategy: drop  # 超过阈值就丢弃（或 wait）
```

### 6.4 TransferQueue 后端

| 后端 | 传输方式 | 适用场景 |
|------|---------|---------|
| SimpleStorage | ZMQ in-memory | 单节点、开发调试 |
| MooncakeStore | RDMA | 多节点、生产环境 |

---

## 7. Agent Loop 的完整接口与隐藏机制

### 7.1 AgentLoopBase 完整接口

```python
class AgentLoopBase:
    # 构造参数
    trainer_config: DictConfigWrap      # 完整 trainer 配置
    server_manager: LLMServerClient     # OpenAI 兼容 LLM 客户端
    tokenizer / processor               # 文本/多模态处理
    dataset_cls: type[RLHFDataset]      # 多模态信息提取
    data_config: DictConfigWrap         # 数据集配置

    # 核心方法
    run(sampling_params, **kwargs) -> AgentLoopOutput  # 抽象方法

    # 辅助方法
    apply_chat_template(messages, tools, images, videos, audios, ...)
    process_multi_modal_info(messages)
    ct_build_initial_tokens(messages, tools)  # Continuous Token 模式
```

### 7.2 AgentLoopOutput 的隐藏字段

```python
class AgentLoopOutput(BaseModel):
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]           # 1=LLM, 0=tool
    response_logprobs: Optional[list[float]]
    routed_experts: Optional[list]     # MoE expert routing replay
    multi_modal_data: Optional[dict]
    reward_score: Optional[float]
    num_turns: int
    metrics: AgentLoopMetrics          # generate_sequences 时间、tool_calls 时间等
    extra_fields: dict[str, Any]       # 包含 teacher_ids/teacher_logprobs
    mm_processor_kwargs: dict           # processor kwargs 必须保持对齐
```

### 7.3 as_dict() 的隐藏逻辑

`reward_score` 被放在 `rm_scores` 的最后一个非 padding 位置。`teacher_ids`/`teacher_logprobs` 从 `extra_fields` 中提取。

### 7.4 后处理的 Position ID 计算

```python
# agent_loop.py:721-858
# 文本模型：标准 1D position ID
# Qwen2-VL：M-RoPE 3D/4D position ID
position_ids = processor.get_rope_index(input_ids, image_grid_thw, video_grid_thw)
```

### 7.5 注册系统

```python
_agent_loop_registry = {}

@register(agent_name)
class MyAgentLoop(AgentLoopBase):
    ...

# agent.yaml 中映射
- name: math_expression
  _target_: recipe.langgraph_agent.example.math_expression.MathExpressionReactAgentLoop
```

---

## 8. TransferQueue 的内部架构

### 8.1 三层设计

```
TransferQueueManager (单一 Ray actor, max_concurrency=100)
    │
    ├── TransferQueueShard (分布式 Ray actors, 按 node IP round-robin)
    │   └── ZMQ ROUTER socket
    │
    └── TransferQueueClient (worker 端)
        └── ZMQ DEALER socket
```

### 8.2 列式存储

```python
class ExperienceTable:
    # 列式存储：indices[col_name][global_id] = (MemorySegment, byte_offset, byte_length)
    # MemorySegment 包装 zmq.Frame，支持引用计数
    # 共享列 (prompt, prompt_length, prompt_uuid)：每组一个 entry，所有 n_samples_per_prompt 槽位索引到同一个
```

### 8.3 写入路径

```
1. 分配 indexes (顺序或 UUID)
2. 分割 shared vs non-shared columns
3. Shared columns：只发到一个 shard，标记已写
4. Non-shared columns：按 global_idx_to_shard 路由
5. 序列化：tensors → raw numpy bytes; objects → pickle
6. ZMQ DEALER → ROUTER: PUT command → ACK
```

### 8.4 读取路径

```
1. Manager 根据 consumer、staleness、version 分配 shard+indexes
2. Client 发送 GET 到每个 shard
3. Shard 从 ExperienceTable.get_batch() 读取
4. Client 反序列化并返回
```

### 8.5 Staleness 控制

`versions_to_indexes` 追踪哪些 indexes 属于哪个模型版本。采样时按 `allowed_staleness` 和 `latest_version` 过滤。

---

## 9. 验证流程的隐藏逻辑

### 9.1 多轮 Session 处理

```python
# 对于 multi-output agent loop，keys 格式：{uid}_{session_id}_{index}
# 验证逻辑找到每个 session 的最高 index 输出作为"最终"输出
session_max = {}
for key in keys:
    uid, session_id, index = parse_key(key)
    session_max[(uid, session_id)] = max(session_max.get(...), index)
```

### 9.2 生成日志的确定性采样

```python
# 用 seed=42 确定性打乱后取 top-N 样本
rng = np.random.RandomState(42)
shuffled_indices = rng.permutation(len(generations))
samples = [generations[i] for i in shuffled_indices[:log_val_generations]]
```

### 9.3 异步 Dump

```python
# trainer_base.py:952-973
# 用 ThreadPoolExecutor(max_workers=1) 异步写 JSONL
executor = ThreadPoolExecutor(max_workers=1)
future = executor.submit(_dump_generations, generations)
# futures 被追踪，异常会被提前暴露
```

### 9.4 Pre-Training 验证

`val_before_train=True` 在训练前运行一次验证。`val_only=True` 在初始验证后直接退出（不训练）。

---

## 10. 4090 复现中的实际踩坑点

### 10.1 NCCL P2P 问题

**症状**：多卡训练时 NCCL 报错或超时
**原因**：4090 的 PCIe P2P 可能不被 NCCL 正确识别
**解决**：
```bash
export NCCL_P2P_DISABLE=1
```

### 10.2 vLLM CUDA Graph 内存开销

**症状**：开启 CUDA Graph 后 OOM
**原因**：CUDA Graph 需要预分配固定大小的内存，4090 的 24GB 不够
**解决**：
```bash
actor_rollout_ref.rollout.enforce_eager=True  # 禁用 CUDA Graph
```

### 10.3 7B 模型的 TP 配置

**症状**：vLLM 启动失败，显存不足
**原因**：7B 模型 BF16 权重 ~14.5GB，单卡装不下模型+KV Cache
**解决**：
```bash
# 公式：gpu_memory_utilization * 24GB * TP > 2 * model_params_bytes
# 7B: 0.5 * 24 * 2 = 24GB > 28GB? 不够！需要 TP=4 或 gpu_memory_utilization=0.6
actor_rollout_ref.rollout.tensor_model_parallel_size=4
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

### 10.4 Offload 导致的训练速度下降

**症状**：训练速度比预期慢很多
**原因**：`param_offload=True` 和 `optimizer_offload=True` 需要在每个 micro-batch 时在 CPU 和 GPU 之间传输数据
**解决**：这是 4090 的必然 tradeoff，只能通过减小 batch size 来减少传输次数

### 10.5 Dynamic Sampling 超时

**症状**：`ValueError: Generated too many batches`
**原因**：数据太简单（全对）或太难（全错），没有学习信号
**解决**：
```bash
# 方案 1：增大采样预算
algorithm.filter_groups.max_num_gen_batches=20

# 方案 2：关闭 filter groups 先跑通
algorithm.filter_groups.enable=False
```

### 10.6 SP (Sequence Parallelism) 在 PCIe 上的性能

**症状**：SP=4 或 SP=8 时训练极慢
**原因**：SP 需要频繁的 all-reduce 通信，PCIe 带宽远低于 NVLink
**解决**：
```bash
# 减小 SP 到 2，或完全不用 SP（SP=1）
actor_rollout_ref.actor.ulysses_sequence_parallel_size=2
```

### 10.7 Reward 函数的 silent failure

**症状**：训练正常但 reward 全为 0
**原因**：reward 函数的 `except Exception: return 0` 掩盖了 bug
**解决**：
```python
# 临时修改 reward 函数，去掉 try/except，暴露真实错误
def char_count_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    # 去掉 try/except，让异常传播
    last_boxed_string = math_reward.last_boxed_only_string(solution_str)
    ...
```

### 10.8 Checkpoint 路径问题

**症状**：GRPO 训练找不到 SFT checkpoint
**原因**：SFT checkpoint 需要先合并为 HuggingFace 格式
**解决**：
```bash
# 必须先执行 model_merger
python3 -m verl.model_merger merge --backend fsdp \
    --local_dir $CKPT_PATH \
    --target_dir $CKPT_PATH/huggingface/
# 然后在 GRPO 脚本中指向 huggingface/ 目录
```

### 10.9 transformers 版本冲突

**症状**：`s_aux=None` 错误
**原因**：transformers 5.6.0 的 flash-attention 路径有 bug
**解决**：
```bash
pip install transformers!=5.6.0  # 用 5.6.1 或更早版本
```

### 10.10 tensordict 版本约束

**症状**：导入错误或运行时异常
**原因**：verl 要求 `tensordict>=0.8.0,<=0.10.0,!=0.9.0`
**解决**：
```bash
pip install "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
```

---

## 附录：复现 Checklist

### Phase 0: 环境搭建
- [ ] Python >= 3.10
- [ ] CUDA >= 12.8
- [ ] PyTorch 版本与 vLLM 兼容
- [ ] `pip install verl==0.6.1`（或对应版本）
- [ ] `NCCL_P2P_DISABLE=1`
- [ ] `nvidia-smi` 确认 8 张 4090 识别
- [ ] `nvidia-smi topo -m` 检查 PCIe 拓扑

### Phase 1: char_count
- [ ] 数据生成成功（检查 parquet 文件存在）
- [ ] SFT 训练完成（验证 score ~0.435）
- [ ] Checkpoint 合并成功（huggingface/ 目录生成）
- [ ] GRPO 训练完成（验证 score ~0.6）
- [ ] wandb 日志正常

### Phase 2: r1 eval
- [ ] 数据下载成功
- [ ] 生成完成
- [ ] 评分结果与 README 匹配

### Phase 3: gvpo
- [ ] GSM8K + MATH 数据下载
- [ ] offload 开启，batch 减小
- [ ] GRPO baseline 跑通
- [ ] GVPO 实验完成
- [ ] 对比结果有意义

### Phase 5: langgraph_agent
- [ ] 数据生成成功
- [ ] agent.yaml 配置正确
- [ ] Multi-turn 训练完成
- [ ] 验证准确率 ~100%

### Phase 6: dapo 7B
- [ ] 数据下载成功
- [ ] 4090 适配脚本创建
- [ ] offload + TP=4 + 减 batch
- [ ] 训练曲线正常（reward 上升，length 稳定）
- [ ] Dynamic Sampling 工作正常

---

*本文档在复现过程中持续更新。*
