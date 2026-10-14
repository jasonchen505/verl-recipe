# verl-recipe 深入学习文档

> 学习对象：`verl-project/verl-recipe`（上游），fork 为 `jasonchen505/verl-recipe`。
> 学习时间：2026-10-14。只学习，不改代码，不开 PR，不评论 issue。
> 数字均为当日 API 实测，不是 README 转述。未本地运行任何 recipe，凡未验证处明确标注。

## 1. 定位与上游关系

- 一句话：社区贡献的、基于 verl 的**端到端 RL 训练 recipe 合集**。每个 recipe 是一个可复现的训练示例（算法 + 数据 + 配置 + 启动脚本），不是 verl 核心库。
- 与 verl 上游的关系：`verl-recipe` 可作为 verl 的 git submodule 使用，保持向后兼容的 `verl/recipe` 路径（`git submodule update --init --recursive recipe`）。即 verl 仓内 `recipe/` 就是这个仓的快照。
- 上游：`verl-project/verl-recipe`，Apache-2.0，创建于 2025-11-25，默认分支 `main`。
- 规模（2026-10-14 实测）：338 ⭐，165 forks，663 个文件（Python 300、shell 130、YAML 92、README 43、`REQUIRED_VERL.txt` 33）。
- 组织方式：**一个 recipe 一个目录**，各 recipe 相对独立，自带 README、版本钉选文件、配置、启动脚本。顶层目录 35 个（见第 3 节）。
- 贡献门槛观察：没有 CONTRIBUTING、PR 模板、issue 模板；`CODEOWNERS` 是空文件；CI 只跑 ruff + verl_tinker 单测（见第 8 节）。补 example、修 doc、修小 bug 是这个仓的常态贡献形态，大量 open PR 都是新增 recipe。

## 2. Recipe 通用结构与数据流

一个典型 recipe（以 char_count / dapo / retool 为范式）包含：

| 部件 | 作用 |
| --- | --- |
| `README.md` | 算法背景、数据集、复现结果、运行命令 |
| `REQUIRED_VERL.txt` | 钉选 verl 版本（canonical，见第 7 节） |
| 数据准备脚本 | 下载 / 预处理数据成 parquet（`create_dataset.py`、`prepare_*.py`、`prepare_dapo_data.sh`） |
| 奖励函数 | 可验证奖励（`reward_function.py`、`compute_score`），常复用 `verl.utils.reward_score`（`math_reward`、`math_dapo`） |
| 训练入口 | 两种形态：(a) 直接调 `python3 -m verl.trainer.main_ppo` + Hydra 命令行覆盖（多数 recipe）；(b) 自带 `main_*.py` + 自定义 trainer（dapo、gkd） |
| 启动脚本 | `run_*.sh` / `test_*.sh`，内含完整超参、GPU 数、Ray 地址 |
| `config/*.yaml` | Hydra 配置，继承 verl 的 `ppo_trainer` 等默认配置，只覆盖差异部分 |

端到端数据流（标准 RL recipe）：

```
parquet 数据（prompt + ground_truth）
→ RLHFDataset / create_rl_dataset（tokenize、按 max_prompt_length 截断）
→ RayPPOTrainer / 自定义 trainer（Ray 单控多 worker）
→ Actor rollout（vLLM / SGLang / Dynamo，n 路采样）
→ 奖励：规则验证奖励函数 / GenRM / 工具沙箱反馈
→ 优势估计（GRPO / DAPO / PPO，algorithm.adv_estimator）
→ Actor 更新（FSDP / Megatron）+ 可选 Critic / Ref
→ 定期 val（trainer.test_freq）+ checkpoint 保存
→ verl.model_merger 把 checkpoint 合并为 HF 格式 → generation + 评测脚本
```

关键点：recipe 本身**不实现 RL 算法核心**，算法在 verl 库里；recipe 做的是接线（数据、奖励、工具、配置、启动命令）。所以读 recipe 的重点是“它把哪些旋钮拧到了什么值、为什么”，而不是算法推导。

## 3. 目录与模块地图

顶层目录（35 个）与一句话定位：

| 目录 | 定位 |
| --- | --- |
| `char_count` | 入门 recipe：数字符 RLVR，SmolLM2-135M，8GB 消费级 GPU 可跑 |
| `dapo` | DAPO 算法复现（Qwen2.5-32B，AIME 2024 50–52%），本仓最重的 recipe 之一 |
| `retool` | ReTool：RL 训练策略性工具使用（代码沙箱 + 数学推理） |
| `gkd` | GKD：异步 on-policy 知识蒸馏（Megatron 后端，教师 top-k 分布） |
| `r1` | DeepSeek R1 复现（开发中，含评测结果表） |
| `open_math_reasoning` | OpenMathReasoning SFT 演示（Qwen3-8B，FSDP2 / Megatron 双后端） |
| `langgraph_agent` | LangGraph ReactAgent 多轮 rollout 示例 |
| `verl_tinker` | Tinker 兼容 HTTP server + cookbook client 示例（唯一有真正 CI 单测的 recipe） |
| `dynamo` | Dynamo rollout 后端（vLLM / SGLang + KV-aware router） |
| `async_flow` | AsyncFlow：训推异步流水线（TransferQueue + CheckpointEngine 解耦） |
| `fully_async_policy` | `verl.experimental.fully_async_policy` 启动示例（含 GLM-5.2 Ascend） |
| `partial_rollout` | 同步 RL 的 partial rollout：长尾样本中断 + 跨步恢复（不要和 fully-async 的 partial rollout 混淆，README 明确警告） |
| `reward_queue` | 解耦推理与奖励计算（fully_async 流水线里的奖励队列） |
| `echo` | ECHO：GRPO + 环境观测 token 上的 on-policy CE 损失（终端 agent） |
| `tmax` | TMax：长时终端 agent RL 全异步 recipe |
| `specRL` | specRL：用 co-adapted diffusion drafter 加速 rollout（`histoSpec` + `rl_spec` 两个子目录） |
| `atropos` | Atropos 环境 GRPO（rollout 与打分由 Atropos 环境处理） |
| `collabllm` | CollabLLM：多轮协作对话训练（ICML 2025） |
| `dance_grpo` | DanceGRPO（顶层无 README，README 在子目录 `dance_grpo_fsdp` / `dance_grpo_mindspeed_mm`） |
| `deepeyes` | DeepEyes：RL 激励“用图像思考”，多轮视觉工具调用 |
| `entropy` | Entropy Mechanism（LLM 推理 RL 的熵机制） |
| `fapo` | FAPO：Flawed-Aware Policy Optimization |
| `flowrl` | FlowRL：用 flow balance 匹配奖励分布 |
| `genrm_remote` | Generative Reward Model 远程调用（vLLM server + RL） |
| `gvpo` | GVPO：Group Variance Policy Optimization（NeurIPS 2025） |
| `infigui-g1` | InfiGUI-G1：GUI grounding 的 AEPO（Adaptive Exploration Policy Optimization） |
| `low_precision` | 低精度 recipe 合集（FP8 端到端 / FP8 仅 rollout） |
| `minicpmo` | MiniCPM-o recipe（README 只有骨架，内容最少的一档） |
| `nemo_gym` | NeMo Gym 相关（**顶层及全目录无 README**，入口需直接读代码） |
| `prime` | PRIME recipe（README 只有骨架） |
| `qat` | NVFP4 QAT（FSDP / Megatron 双后端，vLLM 推理不爆 KL） |
| `rep_exp` | Representation-Based Exploration（arXiv 2510.11686） |
| `spin` | SPIN：自博弈微调 |
| `spo` | SPO：Single-stream Policy Optimization（arXiv 2509.13232） |
| `sppo` | SPPO：自博弈偏好优化（arXiv 2405.00675） |

另有根目录遗留文件（见第 11 节“仓库卫生”）。

## 4. 核心 recipe 逐个讲

### 4.1 char_count（入门，必读第一个）

- 任务：数单词里某个字符出现几次。prompt：`How many {char} are there in {word}?`。单词随机生成（a–z 均匀采样），长度与答案分布均匀。
- 两个降低难度的小技巧：单词字母间加 `-`（让每个字符大概率被切成同一 token）；SFT 数据带 CoT（逐字母比对，最后 `\boxed{答案}`）。
- 数据：`create_dataset.py` 生成 train / val parquet，SFT 和 RL 共用。
- 奖励：`reward_function.py` 的 `char_count_reward_function`，取最后 `\boxed{}` 内容与 ground truth 比对，1/0 二值，异常返回 0。复用 `verl.utils.reward_score.math_reward` 的 boxed 解析。
- 训练：SFT 1 epoch（val ~0.435）→ `verl.model_merger merge` 合并 checkpoint → GRPO 2 epoch（val ~0.6）。
- 启动：`train_sft.sh`（`BACKEND=fsdp|megatron`）、`train_grpo.sh`（调 `python3 -m verl.trainer.main_ppo`）。
- 关键超参（`train_grpo.sh`）：`train_batch_size=128`，`max_prompt_length=128`，`max_response_length=128`，`rollout.n=8`（每 prompt 采 8 条），`actor_rollout_ref.actor.optim.lr=1e-6`，`ppo_mini_batch_size=16`，`use_dynamic_bsz=True`，`kl_loss_coef=0.0`（关 KL），`gpu_memory_utilization=0.7`，单卡（`n_gpus_per_node=1`），`save_freq=-1`（不存中间 checkpoint），`test_freq=5`。
- 版本钉选：`MODE=pinned_tag`，`verl[verl-core]==0.9.1`（`REQUIRED_VERL.txt`）。
- 为什么适合入门：文件只有 6 个，全流程（数据 → SFT → 合并 → GRPO → 评测）在一个目录里闭环，超参少，单卡可跑。

### 4.2 dapo（算法复现，DAPO）

- 论文：DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization），arXiv 2503.14476。Qwen2.5-32B base，AIME 2024：完整版 52%，去掉 Dynamic Sampling 50%，早期版（无 token-level PG loss、无 dynamic sampling）44%。
- 入口：`main_dapo.py` 的 `DAPOTaskRunner`，复用 verl 的 `run_ppo`，只把 trainer 类换成 `RayDAPOTrainer`（`dapo_ray_trainer.py`，约 22KB）。即“标准 PPO 脚手架 + 自定义 trainer”。
- 配置：`config/dapo_trainer.yaml`，继承 `ppo_trainer`，覆盖 `reward.reward_manager.name=dapo`、`overlong_buffer_cfg`（默认关）、`algorithm.filter_groups`（动态采样过滤组，默认关）。
- 启动脚本极多（约 25 个）：覆盖 Qwen2.5 / Qwen3 / Qwen3-MoE / VL / GPT-OSS / GLM / DeepSeek 671B、FSDP / FSDP2 / Megatron、NPU（Ascend）、FP8、rollout_corr 等组合。另有 `dapo_transfer_queue/` 变体（TransferQueue 版 trainer，`main_dapo.py` + `dapo_ray_trainer.py` 独立一份，约 30KB）。
- 数据：`prepare_dapo_data.sh` 下载到 `${HOME}/verl/data`；`runtime_env.yaml` 供 Ray 集群。
- 注意：README 里的复现表引用 `recipe/dapo/...`（submodule 路径），在本仓根目录视角下对应 `dapo/...`。两个路径写法混用，读脚本时以实际文件位置为准。
- 坑（issue #108，仍 open）：DAPO recipe 曾因引用已不存在的 `verl.utils.rollout_skip` 在 verl main 上启动失败（verl 已重构为 `verl/utils/skip/rollout_skip.py`）。教训：recipe 与 verl 版本是强耦合的，`REQUIRED_VERL.txt` 不是装饰。

### 4.3 retool（工具使用 RL）

- 论文：ReTool（arXiv 2504.11536）。基座 Qwen2.5-32B-Instruct；SFT 数据 JoeYing/ReTool-SFT；RL 数据 DAPO-Math-17k；验证集 aime_2025。
- 两阶段：SFT 冷启动（学会调工具 + 读执行结果）→ RL 动态交互（think-execute-feedback 循环）。
- 关键代码 `retool.py`：
- `CustomSandboxFusionTool`：从模型输出抽 ` ```python... ``` ` 代码块；**若最后一行非空且不是 print，自动包一层 `print(...)`**（保证沙箱总有输出可反馈）；经 `execution_pool.execute.remote` 异步执行，返回 `(result, None, None)`（沙箱无 score）。
- `CustomRLHFDataset`：对 AIME_2024 / aime_2025 用 `map_fn`，其余用 `map_fn2` + `num_proc=16`。注意：`map_fn2` 的 `num_proc=16` 在 issue #19、#29 里出现过加载失败报告（环境相关，未定论）。
- `compute_score`：`math_dapo.compute_score(..., strict_box_verify=True)`。
- 工具配置：`sandbox_fusion_tool_config.yaml` / `daytona_tool_config.yaml`（Daytona 沙箱备选）。
- 结果（README 声称，未独立验证）：SFT 6 epoch 后 aime_2025 acc@30 0.24；GRPO 150 步后 0.6；PPO 250 步后（README 截断处，未读全）。
- 版本钉选：README 正文写 `pip install verl==0.6.1`，以 `REQUIRED_VERL.txt` 为准（两者是否一致未逐行核对，见第 12 节）。
- 相关 open issue：#39（`run_qwen2_7b_dapo.sh` 执行失败，Ray 任务失败但训练继续）、#29（`_contextvars.ContextVar` pickle 错误）、#19（`map_fn2` 加载失败）、#28（解释器返回图片能否作为结果）。

### 4.4 gkd/megatron（知识蒸馏，与 PR #160 直接相关）

- 背景：on-policy KD，学生用当前策略采样，教师返回 soft top-k token 分布，学生用 token-wise 稀疏 KL 目标模仿教师。同步版三阶段（rollout → 教师知识 → 优化）互相等待有气泡，于是引入异步调度重叠三阶段，换少量 on-policy 严格性换吞吐。
- 结构：`main_gkd.py`（入口）+ `ray_trainer.py`（约 31KB，Ray trainer）+ `megatron_workers.py`（约 37KB，Megatron worker）+ `megatron_distill_losses.py`（约 26KB，4 种蒸馏损失）+ `megatron_kl_loss.py` + `megatron_utils.py` + `teacher/`（vLLM 教师 server：`vllm_engine.py`、`worker.py`、`proxy.py`、`client.py`、`start_server.sh` / `join_server.sh`）+ `config/on_policy_distill_trainer.yaml`（约 15KB）+ `run_moonlight_dsv3_training.sh` + `test_qwen*.sh`。
- 4 种损失（`megatron_distill_losses.py` 文件头自述）：
1. `kl`：KL(P_topk ‖ Q_full)，教师 top-k 截断的前向 KL。
2. `rkl`：KL(Q_hat_topk ‖ P_hat_topk)，重归一化 top-k 上的纯反向 KL。
3. `kl_rkl`：`(1-r)*KL + r*RKL`。
4. `jsd`：JSD_beta(P_topk, Q_full)，含 Q‖M 在 top-k 之外的解析 rest 项。
- JSD 实现要点：`_VocabParallelJSDivergence` 是自定义 `torch.autograd.Function`（forward + backward 手写，backward 里 `d/dz JSD = (1-beta) * Q * (A - KL(Q‖M))`）；`beta` 在 forward 里被 clamp 到 `(1e-6, 1-1e-6)`（第 442 行：`beta = min(max(float(beta), 1e-6), 1.0 - 1e-6)`）；默认 `beta=0.5`；`build_vocab_parallel_distill_loss(cfg)` 按配置构造。
- 坑（issue #151，见第 10 节）：`beta: 1.0` / `0.0` 被 clamp 后 loss 近似为 0，且全程无任何警告。默认 `beta=0.5` 不受影响，所以容易漏。
- 版本钉选：`MODE=rolling`，`VERL_COMMIT=bcb638649a50e58494a8ddd92085ad1174f674b8`，另有 `RECIPE_SUBMODULE_COMMIT` 与 `RECIPE_FOLDER_LAST_COMMIT` 三级钉选（见第 7 节）。

### 4.5 r1 / open_math_reasoning / langgraph_agent

- `r1`：DeepSeek R1 复现，README 明确写“under development”，指向 verl#708 的 TODO。内容以评测结果表为主（DS-R1-Distill-Qwen2.5-1.5B：GPQA 35.3 / LiveCodeBench 16.9 / AIME 2024 30.4；DS-R1：GPQA 67.7 / LiveCodeBench 64.7 / AIME 86.7，k=1）。复现代码本身不多，更多是评测入口。
- `open_math_reasoning`：SFT 演示，不是复现 AIMO-2（README 明确说目标不是复现）。数据：`nvidia/OpenMathReasoning`（cot 部分）+ aime24/25 评测集；模型 Qwen3-8B-Base；FSDP2 / Megatron 双后端；流程：预处理 → SFT → `model_merger` 合并 → generation → 评测。脚本：`run_sft_qwen3_8b.sh`、`run_generation.sh`、`run_eval.sh`。#161（Intel GPU）改了这个 recipe 的脚本与钉选文件。
- `langgraph_agent`：LangGraph ReactAgent 多轮 rollout 示例，解数学表达式。结构：顶层 README 很薄，实质在 `example/`（`example/README.md` 为 walkthrough）。open issue 三个都与它有关：#139（chat template 返回 BatchEncoding 导致 crash，PR #140 修）、#10（`MathExpressionReactAgentLoop` 无 `tools` 属性，verl==0.7.0 下）、#5（能否 RL 训练 deep research 型 agent，问题贴）。

### 4.6 其他值得知道的 recipe（按机制分组）

- 异步 / 解耦：`async_flow`（TransferQueue + CheckpointEngine，训推解耦 + Staleness + 权重异步同步，README 为中文）、`fully_async_policy`（`verl.experimental.fully_async_policy` 官方示例，训练与 rollout 在独立 Ray placement group 并发）、`reward_queue`（把奖励计算从生成流水线拆出来）、`partial_rollout`（同步循环内长尾中断 + 恢复）。
- 后端 / infra：`dynamo`（Dynamo KV-aware router + vLLM/SGLang，支持训练/rollout 同卡或分卡、权重更新、engine sleep/wake）、`verl_tinker`（Tinker 兼容 HTTP server，SFT/RL/蒸馏/checkpoint/rollout 走 verl GPU worker）。
- 终端 agent：`echo`（ECHO，GRPO + 观测 token CE）、`tmax`（TMax，TMax-15K-Harbor 数据，Harbor 环境）。两者都由 RFC issue #132 / #133 提出，分别由 PR #142 / #141 落地（已合并进 main）。
- 算法变体：`spo`（单流策略优化）、`fapo`（flawed-aware）、`gvpo`（组方差）、`entropy`（熵机制）、`flowrl`（flow balance 匹配奖励分布）、`spin` / `sppo`（自博弈）、`prime`（骨架）、`rep_exp`（表征探索）。
- 多模态 / 视觉：`deepeyes`（图像思考 + 多轮视觉工具）、`infigui-g1`（GUI grounding AEPO）、`minicpmo`（骨架）。
- 精度 / 硬件：`qat`（NVFP4 QAT）、`low_precision`（FP8 合集）、`dapo` 内大量 NPU 脚本、`fully_async_policy/examples/ascend`（GLM-5.2 Ascend 全异步）、`dynamo`（Intel GPU 相关见 #161，但 #161 只动了 char_count 与 open_math_reasoning）。

## 5. 关键代码路径

| 路径 | 作用 |
| --- | --- |
| `install_verl.sh` | 根目录唯一安装脚本，读 `REQUIRED_VERL.txt` 装对应 verl（见第 7 节） |
| `char_count/create_dataset.py` → `char_count/reward_function.py` → `char_count/train_grpo.sh` | 最小闭环：数据 → 奖励 → `verl.trainer.main_ppo` |
| `dapo/main_dapo.py` → `dapo/dapo_ray_trainer.py` → `dapo/config/dapo_trainer.yaml` | DAPO 入口：复用 `run_ppo`，换 trainer 类 |
| `retool/retool.py` → `retool/run_qwen2-32b_dapo.sh` | 工具 RL：沙箱工具 + 自定义数据集 + `math_dapo` 评分 |
| `gkd/megatron/main_gkd.py` → `gkd/megatron/ray_trainer.py` → `gkd/megatron/megatron_workers.py` → `gkd/megatron/megatron_distill_losses.py` | KD 流水线：入口 → Ray trainer → Megatron worker → 蒸馏损失 |
| `gkd/megatron/teacher/vllm_engine.py` + `teacher/worker.py` + `teacher/proxy.py` | 教师 server：vLLM 引擎 + worker + 代理 |
| `open_math_reasoning/prepare_nvidia-OpenMathReasoning_sft.py` → `run_sft_qwen3_8b.sh` → `model_merger` → `run_generation.sh` → `run_eval.sh` | SFT 演示全链路 |
| `verl_tinker/src/verl_tinker/` + `verl_tinker/configs/` + `verl_tinker/client_examples/` | Tinker server 实现 + 配置 + 客户端示例 |
| `dynamo/`（README + config + patches） | Dynamo 后端接线；`dynamo/patches/vllm024` 为 vLLM 0.2.4 补丁 |
| `async_flow/`（`agent_loop/` + `utils/checkpoint_engine` + `utils/transfer_queue` + `workers/`） | 异步流水线实现 |

通用规律：**recipe 的 Python 代码大多是“verl 公共接口的组装层”**（`verl.trainer.main_ppo`、`verl.utils.*`、`verl.tools.*`、`verl.experimental.*`），自带 trainer/worker 的只有 dapo、gkd、dance_grpo、rep_exp、verl_tinker 等少数。

## 6. 配置与复现方式

### 6.1 `REQUIRED_VERL.txt`：每个 recipe 的版本钉选（canonical）

每个 recipe 目录下都有一个 `REQUIRED_VERL.txt`（共 33 个，部分 recipe 共享或缺失）。字段含义：

- `UPSTREAM`：verl 上游地址（现为 `verl-project/verl`，历史上有 `volcengine/verl`）。
- `MODE`：`pinned_tag`（钉 release，如 char_count 的 v0.9.1）/ `rolling`（跟 main，但附精确 commit）/ `mixed` / `rolling_with_baseline` / `dapo reproduction` 等多变体模式。
- rolling 模式的三级钉选：`VERL_COMMIT`（verl 主仓 `git rev-parse HEAD`）、`RECIPE_SUBMODULE_COMMIT`（该 verl 修订下 `recipe/` submodule 的 gitlink）、`RECIPE_FOLDER_LAST_COMMIT`（仅该 recipe 子目录的最后 commit）。
- `PIP_INSTALL`：可直接复制的安装命令；`REFRESH=`：重新计算上述字段的命令。
- 硬件插件：`--hardware-plugin intel` 时用 `<NAME>_PLUGIN_COMMIT`（如 char_count 的 `INTEL_PLUGIN_COMMIT`）。

### 6.2 `install_verl.sh`：一键安装器

- `./install_verl.sh --list` 列出全部 recipe + 钉选的 pip 行；`--recipe NAME --show` 只打印不执行（dry-run）；`--method pip`（默认，跑 `PIP_INSTALL`）/ `--method git`（clone verl 到 `--dest`，默认 `./verl`，checkout 钉选 commit，`pip install -e.`）；`--option NAME` 选多变体 recipe 的某个变体（如 dapo 的 reproduction、flowrl 的 A、spin 的 baseline）。
- 安全细节：只用 `awk` 提取字段，**不 `source` / 不 `eval`** `REQUIRED_VERL.txt`，执行前回显完整命令。只依赖 bash、git、awk、pip。
- 注意：`install_verl.sh` 是 2026-10-13 的 #161 里刚被大改过的（+72/-2，Intel GPU 支持），读时以新版为准。

### 6.3 复现的典型命令形态

```bash
# 1. 装对应 verl
./install_verl.sh --recipe char_count
# 2. 准备数据（各 recipe 不同）
python3 char_count/create_dataset.py
# 3. SFT（如有）→ 合并 checkpoint
BACKEND=fsdp bash char_count/train_sft.sh
python3 -m verl.model_merger merge --backend fsdp --local_dir <CKPT> --target_dir <CKPT>/huggingface/
# 4. RL
bash char_count/train_grpo.sh
```

坑：多数 `run_*.sh` 写死了 `$HOME/...` 路径、GPU 数、Ray 地址（`RAY_ADDRESS`）、数据集路径，**直接跑大概率先报路径错**；先通读脚本再改路径。NPU / Ascend 脚本需要对应硬件与镜像。部分脚本假设在 Ray 集群上跑（dapo README 明确写“on the Ray cluster”）。

## 7. 测试 / CI

- GitHub workflows 只有 2 个：
- `.github/workflows/pre-commit.yml`：每周日定时 + PR + push 到 main / v0.* 时跑，Python 3.12，跑 `pre-commit/action`（即 ruff + ruff-format，`--all-files`）。CI 里会把 `.pre-commit-config.yaml` 的 `--output-format=full` sed 成 `github` 格式。
- `.github/workflows/verl-tinker-tests.yml`：只在 `verl_tinker/**` 变化时跑，`uv sync --group test --locked` 后 `pytest tests`，60 分钟超时。
- 本地 lint：`pip install pre-commit && pre-commit install`，`pre-commit run`（staged）/ `pre-commit run --all-files`（全仓）。只强制 ruff，不强制其他检查。
- `ruff` 配置在根目录 `pyproject.toml` 的 `[tool.ruff]`（line-length 120，select E/F/UP/B/I/G，排除 2 个文件）。注意该 `pyproject.toml` 本身是 verl core 的遗留文件（`[project] name = "verl"`，见第 11 节），ruff 配置是借放在这里。
- 测试覆盖：全仓只有 5 个 tests 目录（`async_flow/tests`、`atropos/tests`、`dynamo/tests`（+ `smoke`）、`verl_tinker/tests`）。**绝大多数 recipe 没有测试**，验证靠 README 声称的复现指标 + 手动跑。这意味着 doc 类、配置类 PR 很难被 CI 拦，review 靠人工。
- PR #160 的作者自述：只做了 helper 抽取 + `py_compile`，完整 Megatron 测试套件需 GPU 未跑。PR #159 的作者用 stubbed `megatron.core` 导入加载模块做验证。两者都说明：这个仓的 GPU 依赖代码在 CI 里基本不测。

## 8. Open issue inventory（2026-10-14 实测）

上游 `open_issues_count = 57`，实际拆分：**20 个 issue + 37 个 open PR**。下面 20 个 issue 按编号倒序，关联 PR 只写有正文证据的（body 含 Fixes/Related to 或标题直接对应）。

### 8.1 Open issues（20）

| # | 标题 | 作者 | 创建 | 评论 | 关联 PR / 状态备注 |
| --- | --- | --- | --- | --- | --- |
| #151 | [gkd recipe] jsd mode 静默 clamp beta 到 (1e-6, 1-1e-6)，beta 1.0 得 ~0 loss 且无警告 | XuMeng00124 | 2026-09-09 | 0 | #159、#160 均称 Fixes #151，均 open（见第 10 节） |
| #145 | [RFC] Environment-Regularized Policy Optimization（ERPO）支持 | adtureven | 2026-08-31 | 0 | 无关联 PR |
| #143 | [RFC] HACPO：verl V1 上异构 agent 协同训练 | JacckMa | 2026-08-24 | 0 | #148 称 Related to #143，open |
| #139 | [BUG] langgraph_agent 在 chat template 返回 BatchEncoding 时 crash | le-czs | 2026-08-20 | 0 | #140 称 Fixes #139，open |
| #133 | [RFC] TMax 终端训练 recipe | kylemontgomery1 | 2026-08-11 | 0 | 实现已由 #141 合并（`tmax/` 在 main 中），issue 本身仍 open |
| #132 | [RFC] ECHO 终端训练 recipe | kylemontgomery1 | 2026-08-11 | 0 | 实现已由 #142 合并（`echo/` 在 main 中），issue 本身仍 open |
| #130 | [RFC] Enc-LLM 解耦多模态 RL 训练 | nekoteai | 2026-08-05 | 0 | 无关联 PR |
| #129 | [RFC] BiDiRL：全异步 RL 双向资源调度 | nekoteai | 2026-08-04 | 0 | 无关联 PR |
| #122 | QAT recipe of Qwen3-8B-Base（问何时发布，README 有结果但无对应 recipe） | aaab8b | 2026-07-22 | 1 | 无关联 PR |
| #108 | recipe 引用已失效的 `verl.utils.rollout_skip`，在 verl main 上启动失败 | MouseSheep99 | 2026-06-07 | 0 | 无关联 PR |
| #74 | swerex-python:3.11 镜像在 Docker Hub 上找不到 | dialectics67 | 2026-03-28 | 0 | 无关联 PR |
| #46 | DAPO 训练进度条问题（过滤后样本不足 continue，进度条不同步） | kkkeepgoing | 2026-02-24 | 0 | 无关联 PR（中文标题） |
| #39 | [BUG] run_qwen2_7b_dapo 执行失败（Ray 任务失败但训练继续） | noob000007 | 2026-02-08 | 2 | 无关联 PR |
| #29 | TypeError: cannot pickle '_contextvars.ContextVar' | noob000007 | 2026-01-30 | 2 | 无关联 PR |
| #28 | 解释器返回图片能否作为结果 | noob000007 | 2026-01-29 | 0 | 无关联 PR（中英双语） |
| #26 | fault_manager=True 时性能比 False 低 1.7% | hxw1234567 | 2026-01-27 | 2 | 无关联 PR |
| #25 | gbs 太大时挂起，只打印 finished_num: 0 | hxw1234567 | 2026-01-27 | 1 | 指向旧 `verl/recipe/fault_recover/` 路径 |
| #19 | `dataframe.map(self.map_fn2, num_proc=16)` 加载失败 | tcxia | 2026-01-19 | 1 | 无关联 PR（中文标题） |
| #10 | 'MathExpressionReactAgentLoop' object has no attribute 'tools' | LunarCM | 2026-01-07 | 4 | 无关联 PR |
| #5 | 请教 langgraph agent 相关问题（能否 RL 训练 deep research 型 agent） | glennccc | 2025-12-01 | 0 | 无关联 PR（中文标题，问题贴） |

观察：20 个 issue 里 6 个是 RFC（#145、#143、#133、#132、#130、#129），其中 #133/#132 的实现已合并但 issue 未关闭；其余多为 bug 报告与提问，0 评论的占多数，issue 区基本无人维护性回复。**本节只做事实整理，不做“值得做”评价。**

### 8.2 Open PRs（37，标题 + 作者 + 草稿态）

| # | 标题（缩写） | 作者 | 草稿 |
| --- | --- | --- | --- |
| #162 | dynamo：把 Claude Code 子 agent 调度为 ThunderAgent 程序 | liz-badada | 是 |
| #160 | jsd distill loss 端点 beta 改警告而非静默坍缩（Fixes #151） | jasonchen505 | 否 |
| #159 | [gkd] jsd distill loss 配置 degenerate beta 时警告 | lozlrc | 否 |
| #158 | ObsSpec 联合训练 + 复现脚本 | kylemontgomery1 | 是 |
| #157 | POISE recipe（NeurIPS 2026） | choiyunho1129 | 否 |
| #152 | add drkernel | wucong25 | 是 |
| #149 | [dynamo] FlexKV L2 CPU KV-cache 支持 | hou2lin | 否 |
| #148 | HACPO 异构策略协同训练（Related to #143） | JacckMa | 否 |
| #140 | 归一化 LangGraph chat template token ID（Fixes #139） | le-czs | 否 |
| #138 | nemo_gym/browser：NeMo Gym interactive_browser 上的 step-wise GRPO | waple0820 | 是 |
| #134 | Ascend INT8 W8A8 QAT rollout | sunsunsun98 | 否 |
| #127 | Qwen3 MoE MXFP8 端到端 recipe | jQizhang | 是 |
| #125 | gsm8k_outcome_gating：shaped reward 下 group-filter 度量安全性 | yuyu0529nya | 否 |
| #121 | tau2_airline：τ²-bench 多轮工具 agent RL | yuyu0529nya | 否 |
| #119 | W4A8 FSDP QAT 仿真 recipe | zhangyimi | 是 |
| #107 | 上下文管理 agent loop（自 verl#5636 搬运） | HarishKMurali | 否 |
| #106 | retool：Qwen3.5 NPU（Ascend）RL 训练脚本 | luoshijiang | 否 |
| #105 | DMPO 算法 | OliverLeeXZ | 否 |
| #102 | gkd_ascend recipe（NPU + FSDP） | vvaen | 否 |
| #100 | Dapo predictor | ml1019-lmx | 否 |
| #99 | VerlBB recipe | ISEEKYAN | 是 |
| #97 | ASFT recipe（Anchored SFT） | zhuchichi56 | 否 |
| #95 | randopt 算法 | sunrainyg | 否 |
| #91 | SWE-Agent recipe（适配新 black-box Agent Framework） | wangtiance | 是 |
| #89 | qwen3-vl-8b NPU recipe + DAPO | yidingdou | 否 |
| #84 | mindspeed-mm 后端采用自定义 trainer 的梯度裁剪 | Zhang1Sheng | 否 |
| #83 | agentic training example | KunWuLuan | 否 |
| #82 | 新 dapo trainer（TransferQueue） | ji-huazhong | 是 |
| #78 | MBPP 代码生成 recipe（GRPO） | dhruvnigam93 | 否 |
| #77 | Online Policy Distillation（OPD）recipe | narutolhy | 否 |
| #69 | retool：工具调用本地线程池执行模式 | xiaohong42 | 否 |
| #68 | atropos reward registry（可插拔奖励桥） | 1HazyOne707 | 否 |
| #63 | HumanLM 集成（用户模拟） | ehejin | 否 |
| #58 | 异步 partial rollout trainer（样本补充 + 缓存） | mamazi0131 | 否 |
| #51 | [gkd/fsdp] TCAD recipe（Ray + FSDP 分布式训练） | aslyc | 否 |
| #48 | test coauther pr | Kirrito-k423 | 否 |
| #22 | Atropos GRPO recipe 实现（Issue #1782，指 verl 主仓 issue） | vyomakesh0728 | 否 |

观察：open PR 积压严重，最早的 #22 来自 2026-01-22，多数新增 recipe 类 PR 长期未合并；#48 标题为 `test coauther pr`，疑似测试性 PR。本节只做事实整理。

## 9. 与 PR #160 的关系

- PR #160（`Warn on endpoint beta in jsd distill loss instead of silently collapsing`，Fixes #151，2026-10-07 提交，open）即 fork 的 `contrib/151-jsd-beta-warning` 分支，head `e286256`。
- #160 的改动（PR 正文 + files API 实测）：仅 `gkd/megatron/megatron_distill_losses.py` 一个文件，+41/-2。新增 once-per-process 警告 helper `_warn_jsd_endpoint_beta()`，在 `jsd` 模式且 `beta >= 0.99` 或 `beta <= 0.01` 时发 `UserWarning`，指引用户用 `kl` / `rkl` / `kl_rkl`；调用点两个：`_VocabParallelJSDivergence.forward`（函数式 API）与 `VocabParallelDistillLoss.__init__`（配置驱动路径）；`VocabParallelDistillLoss` docstring 记录端点坍缩行为；**loss 公式本身未改**。
- 竞争 PR #159（lozlrc，同日提交，open，head `7247c84`）：同样 Fixes #151，改 `megatron_distill_losses.py`（+17/-2）+ `gkd/megatron/README.md`（+3/-0）。实现：`VocabParallelDistillLoss` 构造时警告一次，`beta` 在 `(0.01, 0.99)` 之外即警告；jsd 别名收进 `_JSD_NAMES`；README 记录该行为，并注明 `distill_loss` 不在默认配置中、需用 `+` 前缀设置（以 `run_moonlight_dsv3_training.sh` 为例）；正文特别说明警告不声称训练停止学习（Adam 更新量基本与 loss 尺度无关，引 verl-project/verl#7975）。
- 两个 PR 的关系事实：
- 同一 issue、同日、同一文件、同一修复选项（issue 里的选项 (a)：警告）。
- #159 正文称“无 open PR 触及该文件或引用 #151”，该说法与 #160 的存在不符（以 2026-10-14 实测为准；两 PR 提交先后顺序未考证）。
- #160 的 PR diff 文件干净（仅 1 文件），但** commit 历史含 fork main 的个人笔记 commit**（`97c1896`、`d570554`、merge `5d301cb`）以及一次删文件又重加的中间过程（`5fa788d` 删除 `megatron_distill_losses.py` → `2e92a44` 重写实现）。合并时若用 squash 则无影响，若用 merge commit 则历史会带进去；以 maintainer 实际操作为准。
- #160 的 head 未跟进上游 #161（见第 11 节），目前 base 为上游 2026-09-22 的 `242c73f`，#161 改的 char_count / open_math_reasoning 文件与 #160 无重叠，冲突概率低（未实际 rebase 验证）。

## 10. Fork 与上游的 drift（2026-10-14 实测）

- fork `main`：`5d301cb`（2026-10-06），是一个 merge commit，父母为个人链 `d570554`（← `97c1896`）与上游 `242c73f`（2026-09-22）。
- 上游 `main`：`81d7827`（2026-10-13，PR #161，Intel GPU 支持）。compare 实测：fork 相对上游 ahead 3 / behind 1。
- behind 1：仅 #161 这一个上游 commit 未进 fork（改 10 个文件：README、char_count×4、open_math_reasoning×4、`install_verl.sh`）。
- ahead 3：2 个个人笔记 commit + 1 个 merge commit，均不在上游。
- fork `main` 含 4 个个人笔记文件（**绝不进上游 PR diff**）：`INCREMENTAL_LEARNING.md`、`INTERVIEW_5_CATEGORIES.md`、`INTERVIEW_PREP.md`、`REPRODUCTION_PLAN.md`。`contrib/151-jsd-beta-warning` 分支已删除这 4 个文件，所以 #160 的 diff 是干净的。
- fork 分支：仅 `main` 与 `contrib/151-jsd-beta-warning`。本次新增 `notes/learning` 分支（从 fork `main` 5d301cb 建，不含 contrib 分支的改动）。
- 上游另有两个非 main 分支：`wuxibin/langgraph_agent`、`wuxibin/license`（内容未读）。
- 上游最近 commit 节奏：2026-09-11~10-13 约每月数个合并（#141/#142、#150、#153、#154、#156、#161），PR 积压 37 个，合并速度远慢于提交速度。

## 11. 仓库卫生观察（事实，不评价）

- 根目录 `pyproject.toml`：`[project] name = "verl"`，是 verl core 的 pyproject，2025-11-25 随首个 retool commit 进来，之后未动过。ruff 配置借放在此文件里。是否被任何流程用作本仓包定义，未验证。
- 根目录 `change.md`（约 10KB，中文）：内容是关于 triton-ascend kernel-verifier skill 的备忘，与 verl recipe 无关，随 2026-05-07 的 #59（FSDP backbone recipe）合入。疑似误带文件。
- README 表格里列了 `fault_recover`、`flash_rl_ascend`、`r1_ascend`、`swe_agent` 等目录的 `REQUIRED_VERL.txt` 链接，但这些目录在当前 main 树中**不存在**（顶层目录列表无此四项）。README 与树不一致，属陈旧链接。
- `nemo_gym/` 全目录无 README；`dance_grpo/`、`gkd/`、`specRL/` 顶层无 README（README 在子目录）。
- `minicpmo`、`prime` 的 README 只有骨架，内容近乎空。

## 12. 学习中发现的疑问 / 待验证点（明确标不确定）

1. 各 recipe README 声称的复现指标（如 dapo AIME 52%、retool GRPO 0.6）未独立验证；部分 README 可能滞后于代码。
2. `retool/README.md` 正文写 `verl==0.6.1`，与 `retool/REQUIRED_VERL.txt` 是否一致未逐行核对。
3. issue #108（stale import）是否已被后续 commit 修掉：issue 仍 open，但未逐个检查各 recipe 的 import 现状。
4. #159 与 #160 哪个会被 maintainer 接受；#159 多改 README 且明确处理了 Adam 尺度论点，#160 的调用点覆盖更全（forward + __init__）。纯事实对比，不做胜负判断。
5. `tmax` 的 `REQUIRED_VERL.txt` 钉到个人 fork commit（`Mengyuyang/verl@f765e4f439`，来自 verl PR #7458），该 fork commit 是否仍可拉取、PR #7458 是否已合入 verl main，未验证。
6. 根目录 `pyproject.toml` 是否被任何 CI / 安装流程使用；`change.md` 是否有意保留。
7. 37 个 open PR 中除 #160 / #159 / #148 / #140 外，是否还有其他 PR 与 open issue 对应（只对标题或正文明显相关的做了关联检查，其余未逐个读 body）。
8. `dance_grpo`、`nemo_gym`、`specRL` 的子目录结构与入口未细读（本轮只读了顶层与 README 层）。
9. `REQUIRED_VERL.txt` 的 33 个文件是否覆盖全部 35 个顶层目录（数量对不上，有 recipe 缺该文件或共享，具体缺失项未逐个列出）。
10. 未验证 `install_verl.sh --list` 在本地的实际输出；脚本只做了静态阅读。

## 13. 贡献姿势（基于实测的流程事实）

- 无 CONTRIBUTING / PR 模板 / issue 模板，PR 正文自由格式；现有 PR 正文从一句话到长篇 RFC 都有。
- 唯一硬性 CI 门槛：ruff（pre-commit）。提交前本地跑 `pre-commit run --all-files`。
- 新增 recipe 类 PR 占 open PR 的多数，积压久；小修（bug fix、doc、警告类）文件数少，更容易被 review。
- 每个新 recipe 必须带 `REQUIRED_VERL.txt`（README Contribution 一节明确要求），否则大概率被要求补。
- 历史教训（本仓实例）：个人笔记文件绝不能进贡献分支；贡献分支应从上游 main 建，而不是从含个人 commit 的 fork main 建（#160 的历史即反例，diff 虽干净但历史带了个人 commit）。
