# DeepSeek-V4.1-Flash-1017 本地文本运行

适用于这个 worktree 已完成下载、校验和构建后的 Linux/CUDA 环境。入口复用已验证的 Strata 原生引擎和 tokenizer，支持单轮输入和常驻独立问答。

在 worktree 根目录启动常驻模式：

```sh
python3 tools/deepseek41/run_quantized.py --interactive
```

模型就绪后，每行输入一个独立问题。每题只使用当前问题作为上下文，已加载权重和专家缓存会复用。输入 `/exit` 或在等待输入时按 Ctrl+D 正常结束；Ctrl+C 会结束整个会话。常驻期间继续占用模型内存和显存，退出后释放。

空行会忽略；过长问题被拒绝后可以继续输入。因输出上限结束时提示“回答可能未完成”，随后可以继续提问。`--max-new-tokens` 对这个会话的每个问题生效。

需要一次性处理多行独立问题，也可把每题放在一行送入常驻模式的标准输入。读取和等待输入期间会持续检查资源及引擎状态，不计入每题的生成超时。

单轮命令继续支持：

```sh
python3 tools/deepseek41/run_quantized.py \
  --prompt '中国的首都是哪里？请用一句话回答。'
```

默认使用 `windows` 配置，也可以显式写 `--profile windows`。如需尝试已经过 7aa 对照的实验配置：

```sh
python3 tools/deepseek41/run_quantized.py --profile flow-experimental --interactive
```

单次问答和文件输入也可添加同一个参数：

```sh
python3 tools/deepseek41/run_quantized.py --profile flow-experimental \
  --prompt '中国的首都是哪里？请用一句话回答。'
```

`flow-experimental` 复用已测的原生引擎、chunk 256、流式专家缓存和 1 秒交换预算，RAM 预加载保持开启。1 秒只限制交换循环，不限制后续预加载或完整回答时间。原生路径仍由引擎自动选择。

7aa 在两个固定任务及重复请求中，四次问答合计耗时降低约 23.5%–27.6%；部分短问答反而慢约 6%–16%。每种顺序/配置仅两轮，不代表所有输入都会受益。实验配置需要本 worktree 已构建的候选二进制；入口会检查固定引擎、pack/tokenizer 与权重校验记录。它与默认配置共用下述输入、内存、输出和清理限制。

面向本机内存余量，还可显式选择 7al 新增的配置：

```sh
python3 tools/deepseek41/run_quantized.py --profile memory-experimental --interactive
```

`memory-experimental` 使用 **36 GB（十进制）RAM 缓存**，复用已经过 7aj/7ak 检查的原生引擎及 heat1、LRU、chunk 256、流式专家缓存、交换预算 1000 毫秒和开启 RAM 预加载的组合。它支持相同的 `--prompt`、`--prompt-file`、标准输入和 `--interactive`。每题重新建立 KV，常驻模式复用权重和专家缓存。

7aj/7ak 用同一候选引擎分别按 40→36 GB、36→40 GB 运行固定四题。36 GB 的进程峰值 RSS 低约 3.29/3.52 GB，系统最低可用内存高约 3.04/3.40 GB；四题合计耗时分别增加约 8.46%/5.47%。这是这两个小规模对照的内存与延迟取舍，不能用于预测它相对默认 `windows` 或旧 `flow-experimental` 的速度，也不保证所有任务或长期运行都安全。

新配置启用失败时的资源元数据采集：沿用原资源停止条件，先限时最多 3 秒尽力保存进程、主机和 cgroup 元数据，再继续终止及清理。正常运行不创建诊断目录。本轮未触发原生资源失败；此前 7af 的 swap 失败和未明根因继续保留。

7al 通过真实公开入口运行一次单题、一次常驻两题。18/99/121-token 输入分别产生 34/151/192 token（含 EOS），全部自然结束；说明题六条字数和代码 16 个边界样例通过。常驻模式拒绝超长输入后仍能继续问答；单题从 `/tmp` 用脚本绝对路径启动成功。本轮峰值 RSS 约 35.37 GB、最低系统可用内存约 17.41 GB，未观察到引擎 swap。这是三题功能烟测；长上下文、更多题目的质量和持续稳定性仍待验证。

7am 随后用同一个 `memory-experimental` 进程连续完成 8 题：说明、代码、仓库抽取和多步算术各两次。总计约 **14 分 56 秒**，全部自然结束，未观察到 swap；峰值 RSS **37.77 GB**、系统可用内存最低 **15.14 GB**。36 GB 限制的是专家缓存，整个进程仍有其他内存占用。本轮正常退出并释放模型进程和显存占位；这不是全天运行或无内存泄漏的证明。

质量检查中，内容通过 **6/8**，严格格式通过 **5/8**。两份代码共 **32/32** 个边界样例通过，仓库抽取与算术结果也都正确；两份说明把显存容量是否足够表述得过于绝对，一份说明有两条字数超限。算术只给算式，未满足事先固定的“补充中文计算说明”规则；题目原文只写“计算说明”，这个格式评分采用了更严格的预设规则，不是算术错误。首字等待仍为 **40–98 秒**，解码约 **1.46–3.30 token/s**；这些是八题观测值，不是速度对照或长期保证。完整范围见 [7am](DEEPSEEK41_ROADMAP.md#step-7am-bounded-eight-question-resident-stability-and-quality)。

入口功能实测覆盖 14/257/264-token 输入和 64/128-token 输出上限。[7ac 长输出检查](DEEPSEEK41_ROADMAP.md#step-7ac-longer-explanation-and-code-output-validation)另以 256-token 上限完成 8 次说明/代码问答，实际输出 139–192 token，均自然结束。四份代码的固定边界样例全部通过；四份说明只有一份完全满足逐条字数范围，另有一份主题解释不足。本轮长输出未复现 7aa 的加速表现，实验配置继续保留为可选，默认使用 windows。

也可以从其他目录用脚本的绝对路径启动。模型与构建资产按脚本位置查找。

读取 UTF-8 文件，最多生成 256 token：

```sh
python3 tools/deepseek41/run_quantized.py \
  --prompt-file question.txt --max-new-tokens 256
```

从标准输入读取多行问题：

```sh
python3 tools/deepseek41/run_quantized.py --prompt-file - <<'EOF'
请分三点说明内存缓存为什么可以减少读盘。
每点用一句话回答。
EOF
```

回答流式写入标准输出；加载状态、速度和结束提示写入标准错误。中文字符或 emoji 的 UTF-8 字节跨 token 时会合并后显示。原生 EOS 标记只保留在记录里。

三种配置共有的限制：

- 模型为已校验的完整 IQ1_M/Engram 基础权重，DSpark 和视觉关闭，使用非思考模式与贪心采样。
- RAM 缓存上限：`windows` 和 `flow-experimental` 为 **40 GB**，`memory-experimental` 为 **36 GB**，均为十进制。这是缓存上限，不是整个进程的 RSS 硬限制。
- 显存容量限制为 **20 GiB**；在当前物理 22 GiB 的卡上，单独占位 2 GiB。
- 启动前和运行采样时要求系统 `MemAvailable` 至少 **8 GiB**。采样发现低于该值、引擎使用 swap 或显存越界时停止本次推理；每秒采样不能保证捕获所有瞬时峰值。
- 上下文为 **1024 token**，包含模板、输入和预留输出。输出上限默认 128，可设为 **1–256**。单轮模式的过长输入在加载模型前拒绝；常驻模式拒绝本题并继续等待下一题。
- `--timeout` 默认 600 秒，分别用于模型启动和每次请求；等待输入的时间单独处理。

7l 的单轮短题实测：模型加载到就绪约 43.9 秒，请求提交到首个 token 约 21.8 秒；从命令启动到首次回答输出约 67.2 秒，包含预检与加载。每次单轮命令都会重新加载；常驻模式只在会话启动时加载一次。7k 的三个较长任务测得请求到首字约 49–127 秒；这些是不同任务的观测值。已验证的长度与质量范围见 [roadmap](DEEPSEEK41_ROADMAP.md#step-7k-bounded-longer-input-and-longer-output-checks)。

因输出上限结束时会提示“回答可能未完成”，记录中的 `finish_reason` 为 `length`；自然 EOS 结束为 `stop`。单轮命令在这两种情况下返回 0；常驻模式继续接受输入，正常结束会话时返回 0。参数、问题读取或空内容错误返回 2，模型预检或运行失败返回 1，Ctrl+C 取消返回 130。取消或超时会结束原生进程并释放显存占位。

默认记录写入 worktree 下的 `build-deepseek-cuda/text-runs/<UTC时间>/`。也可指定一个**尚不存在**的目录：

```sh
python3 tools/deepseek41/run_quantized.py \
  --prompt '用一句话解释显存的用途。' \
  --output-dir /tmp/deepseek41-example
```

目录内的 `result.json` 按问题保留输入、token IDs、`completion_text`、结束原因、耗时和内存观测；常驻模式还保留输入拒绝记录及会话状态。其他文件保存原生协议、事件和内存采样。已存在的目录不会被覆盖。入口沿用先前的完整权重 SHA256 校验记录，并检查当前文件大小、完成标记以及固定的引擎、pack 和 tokenizer 哈希。

常驻模式的实测范围与限制见 [7m 记录](DEEPSEEK41_ROADMAP.md#step-7m-resident-independent-question-entry)。目前验证以少量独立问题为主；一次说明题的某条为 23 个汉字，未满足要求的 25–35 字，严格格式遵循仍有波动。
