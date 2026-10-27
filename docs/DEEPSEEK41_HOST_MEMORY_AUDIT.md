# DeepSeek-V4.1-Flash-1017 主机内存审计（7ai）

**40 GB 是专家缓存上限，不是整个引擎的内存上限。** 在当前配置下，
缓存之外可明确计算的容量约 **2.27 GB**，其中 Engram 按满容量计算，
另有尚未量化的开销。现有证据不能为 40 GB 缓存给出保守的零 swap 保证。
建议下一步单独测量 **36/40 GB** 的内存余量与缓存代价；36 GB 尚未实测，
不改变默认配置，也不把它当成原来 swap 故障的修复。

本步只读固定源码、pack 元数据和已有日志，新增模型进程和请求均为 **0**。
原来的 7af 故障保持未解决。以下明确区分源码推导、容量假设和实际采样。

## 范围与单位

- 固定模型为完整 IQ1_M/Engram 包，DSpark none、vision off、context1024。
- 固定原生引擎为 7ae 的 CUDA12.8/SM75 版本，SHA256
  `d6d9e7cdafbd3591faa61df7915b0f9bdd2f30aee4eeacb3befabb469a1b1553`。
- 20 GiB GPU 目标，40 GB LRU 专家缓存，18 CPU workers，chunk256。
  heat1；7af 包含 preload on/off，7ag/7ah 为 on、GPU settle1000。
- GB/MB 为十进制，GiB/MiB 为二进制。原生日志的部分 `MB` 实际按 MiB 配置，
  例如 Engram 的 `256 MB` 和 staging ring 的 `400 MB`。
- 7ai 当前读取的 `MemTotal` 为 **67,316,617,216 字节**，即 67.32 GB /
  62.69 GiB。这是用户所称“64G”机器的当前内核可用总量；它不是旧运行时的
  cgroup 配额或压力快照。预算使用旧运行起始的 `MemAvailable`，不直接拿
  标称 64 与缓存数字相减。

所有静态字节数均为该源码/元数据对应的请求容量推导，没有执行新的原生
分配测试。CUDA、分配器和内核实际记账可能不同。

## 分配所有者账本

| 所有者 | 推导的容量 | 类型与限制 |
| --- | ---: | --- |
| ExpertStore 最大可分配槽 | 39,994,880,000 B | 私有 pageable arena；不是实测驻留字节 |
| BF16 embedding 表 | 1,323,827,200 B | 私有 pageable；129280 ×5120 ×2 |
| Engram 行缓存满容量 payload | 269,254,656 B | 私有 pageable；另有哈希表、keys、ref 等元数据 |
| flow 主机 staging | 474,218,560 B | 锁页请求；单 GPU，含 allocator wrapper 的64字节 |
| GPU 专家换入换出缓冲 | 26,063,040 B | 默认1 worker、3份最大 blob，锁页请求 |
| chunk 双份 handoff | 47,202,432 B | 锁页映射请求；单 GPU 也会分配 |
| CPU lane 工作向量 | 103,465,984 B | 私有 float 向量元素；不含分配器额外容量 |
| 其余已计 embedding/Engram 行缓冲、window handoff、bounds 等 | 21,287,752 B | 锁页请求；明细见审计 JSON |
| **缓存以外已计容量合计** | **2,265,319,624 B** | **不完整容量小计；不是同时刻 RSS** |

源码依据位于固定的 `build-deepseek-cuda/step7ae-upstream/src/dsv41/`：
`ds4_model.cu` 的 `halloc/hmap`（1332起）、embedding/host rows（2413起）、
CPU work vectors（2650起）、flow staging（2680起）、swap buffers（2778起）、
`alloc_chunks`（2844起）；`ds4_engram_io.hpp` 的 `EgRowCache`（40起）。
输入源码与元数据的 SHA256 固定在 `step7ai-memory-audit.json`。

CPU 工作区按 `kPicks=8` 的容量分配，实际路由 top6；不能把所有工作区容量
直接乘实际选中的6个专家。这个区别来自 `ds4_kernels.cuh:19` 和
`ds4_model.cu` 的 `kSlots`，没有修改路由。

### 专家缓存的三个不同数字

40层混合量化专家中，最大 blob 为 **8,687,616 B**。每个角色按256对齐并留
8192字节间隙；store 再加入首尾余量并按4096对齐，得到固定槽
**8,704,000 B**。源码见 `ds4_model.cu:1677`、`ds4_store.hpp:79`。

1. 最大槽数：`floor(40,000,000,000 /8,704,000) =4595`。
2. 最大可分配容量：`4595 ×8,704,000 =39,994,880,000 B`，约37.25 GiB。
3. 虚拟地址预留按64槽取整：`4608 ×8,704,000 =40,108,032,000 B`。
   最后多预留的地址不是额外已使用 RAM；最终分配仍受4595槽限制。

每个增长块为64槽，即 **557,056,000 B**。Linux 的 `range_reserve` 使用
`PROT_NONE/MAP_NORESERVE`，`range_commit` 调用 `mprotect`，没有主动触碰
整块的代码。见 `ds4_os.hpp:425`、`ds4_store.hpp:502`。
因此 STORE 的 `slots=4595` 只证明分配容量到顶，不能据此断定4595个专家
全部有效、所有页已驻留或这些页就是被 swap 的页。不同层共用最大尺寸槽，
内部余量也不能简单当成实测物理浪费。

当前环境没有启用 `DS4_STORE_ZC`，所以 store arena 没有走注册为 GPU
zero-copy 的路径；`DS4_FLOW_STORE=1` 是把已完成的 flow 读结果保留在
这个 store 中。`Slot::pin` 是缓存使用期间的内部所有权计数，与 OS 锁页
不同。不能由日志“CPU lane's experts pinned in0.0s”推断整个专家文件
已被锁页。见 `init_store()` 和 host_maps 注册循环。

### staging、Engram 和重复计数

flow 的48个 GPU ring 槽共 **417,005,568 B**。它已在 GPU 容量观测之内，
不能再次加到主机账本。主机 staging 每槽9,879,552 B，因为 gate/up/down
各自使用跨层最大角色尺寸并保留 direct-read 对齐余量；这比单个最大专家
blob 更大。日志的“400 MB ring”不代表主机只分配400 MB。

Engram 按264字节分配一个缓存行，容量1,016,800行；当前 IQ4_NL 的一行
文件 payload 为144字节。缓存按4096行一块增长，满容量需要249块，
因此 payload 分配为269,254,656 B。哈希表、keys/ref、分配器等仍另计。
这里列的是满容量场景，旧日志没有给出这些时刻的全部行数或驻留字节。

五份量化文件约231.32 GB、其中 Engram 文件约110.60 GB，都是存储大小。
`Pack::map()` 对文件做只读映射，不能把全文件大小直接加为私有 RSS。
被访问的文件页和 OS page cache 还可能指向同一批物理页，不能加两遍。
store 日志确认专家读为 unbuffered；Linux Engram reader 仍使用 buffered
读路径。dense 上传临时缓冲和源文件缓存也不能等同于一份常驻完整 dense 副本。

未完全计量的项目包括：store 槽/索引/热度表、tensor 元数据、Engram 索引、
分配器碎片、线程栈、队列、小型共享 KV/flag 缓冲、CUDA/驱动和页表开销、
加载转换瞬态、其他进程及 cgroup 压力。它们不能默认为零，也不能用
`MemTotal -峰值RSS -最低MemAvailable` 反推：这些数不是同一时刻或同一所有权。

DSpark、vision、多 GPU 缓冲及整份 CPU 专家副本不属于固定配置。
layer-major 的额外层缓存要求至少两个 prompt chunks；本组每题99/121 tokens
都只需一个256 chunk，因此未进入该路径。长上下文需要另行核算。

## 对齐六次旧会话的实际观察

这些记录跨步骤、请求顺序和 preload 设置；本审计只比较内存事实，
不据此计算吞吐提升或补齐7af缺失的性能样本。

| 会话 | 完整答案数 | 起始可用GB | 峰值RSS GB | 最低可用GB | 最大进程swap B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7af-preload-on-explain-code | 2 | 50.719 | 38.406 | 11.867 | 0 |
| 7af-preload-off-explain-code | 2 | 50.820 | 38.439 | 12.300 | 0 |
| 7af-preload-off-code-explain | 2 | 50.824 | 38.605 | 12.163 | 0 |
| 7af-preload-on-code-explain | 1 | 50.983 | 38.430 | 12.137 | 294912 |
| 7ag-diagnostic-on-code-explain | 2 | 52.568 | 38.821 | 13.693 | 0 |
| 7ah-diagnostic-on-code-explain-repeat | 4 | 52.714 | 39.983 | 12.626 | 0 |

共14题已启动、13题完整，最后失败的半题保持中断状态。峰值RSS和最低可用
内存一般不是同一时刻。审计 JSON 保留各自的完整原始 sample（时间、阶段、
RSS、HWM、swap、GPU和可用内存），而不是拼成不存在的一张快照。

原来7af的触发采样在282.628秒：进程swap **294,912 B =288 KiB**，
可用内存 **13,650,202,624 B**，RSS **38,429,683,712 B**。
该会话最低可用内存12,136,636,416 B出现在258.013秒，不能替换触发值。
触发的是进程swap分支；原来的错误和SIGTERM退出保持不变。

7ag/7ah的两次成功诊断没有触发故障快照。旧记录没有 RssAnon/RssFile、
VmLck/VmPin、smaps交换映射或同时刻cgroup压力。无法确定旧swap的是
embedding、store、线程栈还是其他页面，也没有证据把它归因于 preload、
主机OOM或内核策略。每秒/边界采样不能排除未采到的瞬态。

## 保护边界与预算

Python 启动前只检查主机 `MemAvailable >=8 GiB`；运行中同时检查该值、
引擎 `VmSwap ==0` 和 GPU20 GiB。启动前检查未核算完整的主机内存容量需求。原生 `ram_avail()` 还会参考各级cgroup-v2的
`memory.max/current/stat[file]`，与包装层只看主机可用内存的视图不同。
旧记录没有足够数据还原当时的cgroup余量或 `memory.high` 压力。

store 增长块会检查余量加8 GiB；flow retain 在拷贝前也检查余量。
这些检查不能把 Linux `mprotect` 的地址提交等同于一次物理内存预留。
`store_watch()` 每16个窗口、可用量低于4 GiB才尝试收缩尾部整块，目标
约5 GiB；遇到使用中、loading或GPU-visible槽会停止。
其旧“约0.7GB一块”的估算也不是这里557,056,000 B实际块大小的精确承诺。

因此不能把原生4 GiB回收视为包装层8 GiB停止线之前的保障。
六会话的**主机**采样均未到8 GiB，且没有收缩日志；原生cgroup视图和
采样时序仍可能不同。更不能用调整这两条阈值解释或修复旧288 KiB swap。
本步没有改变任何保护或OS设置。

为检查容量余量，使用一个明确的假设场景：

`起始MemAvailable -专家最大槽容量 -已计其他容量2.265319624GB -8GiB`

假设这些容量同时驻留，且起始外部负载不变；尚未计量的开销还需由剩余量
承担。该算式只是预算情景，既不预测实际RSS，也不为零swap提供保证。

| 缓存GB | 最大槽 | 专家容量GB | 加已计其他容量与8GiB所需GB | 六次起始余量情景GB |
| --- | ---: | ---: | ---: | ---: |
| 32 | 3676 | 31.995904 | 42.851158 | 7.868 至9.863 |
| 36 | 4136 | 35.999744 | 46.854998 | 3.864 至5.859 |
| 40 | 4595 | 39.994880 | 50.850134 | -0.131 至1.864 |

40GB场景在三个旧启动状态下余量为负或接近零。**负数不是旧运行已超额的
观测**：store未触碰余量页、缓存内容和Engram活跃行数等都会使驻留情况与
满容量场景不同。旧采样最低仍有11.87GB可用。这里说明的是：即便只计
已知容量，也不能把40GB当成有保守余量的通用配置。

36GB场景少459槽，容量减少 **3,995,136,000 B**。算式的额外余量约4GB，
不是已经测到释放4GB RSS，也不能排除同样的swap故障。32GB仅作为容量
参照，过去其他配置的32/40GB成绩不迁移为本配置的性能结论。

## 建议下一步：独立36/40 GB预算试验（7aj，尚未启动）

先在 benchmark 中窄范围允许原有四题诊断配置使用36GB。保留所有公共
默认、旧参数拒绝条件和原生二进制；先完成不启动GPU的参数/转发控制。

最多两个模型进程，固定顺序40GB、36GB；每个进程只跑原来的
代码→解释→代码重复→解释重复，共最多8题，每题fresh KV、cap256。
保持同一引擎/pack、heat1、RAM preload on、GPU settle1000、chunk256、
20GiB GPU、8GiB主机停止线、18 workers、context1024；唯一试验变量
为专家缓存上限。沿用故障时才采集的3秒元数据快照。

记录全会话和逐题时间、首token、decode、读量、miss/wait、分配容量、
内存与质量。任一资源或其他运行失败就停止该试验，保留部分结果，
不重跑、不用历史结果补位。每档只有一次会话、OS缓存不清除且有顺序效应，
只能做这一组的描述性观察，不能声称稳定性能差异。

这是一项主机余量/缓存代价试验，不替代7af的 preload on/off 对比，也
不关闭其故障。36GB不预先通过资源门槛，不承诺更快，不直接升为默认。
需要用户确认后才开始7aj。

## 验证与保留

本步通过31项离线证据检查，复核44个输入哈希、40层专家元数据、六次
旧会话和26份STORE快照。没有重新运行模型、单元测试、原生构建或质量
采样；旧的48项测试/131项控制和答案审阅作为历史证据保留。
只有本审计文档及路线图发生本步 canonical 变更；运行/算子代码保持不变。

可复现脚本 `build-deepseek-cuda/audit-step7ai.py`、
`check-step7ai-audit.py`，结果 `step7ai-memory-audit.json`、
`step7ai-audit-checks.json`，最终索引 `step7ai-validation.json`。
原始运行输入只读，7af的失败与7ag/7ah“未复现”状态均不被改写。
