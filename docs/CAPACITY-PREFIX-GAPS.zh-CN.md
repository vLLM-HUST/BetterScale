# 上下文、KV 容量与前缀复用缺口

2026-09-14 源码调查；BetterScale main `e584936`，vLLM `752a3a50`、
vLLM-Ascend `9bf964cb`。不是新硬件验收或新发布能力声明。

## 四个不能混用的数

| 项目 | 当前值 | 含义 |
|---|---|---|
| 模型配置长度 | 1,048,576 | hw3 实际模型 config.json 的 max_position_embeddings；不是本插件已验收的长度 |
| 服务 max_model_len | TP 15,104；DP 16,384 | 单请求输入与输出共同受限的序列长度；启动命令指定，插件 guard 又限制其上限 |
| 单波 token 预算 | TP 4,128；DP 每 rank 1,026 | 每步调度工作量，不是请求总上下文；长 prefill 可跨多个 chunk |
| 活跃请求席位 | TP 4；DP 每 rank 2，共16 | 当前发布准入范围，不是 KV 可容纳请求数或已测吞吐最优点 |

TP 的15,104来自此前32题 retrieval 验收输入所需范围
（prototypes/full-mixed/QUALITY.md），不是显存容量反算出的上限。
DP16,384同样是固定验收配置。把这些数变成发布硬限制保护了已测组合，
但也把实验范围误当成了产品能力边界；需要扩展，不能宣传为模型/硬件限制。

模型配置通过 hw3 只读获取：
`/data/shared/models/DeepSeek-V4-Flash-0731-w8a8/config.json`，
`num_hidden_layers=43`、`sliding_window=128`、YaRN factor16、
original_max_position_embeddings65536。最大长度配置本身不证明1M质量、
显存可容纳性或 graph 可执行性。

## 实际容量怎么算

给定请求长度 L，compressed MLA 的 C4/C128 每层页数按
`ceil(L / (physical_block_size * compress_ratio))` 估计；页字节数还包含
实际存储 dtype、head size、量化 scale 与 padding。C4 indexer 也要计入。

SWA 不是为 L 个历史 token 全部保留状态。原生上界使用
`ceil(min(window-1 + wave_token_budget, L) / block_size) + 1` 页。
因此 window128不等于永远只分128行：在大 prefill 波次期间仍需较大的活跃窗口。
Compressor state 也有自己的窗口、页大小和 dtype。

把这些组放进 Ascend 的共享池，还要计入 layer tuple 对齐/填充和 draft 状态。
池预算由显式 kv-cache-memory-bytes 或原生可用显存预算决定；不是把八卡剩余
显存直接加起来再除以一个全模型 bytes/token。TP 的复制状态不会因卡数自动
获得八倍长度；DP 则是各 engine 各自承载请求。

日志的 GPU KV cache size 是由当前 max_model_len 下的估计并发乘以该长度
得到的等效 token 数，不是单请求最大长度，也不能据此直接承诺最大活跃并发。
例如既有 run163 的 DP 每 engine8GiB KV，日志179,972等效 tokens、10.98x
16,384长度并发；该运行实际仍限制2个活跃席位/rank，并未验收11席位。

## 缺口清单

| 缺口 | 已确认事实 | 验收需要回答什么 |
|---|---|---|
| 发布长度过窄 | guard 固定15K/16K上限 | 更长长度下 block table、attention/QLI bounds、draft metadata、graph 内存与输出是否成立 |
| APC 未开放 | 命令与 guard 均关闭 | 命中后的 target、compressor、SWA、draft 状态是否正确续跑 |
| APC 粒度过粗 | pinned Ascend coordinator 按逻辑页LCM对齐；block128的 C128产生16K边界 | 减小物理页或支持部分页时，hash、状态检查点与共享页写入如何一致 |
| 当前范围内 APC 无有效命中空间 | KV manager至多查 prompt_length-1；当前上限不超过16K | 不能只开开关；必须同时扩大可用范围或细化命中粒度，并实测非零复用 |
| DP draft 未图化 | DP没安装TP split_draft，DSpark强制 use_cuda_graph=False | DP专属 metadata、EP协调、上下文写入与 query图的正确性/收益 |
| 容量与活跃席位未分开验收 | 当前只准入TP4、DP每rank2 | 实际分组占页、回收、峰值、驻留量与QoS下的活跃并发分别是多少 |

SWA specs、旧页回收和 admission cap 已存在，不能先把容量问题定性成
“没有实现滑动窗口”。尚待实际分组/池占用证据判断具体浪费在哪里。

原生通用 draft FULL graph 已存在，不是 BetterScale 的贡献。当前补丁增量
是 pinned DSpark TP 路径图执行和 context/query 拆分。enforce_eager=true
关闭原生 drafter 捕获，不代表安装TP补丁后实际还全部 eager；详见 split_draft README。

本次仅记录边界与缺口，未修改发布 guard、启动默认值或既有成绩。
