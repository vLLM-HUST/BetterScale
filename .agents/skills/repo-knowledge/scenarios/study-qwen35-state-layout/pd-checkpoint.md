# PD 分离阶段记录点 — 2026-10-03

## 当前状态与阅读顺序

**暂停开发/联合压测，保留为 draft PR；不申请立即合并或生产发布。**
Fletcher 已关闭 hw81、hw86 两个容器。不要再使用旧地址、PID、监听端口或
运行中服务的假设；下面的硬件结果均为已完成的历史实验，不是当前在线服务。
代码分支 `codex/pd-incremental-online`；最后实验/分析代码记录点 `6e06fab`，
本次收尾只补文档导航，不改执行路径、不重跑 NPU。

先读本页，再按需要进入：
- [在线协议、逐次修复和实验](online-pd.md)。早期 Store/shared-memory/
  target-only/串行事务章节是历史路径，不能覆盖后续 rank-private/MTP 实现。
- [pinned 容量、LRU 和当前外部依赖](rank-pinned-capacity.md)。
- [D 效率、100K dummy 与供给估算](d-cluster-efficiency.md)。
- [已接受数值边界](pd-storage.md#accepted-numerical-envelope-and-non-strict-hccl-assessment)。
- [早期双机搭建及 donor 来源](dual-host-pd.md)，仅作恢复线索。

## 保存下来的实现

拓扑：一台 P 机器上 4 个 TP2 instance；一台 D 机器上 DP4/TP2/EP8，
Qwen3.5-35B-A3B BF16、MTP2，最大上下文 262144。
最后 D 配置为每 attention group C48/R56、每 rank 44GiB State；
P 为 C16/R20、24.25GiB State/rank。C 是执行并发，R 是 resident seats；
State 预算包含固定 recurrent/MTP 与 dense pool，不等于全部物理 HBM。

- 每 session 单活跃写者；新 session 按容量选择独立 P/D owner，后续两侧粘性。
  TP rank 各持对应分片的私有、NUMA-local pinned arena，没有节点共享 KV 大池。
- immutable FA 页版本、generation-private tail、GDN/conv 边界快照按增量传递；
  当前 MTP wire/恢复按已验证实现处理，不能把早期“忽略 MTP”的设想当成现状。
- Mooncake CPU TransferEngine 负责跨主机数据面，控制面只传小元数据。
  选中的 rank-private 路径不依赖 Store/disk/pageable cache 层。
- State load/store 与正常 D cadence 解耦，经异步 warm admission 接入，
  不要求全局 DP idle、不制造空计算轮推进传输；空闲 owner 仍正确参与 EP graph 波次。
- 独立 session 的 State 事务并发；local staging/device release 与 peer commit
  分离，均保留真实 DMA fence、page/seat lifetime、TP quorum。
  输出不等待最终备份提交；同 session 下一轮仍须等一致的 checkpoint 发布。
- P 计算不预占 D device seat；P→D 排队保存已提交的 host checkpoint，
  到 D admission 才获取 D 资源。control RPC 不在 admission 路径同步排空旧模型响应。
- startup arena 预留、host-byte admission、idle LRU、双副本退休及 cache-miss
  重新 prefill 路径已在限定预算下验证。失败/不确定 DMA 保持隔离，不把数值容忍
  扩展成数据损坏、错误命中或提前复用的许可。

### 代码入口（仓库相对路径）

| 职责 | 入口 |
| --- | --- |
| Core/worker 增量缓存与两阶段完成 | `src/betterscale/models/qwen35/cache_{actions,engine,pages,policy,worker}.py`、`state_dma.py` |
| 页状态与主机传输接口 | `src/betterscale/live/runtime/{host_state,page_state,page_transport}.py` |
| rank-private arena/对象池/收发/生命周期 | `prototypes/pd-kv-layout/rank_{pinned_arena,state_pool,state_transport,replica_receiver,replicator,peer_control,state_runtime}.py` |
| 会话路由、host admission、退休与输出 | `prototypes/pd-kv-layout/online_{coordinator,host_cache,actor,stream,frontend}.py` |
| 节点及隔离运行胶囊 | `prototypes/pd-kv-layout/online_node.py`、`run_pool_node.sh`、`stage_online_candidate.py` |
| D-only graph 与 C16→C48 接线 | `prototypes/pd-kv-layout/decode_graph_policy.py`、`src/betterscale/models/qwen35/execution_capacity.py` 及关联 metadata |
| 正确性/容量/profile | `online_probe.py`、`online_quality_probe.py`、`dummy_decode_{entry,probe}.py`、`analyze_dummy_timeline.py`（均在 prototype 目录） |

这仍是 opt-in 的研究集成，不是把 prototype launcher 等同于公开 Python 包支持。
仓库保留早期比较臂和失败复现，不在收尾时删历史、改写 129 个既有开发提交，
或混入 main 上另一条 shared-expert overlap 工作。恢复时再做有边界的整合与重验。

## 已验证的结果及不可混用的口径

| 观察 | 结果 | 边界 |
| --- | --- | --- |
| MTP v10 正确性 | 16 并发 4K cold/warm、TP 字节审计、精确 262144 context；另有 16-session code-retrieval oracle | 不是全任务质量或跨分段 greedy token 全等证明 |
| C48 v23/v24 供给 gate | 192×4K/2048 输出加 warm8，16 卡字节审计；D 各 rank 达 48 live rows | 重复 token 合成负载，不是自然 SWE goodput |
| v28 LRU gate | 128 session cold/warm，240 次 eviction，全部 continuation miss 后仍完成；退休后 16 pools/arenas 归零 | 小 P pool 强制压力，不是生产总容量验证 |
| v33 真实 SWE，2 session/s、240s | 2450.80 输出 tokens/s；2345 requests、零失败/漏发、2345 commits；TTFT P95 5.928s | 全窗口、16 卡联合实测；末 60s D yield 4044.35/s 是另一口径 |
| v34 真实 SWE，4/s、240s | 2687.30 tokens/s；3253 requests、零失败/漏发；TTFT P95 24.293s | 含 profiler 扰动；host admission P95 100.259s，不能当峰值 goodput |
| v5 D-only 100K dummy | 每组 28 条、共 112；State 占用 78.36%；32 轮无 profile 均值约 70.0ms | 零值合成 KV，非真实请求、不含 PD I/O/输出路径 |
| v6 同几何八卡 profile | 内部周期约 70.6ms；attention 36.01–36.22ms，matmul 9.04–9.31ms，HCCL 7.07–7.53ms，未覆盖 <1ms | HCCL 含等待；MTE2 比率高不等于已经证明 HBM 带宽饱和 |

dummy 按经验值 2.826 accepted tokens/request/cycle 推算约 4521 tokens/s
（565/D chip），**不是实测 goodput**。78.36% 是固定 resident/MTP 加活跃 dense
相对于所声明 State 预算的比例，不是 64GiB 物理 HBM 的比例。
Fletcher 已把后续周期规划预算放宽到 **100ms**：按同一接受率约 28.3 tokens/s/
request；70ms 约 40.4。未声称已完成 100ms SLO 下的生产容量资格验证。

## 为什么在这里停

两个“内存”问题必须分开：
1. **阻止继续联合压测的是 host 物理 pinned 容量和重启稳定性。**
   容器 1960GiB memory.max 不预留 NUMA-local 高阶物理页。native HOST_NUMA
   路径查询到 2MiB 粒度；v35 甚至无法重建曾成功的 12GiB P rank arena。
   高阶页短缺/碎片化有源码和实验支持，但未证明固定配额或驱动泄漏。
2. **dummy profile 说明设备内 attention 搬运是主要优化候选。**
   attention MTE2 约 91.7%，MAC 约 22.2%；不是 host arena 分配失败的原因，
   也没有 DRAM 流量计数证明额外流量可消除或 HBM 已满带宽。

4KiB private mmap+register 虽达到 8×128GiB，双向各约 5–6GB/s，明显低于
native pinned 约 25GB/s，不作为生产替代；没有暗加 pageable/shared cache 层。
最后有效小池为 P 每组各 rank 32/32/12/48GiB，D 32GiB/rank，远未满足
约 80% 系统 DRAM 的目标。部署需求见 [issue #10](https://github.com/vLLM-HUST/BetterScale/issues/10)。

## 恢复工作的最小顺序

1. 获取新硬件/管理员支持后，先验证 NUMA-local pinned 预留、双向 DMA、
   释放和重复冷启动；不能只看 MemAvailable/cgroup limit 就开联合压测。
2. 恢复同一 donor：vLLM `752a3a504485790a2e8491cacbb35c137339ad34`，
   vllm-ascend `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`。
   实验环境 CANN 9.1.0、torch 2.10.0+cpu、torch_npu 2.10.0.post4、driver 26.0.rc1；
   使用已验证的 CPU Mooncake 0.3.13.post1，避免系统 Ascend wheel 抢 import。
   这些是实验胶囊基线，不是修改公开 release pins 或默认安装兼容性承诺。
3. 从保留的构建/胶囊记录恢复，核对 qualified baseline 再运行 staging helper；
   它不是脱离原 baseline 的自包含安装器。旧服务地址和旧 SQLite 不能当成存活缓存。
4. 冷启动两侧新 TE incarnation，先 cold/warm、并发入场、长上下文、LRU/miss、
   exact transfer 与 TP completion gates，再测真实在线负载。TE endpoint 复用后的
   缓存失效/重连尚无生产方案；协调重启只是实验恢复手段。
5. 保留 100ms 新预算及真实输出/窗口定义，测供给、TTFT、队列、cadence 与 occupancy。
   若继续优化内核，再针对 attention traffic/reuse/tiling 采集证据；不靠丢审计契约提速。

仍未完成：生产规模 host 容量及长期压力、HA/跨进程恢复、TE incarnation/reconnect、
认证/TLS 等生产控制面、disk 回写、通用发布打包。最新 partial-startup 清理修复仅
通过 CPU 子进程/生命周期测试，尚无下一轮 native 故障路径重验。

## 保存的证据与本次收尾检查

Git 保存实现、CPU 回归和逐次实验记录；大模型/原始 profile 不塞进 PR。
本机 `BetterScale-migration-20261001` 备份目录保留关键证据：
`pd-v28-arena-lru-*-evidence.tgz`、`pd-v32-v33-evidence.tgz`、
`pd-v34-profile-evidence.tgz`、`pd-v35-*-evidence.tgz`、
`dummy-kv80-evidence.tgz`、`dummy-kv80-v6-profile-evidence.tgz`。
最后一个包含 8 个 native DB；独立下载的合并 timeline 在
`dummy-kv80-v6-profile/d8-mtp-dummy-timeline.json.gz`。
这些是已备份的有界证据，不承诺旧容器全部文件或全部历史 profile 都可恢复。

本次仅文档变更：校验知识树、文档链接和 Git diff；不把此前各批 CPU 通过数相加，
不宣称重跑了整个分支的全量回归或硬件资格验证。
