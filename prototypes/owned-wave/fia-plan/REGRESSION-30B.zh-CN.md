# 30B candidate：回退定位与静态计划缺口

**当前结论：host更新确实消除了，但30B TP2上的attention设备执行更慢。
不能把这版候选称为已经验证的30B提速版。** 不改变已发布的DSV4/PyPI Worker。

## Candidate-only完整轨迹计时

`runs/owned-wave/swe-static-candidate1`：真实48层Qwen3-30B-A3B，TP2/EP2，DP1，
同机device4/5、四会话、完整44调用/12,111输出token、6GiB KV/rank。
只跑candidate一冷两热；没有重新运行native baseline。
`compare_candidate.py`验证历史baseline的配置、完整trace与新candidate一致，
保留baseline来源，不冒充同时间A/B。

| 缓存 | 新candidate | 历史原生均值 | 相对历史原生 |
|---|---:|---:|---:|
| 冷 | 106.805s | 94.413s | 慢13.1% |
| 热1 | 95.839s | 87.561s | 慢9.45% |
| 热2 | 95.814s | 87.561s | 慢9.43% |

历史旧owned热轮均值89.809s，新candidate热轮95.827s，约慢6.7%。
热轮均3843waves/APC命中625664；冷轮新candidate3998waves/545408命中，
历史旧owned3996/545152，冷轮不是完全相同内部工作。
两rank协议/输出一致，24图、48次warmup/capture数值Python入口；每rank
24个bootstrap模板、1152个层/bank计划，2304次attention Python launch全部
发生在warmup/capture，replay阶段为零。activation59.10s单独记录。

## TraceLoom定位：四步，而非重跑baseline

第一次短窗口 `swe-static-short-profile1` 使用1K合成prompt，8步warmup后采4步；
验证host更新为零，但不能拿它解释4K+上下文的回退，因此未据此归因。

随后 `swe-static-matched-profile1` 使用四会话的**原始第一次完整prompt**，
只将输出预算限制为16以有界结束。40步必要prefill/预热后，仅采第40–43步。
此时输出预算还未耗尽，都是四请求decode。对照直接复用历史
`swe-profile1/traceloom/owned-rank{0,1}.db`的相同步号，不重跑旧candidate或native。
上下文约4.4–4.6K；生成token/MoE路由可能不同，不主张完全相同数值输入。

官方provider DB离线解析，再由固定TraceLoom37323af处理、导出并校验。
`analyze_matched_steps.py`按连续provider model-ID计算体定位：每体必须有
48个FIA和1个ArgMax，两bank交替。TraceLoom本身的精确Ascend replay partition
仍不支持，不能把该断言重新命名成已支持的exact partition。

以下为每步均值，再取两rank平均，单位ms：

| 部分 | 历史旧owned | 新candidate | 差值 |
|---|---:|---:|---:|
| attention设备kernel合计 | 2.8100 | 3.3852 | **+0.5752（+20.5%）** |
| dense matmul | 2.0553 | 2.0601 | +0.0048 |
| expert grouped matmul | 4.8724 | 4.9355 | +0.0631 |
| routing | 1.1781 | 1.1909 | +0.0129 |
| collective envelope并集 | 4.1931 | 4.1147 | -0.0784 |
| compute/collective未覆盖间隙 | 3.1790 | 2.9636 | -0.2154 |
| 整个设备计算体跨度 | 20.5231 | 20.8898 | +0.3667 |

两rank都显示同一主项：attention每层约58.5→70.5us。
图间间隙约0.091→0.096ms，不是主要新增成本。新profile每rank仅4次replay、
192个图内FIA/4个图内ArgMax，host FIA/GetWorkspaceSize/task-update/tiling均为0。
因此不是host更新回来了，也没有证据把主要回退归因给通信或dense GEMM。

kernel合计与collective envelope可能重叠，不能直接相加当墙钟；collective
包括等待，不是纯链路带宽。未覆盖间隙也不是所有设备引擎空闲的证明。
四步早期decode定位了慢项，但没有量化整条长轨迹全部6.7%回退的归因。

## 计划缺口：冻结了non-FD，也冻结掉了KV拆分

单算子**调用元数据诊断** `fia-plan-contrast1` 使用相同Q16/KV2、batch4、
head128/page128，不计时、不重新跑模型baseline。记录当前安装的原生FIA
真正传给runtime的function name、block数量和opaque plan：

| KV长度 | 原生kernel key | blocks | 计划内容 |
|---|---|---:|---|
| 启动[1,1,1,1] | 5000000000010200203 | 24 | non-FD，totalTaskNum=8 |
| 约4.5K | **5100000000010200203** | **23** | FD，含split/归并元数据 |
| 16K | **5100000000010200203** | **24** | FD，拆分元数据再次变化 |

安装包kernel源码将510…映射为FD分支。当前wrapper明确只接受500…，按
短上下文bootstrap后始终复用；non-FD循环按coreIdx分配totalTaskNum=8个任务，
名义launch24 blocks不等于24个核都有有效任务。四请求×每卡2KV heads仅8任务，
而FD会沿KV维拆分以利用更多核。这是实际原生dispatch与候选计划之间的明确缺口。

0.6B TP1每卡8KV heads，四请求已有32个任务，不能把其40.9%热轮收益外推到
30B TP2的8任务几何。

注意历史图的device任务名保留capture标签；旧augDB还含FD的COMPUTE_TASK_INFO
记录，但没有对应TASK行。因此不能只看旧device标签500…便断言整个动态更新
期间始终执行non-FD。这里分别保留了：图时间的观察、原生launch选择的实测、
以及丢失FD可能解释回退的机制；尚未做修复后消融证明全部因果份额。

正确后续方向是**保留FD，把其split/core/reduction元数据纳入wave级协议并让
各层共享**，不是恢复每层host task update，也不是直接拿短上下文FD表无限复用。
新的FD绑定、变化长度/非均衡请求和归并工作区仍须独立验证；本次没有宣称修复。

## 可复查产物

- `swe-static-candidate1/reused-baseline-comparison.json`
- `swe-static-matched-profile1/traceloom/matched-rank0.json`、`matched-rank1.json`
- 同目录 `static-fia-audit.json`、两份`owned-rank*.perfetto.json.gz`
- `fia-plan-contrast1/run/contrast.json`与三种原生launch参数快照

上述路径均在仓库的`runs/owned-wave/`下，大日志/设备快照不进入Git。
