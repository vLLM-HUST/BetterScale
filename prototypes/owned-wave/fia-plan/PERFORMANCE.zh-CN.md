# 原生 FIA 静态计划：小模型结果（2026-09-16）

**结论：本机 Qwen3-0.6B 的有界对照中，逐层 host attention 更新已消失，
缓存保留轮总耗时约下降40.9%；相对保留旧更新的 owned 路径约下降34.3%。**
这不是30B、TP/DP/EP或生产HTTP吞吐的结论。

## 条件与口径

- CANN9.0.1、910B2、同一张本机device4；每次独占子集 admission，串行测量，
  无profile的性能轮与独立profile分开。没有修改已安装runtime。
- Qwen3-0.6B真实BF16权重，28层，Q16/KV8、head128，TP1/DP1，dense无EP。
- 四条已固定SWE轨迹，各取前两次完整调用；不截断prompt、不改变token IDs或
  原输出预算，共8调用/951输出token。输入是token级回放，不是SWE解题质量测试。
- 最大长度12288、prefill chunk256、12GiB KV、四resident、原生前缀缓存。
  两个bank，18张prefill图+2张decode图，真实LiveModule State/shadow/N+2。
- 原生计时到client engine output；owned计时到worker receipt retirement。
  旧owned与新owned的调度/边界相同，缩小本次attention接缝收益的归因范围。
- 每个进程一冷两热轮。native-first与owned-first两种顺序复测。
  activation/capture不计入表中：新路径约85–105秒，不是零启动成本。

## 无profile结果（秒）

| capsule | 轮次 | 原生 | owned静态FIA | owned旧host更新 |
|---|---:|---:|---:|---:|
| fia-small1，native-first | 冷 | 3.3063 | 2.5795 | — |
| 同上 | 热1 | 2.9662 | 1.7567 | — |
| 同上 | 热2 | 2.9277 | 1.7564 | — |
| fia-small2，owned-first | 冷 | 3.3175 | 2.5848 | — |
| 同上 | 热1 | 3.0193 | 1.7622 | — |
| 同上 | 热2 | 2.9955 | 1.7620 | — |
| fia-small-legacy1 | 冷 | 3.3595 | — | 3.6041 |
| 同上 | 热1 | 2.9940 | — | 2.7115 |
| 同上 | 热2 | 2.9507 | — | 2.6462 |

新路径两种顺序的四个热轮：原生均值2.9772s，新路径1.7593s，耗时下降40.9%，
输出吞吐319.4→540.5 token/s（约1.69倍）。旧owned热轮均值2.6789s，
对比新路径耗时下降34.3%。这来自不同进程的小样本控制，不是完整统计置信区间。

热轮两边APC命中都为42368；owned热轮固定311 waves。冷轮原生命中28928、
owned17792，不能把冷轮差异全归因于attention。旧/新owned冷轮都是412 waves，
命中17792。完整机器可读结果：`runs/owned-wave/fia-small-summary.json`。

不主张全模型token等价：每轮native/static完整输出相同1/8，legacy/static2/8。
按Fletcher已确定的口径，保留差异，不将cross-arm token一致性作为性能门槛。
原生FIA叶子对照的max-error0仍是独立的算子证据，不能替代整模型质量评估。

## 独立msprof / TraceLoom验证

`runs/owned-wave/fia-small-profile1`，每边64waves，原始provider DB经官方offline
解析，再由固定TraceLoom37323af分析/导出；两种DB的API计数交叉验证一致：

| 项目 | 原生 | 新owned |
|---|---:|---:|
| ACL graph replay | 64 | 64 |
| 图内FIA device任务 | 1792 | 1792 |
| host FIA execute | 1792 | **0** |
| host FIA GetWorkspaceSize | 1792 | **0** |
| task update begin/end（各） | 1792 | **0** |
| 命名FIA Tiling事件 | 64 | **0** |
| 图内sampler ArgMax | 0（图外63） | 64 |

新路径所有FIA与ArgMax都归属graph model23/24，各32次replay。
注意：原生图内FIA本来就存在；本次消除的是host逐层重绑定/发布，不是第一次
把attention矩阵计算放进图。两边profile窗口prompt/batch工作不同，只用来验证
调用边界，不比较kernel总时长或把事件持续时间相加当可节省墙钟时间。

审计：`traceloom/static-fia-audit.json`；导出：
`traceloom/native-rank0.perfetto.json.gz`、`traceloom/owned-rank0.perfetto.json.gz`。
执行 `serving/audit_static_fia.py CAPSULE --layers 28` 可复查计数与图成员条件。

## 适用边界

当前是固定CANN二进制/布局的研究原型，以LD_PRELOAD进行启动期计划捕获，不是
已发布的通用vLLM插件。零runner执行、零replay阶段Python attention launch、
zero-live-invocation/lease退休检查均通过；20模板/560层bank绑定，1120次Python
launch仅在warmup/capture发生。12GiB共享arena快照与同进程原生/owned资源共存，
因此峰值内存不能解释为独立部署容量。尚未把这次薄包装重新资格化到30B TP/DP/EP。
