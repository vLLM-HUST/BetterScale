# BetterScale 原生安装、启动与回退

在已有的 Ascend 服务环境安装；不升级 donor/CANN/torch-npu，不设置动态库、HCCL、
端口或设备环境。唯一服务入口仍是 `betterscale.worker.Worker`。

```bash
python -m pip install --no-deps vllm-betterscale==0.4.1
```

固定兼容范围：vLLM0.25.1、vLLM-Ascend0.25.1rc1、torch-npu2.10.0.post2；相关
源文件必须匹配pins。模型为DSV4 Flash43层、4096 hidden、256 experts的Ascend W8A8版本。
以下替换模型路径即可；鉴权、TLS与外部访问控制仍由你的部署环境负责。

## TP8 / EP，四席位

每波预算4128、K5、DSACP；target和有限draft目录共享graph池，在READY之前准备好。

```bash
vllm serve /models/DeepSeek-V4-Flash \
  --worker-cls betterscale.worker.Worker \
  --tensor-parallel-size 8 --enable-expert-parallel \
  --quantization ascend --dtype bfloat16 --async-scheduling \
  --max-num-seqs 4 --max-num-batched-tokens 4128 --max-model-len 524288 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[24,4128],"max_cudagraph_capture_size":4128}' \
  --additional-config '{"enable_dsa_cp":true,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}}' \
  --host 127.0.0.1 --port 8000 --served-model-name dsv4
```

## DP8 / TP1 / EP8，每rank两席位

每rank预算1026、K5、原生DSA；target FULL，draft保留原生eager，辅助元数据提前准备。

```bash
vllm serve /models/DeepSeek-V4-Flash \
  --worker-cls betterscale.worker.Worker \
  --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel \
  --quantization ascend --dtype bfloat16 --async-scheduling \
  --max-num-seqs 2 --max-num-batched-tokens 1026 --max-model-len 524288 \
  --enable-prefix-caching \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[6,12,132,264,516,1026],"max_cudagraph_capture_size":1026}' \
  --additional-config '{"enable_dsa_cp":false,"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false}}' \
  --host 127.0.0.1 --port 8000 --served-model-name dsv4
```

## 容量与启动成本

两种配置都接受最多524288token上下文（输入加输出）。不传固定KV字节，按真实启动
空闲、模型、eager/non-Torch峰值、完整共享graph池和1GiB/rank安全量自动定容。
自动路径不再以90%作为上限；这不代表与其他无协调服务共享设备时也能保证内存安全。
显式 `--kv-cache-memory-bytes` 仍选择原生手工预算，不建议用诊断用的小预算交付。

启动多一次试捕获；试验State与元数据退休，保留旧graph句柄但永不再replay。
不创建另一个完整graph池，不在稳态捕获新draft形状。
详见[容量账与验收边界](CAPACITY-0.4.zh-CN.md)、[自动定容补丁](../src/betterscale/patches/auto_kv/README.md)。

席位数不是保证能同时容纳所有满长请求。APC已通过TP/DP冷／热复用验收；此前3GiB诊断配置的抢占恢复
失败未声称修复。使用原生容量准入，保留足够的显存余量，并遵守设备租约。

## 验证与回退

健康检查用 `/health`；CPU/安装测试不代替真权重质量或硬件吞吐测量。
当前32题质量证据、历史吞吐比较与dummy容量压力在报告中分别标注。

```bash
curl --fail http://127.0.0.1:8000/health
PYTHONPATH=src python -m unittest discover -s tests
```

回退时停止服务并等workers退出，恢复原来的native worker和完整启动参数即可；
不要在活跃worker内热切补丁。K5/TP8某些形状需要LCM修复，删除worker参数也删除该修复，
不保证完全相同的参数能作为裸donor对照。对比必须核对工作量、warmup和配置。

## 前缀复用

原生hash、SWA/压缩检查点与页生命周期不变；本版解除APC关闭限制。TP/DP
冷与热各32/32检索全对，DP另通过16路重复请求。默认物理页32对应4K公共
检查点边界；命中还取决于所有状态组与draft恢复条件。DP缓存为engine-local，
上层应保持会话亲和，或使用原生`X-data-parallel-rank`请求头指定同一engine。
可用`--enable-prompt-tokens-details`观察请求实际缓存token数。
详见[前缀复用机制与边界](../prototypes/prefix-caching/INTERPRETATION.zh-CN.md)。
