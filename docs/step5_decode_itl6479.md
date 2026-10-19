# Step5 Decode 64.79ms：原生集成与实测记录

2026-10-19，p5c1–p5c4，四节点各四张 BW1300_LC。DP4/TP4/EP16，
Channel FP8 W8A8、FP8 E4M3 KV，输入65536、输出4096、并发64，
每DP 16个请求。MTP3，synthetic拒绝采样参数
`[0.838992,0.598322,0.362686]`，保持严格FA DSA。

| 配置 | Decode tokens/s | ITL均值 ms | ITL P99 ms |
| --- | ---: | ---: | ---: |
| 本轮重测基线 | 2628.4730 | 68.3991 | 73.5808 |
| 真实专家重排＋打包AG＋路由/共享专家配置 | 2763.5631 | 64.7934 | 69.3880 |

候选完成一整轮预热和一轮正式测量，64/64完成，零抢占；正式窗口没有编译或缓存更新。
接受率为服务计数器accepted/drafted=59.8287%，为synthetic压测设置下的值。
公共Decode窗口91.491670秒、252843个token、90311个ITL样本。
ITL是相邻流式chunk的间隔；等效TPOT为23.1585ms，二者口径不同。
结果尚未达到ITL<60ms，也没有完成三轮稳定性验证。

## 已纳入插件的代码

- `vllm_hcu/distributed/device_communicators/packed_all_gather.py`：
  FP8激活、FP32路由权重、整数专家ID、FP32 scale按原始字节打包。
  四个已分组的AllGather合成一次，随后解包，数值与dtype保持一致。
  仅支持本次uniform M64对应布局；其他布局保留原路径。
  非连续输入先本地整理，避免某些rank因stride不同进入不同collective。
- `patch_cuda_communicator.py`：沿用已有适配器注册，
  `VLLM_HCU_DP_PACKED_AG=1`显式启用，默认关闭；已有单节点IPC路径优先。
- `router_gemm.py`：`HCU_STEP5_ROUTER_SHARED_MAX_M=32`限制路由/共享激活融合范围，
  默认仍为64。共享专家沿用已有`HCU_SHARED_EXPERTS_STREAM_MIN_TOKENS=64`配置。
- 专家映射位于`benchmarks/step5/assets/expert_placement_decode_real_c64_ep16.pt`，
  shape为[88,352]，由本次真实Decode路由统计拟合。它与原先缺失的
  `expert_placement_prefill.pt`是不同文件。

## 原生启动

```bash
# 在xuwq_vllm_1017容器内，每节点分别运行rank0/1/2/3。
export STEP5_STATE_DIR=/var/lib/xuwq_step5_itl6479_native
export STEP5_TOKENIZER=/opt/xuwq_step5_new_20261018/tokenizer-bytelevel
bash benchmarks/step5/launch_decode_itl6479.sh 0
```

启动入口通过插件已有checkpoint loader加载上述PT映射，启用打包AG并设置融合上限。
无需worker extension、RPC注入或另行安装优化库。权重、缓存和日志应放在数据盘支撑的目录。

**验证范围：64.79ms来自此前运行时扩展方案。原生版保留相同算法和参数，
但改为启动时启用及加载PT，因此不能将该数值视为原生版已复测结果。**
原生版已通过通信回归、字节一致性与graph replay检查；四节点完整性能复测等待
p5c4重启后的`/data`恢复。重启原因未确认。后续共享激活缓存、提前finalize重叠及
通信器热切换试验均不属于本次原生集成。

本轮Decode实例的Prefill吞吐为23727.3937tps。此前DP16/TP1/EP16的
Prefill专用配置29085.03tps是另一实例的独立结果，不并入同一配置成绩。
