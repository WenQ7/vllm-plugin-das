# Step5 Decode 64.79ms：历史成绩与原生迁移验证

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
- 专家映射`expert_placement_decode_real_c64_ep16.pt`是外部运行资产，不纳入源码仓库。
  性能资料包将其放在`assets/`，shape为[88,352]，由本次真实Decode路由统计拟合。它与原先缺失的
  `expert_placement_prefill.pt`是不同文件。

## 原生启动

```bash
# 在xuwq_vllm_1017容器内，每节点分别运行rank0/1/2/3。
export STEP5_STATE_DIR=/var/lib/xuwq_step5_itl6479_native
export STEP5_TOKENIZER=/opt/xuwq_step5_new_20261018/tokenizer-bytelevel
export STEP5_DECODE_PLACEMENT=/opt/xuwq_step5_itl6479_native/assets/expert_placement_decode_real_c64_ep16.pt
bash benchmarks/step5/launch_decode_itl6479.sh 0
```

启动入口通过插件已有checkpoint loader加载上述PT映射，启用打包AG并设置融合上限。
四节点容器必须均能读取`STEP5_DECODE_PLACEMENT`指定的文件；未指定时脚本直接报错。
无需worker extension、RPC注入或另行安装优化库。权重、缓存和日志应放在数据盘支撑的目录。

**验证范围：64.79ms来自运行时扩展方案，不能作为原生版的成绩。**
`/data`恢复后，同日已完成相同C64、64K/4096、MTP3、FA DSA口径的复测：

| 实际执行版本 | Decode tokens/s | ITL均值 ms | ITL P99 ms |
| --- | ---: | ---: | ---: |
| 外挂历史方案复现 | 2774.0950 | 64.8157 | 70.3956 |
| Python基于`ae60f22`、缺少E22表的旧部署包 | 1571.6223 | 114.3302 | 120.7674 |
| 同一主体代码，生效表切为Git中的完整E22表 | 2785.1913 | 64.5636 | 69.5529 |

各正式轮均完成一轮完整预热，64/64完成、零抢占，四节点正式窗口无编译或缓存更新。
修复部署后的完整预热为65.0224ms；正式Decode为2785.1913tps，同一Decode实例的
Prefill为23718.3827tps。此轮没有加入下述共享激活修复，唯一变量是生效的MoE参数表。

退化根因已通过单变量复测确认：旧打包脚本在D5快照上只覆盖`f6fee53f..ae60f22`
之间的改动，但E22表在`f6fee53f`时已经存在，因此没有被覆盖进包。部署表实际只有
E44键，EP16落到了未调优的library配置。排除CRLF差异后，411个核心Git文件的唯一
内容差异就是`ep_table.json`。仓库中的完整表从一开始就有E22，故114ms不能表示
完整Git源码的性能。

`check_ep_table.py`现在在加载模型前核对生效表；`package_runtime.py`从指定Git提交
完整导出文件，并在解包时逐个核对Git对象，避免将不匹配的旧快照标为新提交。
PT仍外置。延后重排/捕图实验是在缺表环境下进行的，已经归档，不作为当前启动依赖。

2026-10-20代码复核新增：已检查的rank0主模型285个编译子图缓存键在两版间全部相同；
四节点16卡直接对照的打包AllGather及更换输入后Graph重放结果逐字节一致。
这证明了该通信实现可以内置，但不证明整模型运行路径或性能已经等价。
实际主体轨迹也确认执行了打包AG、EP优化核和FA稀疏注意力。参数表遗漏修复后已恢复
到参考性能，但ITL<60ms目标尚未达到。

本地还修复了内部路由路径丢弃共享激活的问题：`MoERunner`仅接收显式声明
`returns_fused_shared_activation`的gate第二返回值，并交给`Step5MLP`使用。
普通Linear的bias不作激活处理。该修复有3项CPU语义检查，以及实际GPU上
K4096/路由352/共享中间维384、M16/32/64/65的数值和更换输入后Graph重放检查；
后者相对非融合GPU实现的最大相对L2误差约0.02%。这验证的是局部激活传递，
不能作为完整模型精度验证或额外性能收益证明；该修复的整模型性能调优仍在进行。

共享激活缓存、提前finalize重叠及通信器热切换试验均不属于原64.79ms成绩。

本轮Decode实例的Prefill吞吐为23727.3937tps。此前DP16/TP1/EP16的
Prefill专用配置29085.03tps是另一实例的独立结果，不并入同一配置成绩。
