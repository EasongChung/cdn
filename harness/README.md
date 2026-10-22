# 系数写入器越界探测（真实 Mesa 源码 + 守页）

**目的**：用**未经改写**的 `etnaviv_ml_nn_v7.c` 回答一个问题 ——
「驱动为某个形状生成 NPU 系数时，会不会读写越界？」这是 244/122 通道挂死类别的**代码级证据**，
而不是从真机现象反推。

## 为什么是守页而不是 ASan

本机是 aarch64 / Alpine，**没有 libasan**（`apk` 里也没有该包）。
`guard_alloc()` 用 `mmap` + `mprotect(PROT_NONE)` 把缓冲区**末端**紧贴一页不可访问内存，
任何 1 字节的越界读/写都会立刻 `SIGSEGV`，确定性不输 ASan，且零依赖。
（CI 上另加 `-fsanitize=address,undefined` 作为**第二重**检测。）

## 关键点：编译的是真源码

`build/etnaviv_ml_nn_v7.c` 是指向
`third_party/mesa-src-26.2.5/src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c` 的**符号链接**，
不是副本 ⇒ 不存在「测的是改写版」的漂移。仓库里核对的 md5：

```
6c1e04aa82fc7501ead299473b2e9674  etnaviv_ml_nn_v7.c
```

（用符号链接还有一个必要原因：Mesa 的 `#include "etnaviv_context.h"` 会优先命中**同目录**的真实头文件，
必须让被编译文件位于只有 stub 头的目录里，才能用 stub 顶替。）

## 依赖注入方式

`stubs/` 里只有 4 个头文件（`etnaviv_context.h` / `etnaviv_screen.h` / `etnaviv_ml_nn.h` /
`etnaviv_debug.h`）。其中 `struct etna_operation` **逐字复制**自 Mesa 的 `etnaviv_ml.h`
（仅去掉 `list_head`）—— 若字段改动，真实源码会编译失败，**不会静默漂移**。

`stubs.c` 实现 6 个外部符号：`etna_ml_get_core_info`、`etna_ml_create_bo`、`etna_bo_map`、
`etna_bo_cpu_prep/fini`、`etna_buffer_resource`、`etna_ml_calc_addition_sizes`。

## 构建与运行

```sh
cd verify/asan-harness
mkdir -p build
ln -sf "$(cd ../.. && pwd)/third_party/mesa-src-26.2.5/src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c" build/

# 生成 case 表（只导出**形状**，不含模型权重/评测图）
python3 gen_cases.py cases.h \
  mb=verify/evidence/real-model-20261021/mobilenet_v1_0.25_128_quant.tflite \
  c248=verify/evidence/perchannel-20261021/c248-perchannel.tflite \
  photos=verify/artifacts/local-surgery/legacy-clean-nodepad.tflite

gcc -O1 -g -I stubs -I . -o harness harness.c stubs.c guard.c build/etnaviv_ml_nn_v7.c \
    -Wno-unused-function -Wno-unused-variable

./harness          # 单套硬件参数（cores=8, zrl=4, accum=4096, input=3072）
./harness -v       # 打印每个 case 的结果
./harness --sweep  # 扫 4 套硬件参数，确认结论不依赖猜测值
```

CI 上（x86_64 或 aarch64 均可）加一重 sanitizer：

```sh
gcc -O1 -g -fsanitize=address,undefined -I stubs -I . \
    harness.c stubs.c guard.c build/etnaviv_ml_nn_v7.c -o harness-asan
```

## 判据

| 输出 | 含义 |
|---|---|
| `ok` | 该形状下写入器的所有访问都落在缓冲区内 |
| `FAIL ... -> weights[...]` | **权重越界读**（幻影通道） |
| `FAIL ... -> biases[...]` | **偏置越界读**（幻影通道） |
| `FAIL ... -> coeff_bo[...]` | **系数缓冲越界写**（`calculate_weight_bo_size` 估算偏小） |
| `FAIL ... sig=8` | 除零（驱动里会 abort/挂死） |

## 2026-10-22 基线结果（aarch64 实跑）

31 个去重形状（MobileNetV1 + c248 + Photos 折叠/展开后的真实形状）：

| 结论 | 数值 |
|---|---|
| 越界形状 | **8 / 31**，全部 `oc % 8 != 0`（1001、122、244） |
| 越界偏移 | 全部**正好等于缓冲区末尾**（读 1 字节越界） |
| `oc % 8 == 0` 的形状 | **全部干净**（8/16/24/32/64/128/248/256/488/2048） |

⇒ ① 幻影通道机制在**代码级**被复现；② 混合接管的 10 个候选层（24/488/2048）在写入器层面**无越界**；
③ MobileNetV1 的 1001 通道层确实越界 —— 与真机「它没有被接管」一致。

## 已知边界（不要过度解读）

* 只覆盖**系数写入器**，不覆盖 `expand_depthwise` / `pointwise_to_2x2` / `transpose` 等**变换**阶段
  （那些在 `etnaviv_ml_nn.c`，依赖更重）。
* **不覆盖 GPU 侧**：幻影通道最危险的是「输出偏移越界写」，那是 GPU 执行的，host 看不到；
  本 harness 用**同根因的 host 侧越界读**作为代理证据。
* 不覆盖 CONCATENATION / SPLIT / PAD / RELU 等非卷积作业（它们不走系数写入器）。

## CI（已跑通，2026-10-22）

用户授权用 `EasongChung/cdn`（public、闲置）跑 CI，**不碰 `rk_syno_pack`**。

* 分支：`harness-asan`；run: https://github.com/EasongChung/cdn/actions/runs/39726424569
* 判定用 `gate.py`：**实测越界集合必须逐字等于 `C % cores_used != 0` 的预测**
  （不是阈值；cores=8 时 8 个，cores=4 时 5 个 —— 244 在 4 核下 `244%4==0` ⇒ 不越界）
* ASan 第二遍给出精确栈：`write_core_6` @ `etnaviv_ml_nn_v7.c:234`（幻影 out_channel 解引用）
* CI 载荷不含任何模型权重或评测图（`cases.h` 只有形状数字）

## 第二遍：depthwise × per-channel 折叠后态（`harness-dw`）

`requantize_weights()`（`etnaviv_ml_nn.c:231`）对 depthwise 强制 `input_channels = 1`，
把权重缓冲区从 `C·k²·C` 缩到 `C·k²`，但 `operation->input_channels` 仍为 `C`
⇒ 系数写入器越界读 **C 倍**。详见 `verify/reports/54`。

* 提取真实函数：`extract_funcs.py --src mesa/etnaviv_ml_nn.c --func requantize_weights
  --expect-body-md5 b648d88873bda693860424212471cdf3`（body 逐字节相同，上游一改就红）
* 判定：`gate_dw.py` —— dw 案必须越界到缓冲区末尾且 `post == oc·k²`；控制案必须干净
* 负控制：同形状非 depthwise（`ic` 不缩）、depthwise 但 per-tensor（不折叠）

⚠️ 交付件里已有 `och-patch-perchannel-layout` 补丁覆盖这条（二进制标记可查），
所以本门是**回归门**，不是现网缺陷。若上游自己修好，本门会失败 —— 那是**升级信号**。

### 已纠正的 case 表

`gen_cases.py` 原先按「4D 的 `inputs[1]` 就是权重」判断，把 `CONCATENATION`/elementwise
的常量张量当成卷积核，造出 50×50 / 25×25 / 13×13 三个**模型里不存在**的形状。
现在按 opcode 名称只取 `CONV_2D` / `DEPTHWISE_CONV_2D`：31 → 28 个真实形状。

## 第三遍：fold-early（报告 55 方案 A，`harness-fold`）

第二遍证明的是**上游后态会越界**；本遍证明**修好之后不会越界**，而且证明方式不是「全干净」。

* `build_fold.sh`：把源码复制一份 → 用 `build-recipe/patch-etnaviv-perchannel-foldearly.py`
  打补丁 → 从**补丁后**的副本里提取 `och_fold_channel / calc_quantization /
  requantize_weights / expand_depthwise / transpose`；同时从**未打补丁**的原件提取
  `upstream_calc_quantization / upstream_requantize_weights`（用 `--signature` 改名，
  **函数体逐字节不动**）⇒ 同一进程内 A/B。
* 三种模式（形状表复用 `cases_dw.h`）：
  * `mode=fe`：`fold(原地) → expand → transpose → 写入器`
  * `mode=up`：`expand → transpose → upstream fold(C 倍缩容) → 写入器`（负控制）
  * `mode=i2`：非 depthwise 走两条管线，比较**transpose 之后**的缓冲，必须 `memcmp == 0`
* 判定 `gate_fold.py`：**fe 的实测必须逐字等于「幻影通道模型」的预测**
  （`C % min(C,cores) == 0` 才期望干净）。理由：两个缺陷互相独立，
  fold 缺陷修好后 `C % cores_used != 0` 的层仍会越界（那是报告 53 的幻影通道，
  现网由 `TEFLON_PHANTOM_GATE` 拦）。把它们混成一个「必须全干净」的门，
  会在幻影通道被真正修好那天给出**假红**，也会在今天给出**假绿**。
* 反作弊：fe 里必须同时出现 OK 与 OOB（否则门无分辨力）；跨模式断言
  `fe.sz == up.sz × C`；门自身另有 3 条文本突变负控制。

本机（Alpine / gcc 15）实测：**84 项检查 0 失败**；CI 在 glibc 下再跑一遍。
