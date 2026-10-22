#!/bin/sh
# V4a 构建：fold-early 守页验证（本地与 CI 共用同一条路径）
#
# 关键纪律：
#   1. **被测函数体逐字节来自源码**（extract_funcs.py），不复制、不改写
#   2. fold-early 的函数体来自"打过补丁的副本"（build/nn-foldearly.c），
#      upstream 对照来自"未打补丁的原件"（mesa/etnaviv_ml_nn.c）
#   3. 上游两个对照函数用 --signature 改名，**函数体不动** ⇒ 同一进程内可直接 A/B
#
# 用法：./build_fold.sh <pristine_nn.c 路径> [gcc 额外参数...]
set -eu

SRC="${1:?用法: build_fold.sh <etnaviv_ml_nn.c> [额外 gcc 参数]}"
shift || true

HERE=$(cd "$(dirname "$0")" && pwd)
# 补丁器位置：本地在 build-recipe/；CI 仓（cdn）只同步 harness/**，故允许
#   FOLD_PATCHER 显式指定，或退回到 harness/ 内的副本。
PATCHER="${FOLD_PATCHER:-$HERE/../../build-recipe/patch-etnaviv-perchannel-foldearly.py}"
if [ ! -f "$PATCHER" ]; then
	PATCHER="$HERE/patch-etnaviv-perchannel-foldearly.py"
fi
if [ ! -f "$PATCHER" ]; then
	echo "ERROR: 找不到 fold-early 补丁器（试过 FOLD_PATCHER 与 harness/ 内副本）" >&2
	exit 1
fi
echo "== patcher: $PATCHER =="
mkdir -p "$HERE/build"

echo "== 1/4 生成 fold-early 版源码副本 =="
cp "$SRC" "$HERE/build/nn-foldearly.c"
python3 "$PATCHER" "$HERE/build/nn-foldearly.c" --verify
python3 "$PATCHER" "$HERE/build/nn-foldearly.c"

echo "== 2/4 提取 fold-early 的函数体（来自已打补丁副本）=="
python3 "$HERE/extract_funcs.py" --src "$HERE/build/nn-foldearly.c" --func och_fold_channel \
  --out "$HERE/build/fe_helper.c" \
  --signature "unsigned och_fold_channel(const struct etna_operation *operation, unsigned i, bool depthwise)" \
  --includes etnaviv_ml_nn.h,pipe_ml.h

python3 "$HERE/extract_funcs.py" --src "$HERE/build/nn-foldearly.c" --func calc_quant_params \
  --out "$HERE/build/fe_params.c" \
  --signature "void calc_quant_params(float min, float max, float *out_scale, uint8_t *out_zero_point)" \
  --includes etnaviv_ml_nn.h,pipe_ml.h \
  --expect-body-md5 ce2291dbd2a627a9447ee9fe7f539de8

python3 "$HERE/extract_funcs.py" --src "$HERE/build/nn-foldearly.c" --func calc_quantization \
  --out "$HERE/build/fe_calc.c" \
  --signature "void calc_quantization(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation, struct etna_operation *operation, float *out_scale, uint8_t *out_zero_point)" \
  --includes etnaviv_ml_nn.h,pipe_ml.h

python3 "$HERE/extract_funcs.py" --src "$HERE/build/nn-foldearly.c" --func requantize_weights \
  --out "$HERE/build/fe_req.c" \
  --signature "void requantize_weights(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation, struct etna_operation *operation)" \
  --includes etnaviv_ml_nn.h,pipe_ml.h

python3 "$HERE/extract_funcs.py" --src "$HERE/build/nn-foldearly.c" --func expand_depthwise \
  --out "$HERE/build/fe_expand.c" \
  --signature "void expand_depthwise(struct etna_ml_subgraph *subgraph, struct etna_operation *operation)" \
  --includes etnaviv_ml_nn.h,pipe_ml.h

python3 "$HERE/extract_funcs.py" --src "$HERE/build/nn-foldearly.c" --func transpose \
  --out "$HERE/build/fe_transpose.c" \
  --signature "void transpose(struct etna_ml_subgraph *subgraph, struct etna_operation *operation)" \
  --includes etnaviv_ml_nn.h,pipe_ml.h

echo "== 3/4 提取 upstream 对照（来自**未打补丁**原件，函数体逐字节不变）=="
python3 "$HERE/extract_funcs.py" --src "$SRC" --func calc_quantization \
  --out "$HERE/build/up_calc.c" \
  --signature "void upstream_calc_quantization(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation, struct etna_operation *operation, float *out_scale, uint8_t *out_zero_point)" \
  --includes etnaviv_ml_nn.h,pipe_ml.h \
  --expect-body-md5 660a01edd6e26f88e128821065aec127

python3 "$HERE/extract_funcs.py" --src "$SRC" --func requantize_weights \
  --out "$HERE/build/up_req.c" \
  --signature "void upstream_requantize_weights(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation, struct etna_operation *operation)" \
  --includes etnaviv_ml_nn.h,pipe_ml.h \
  --expect-body-md5 b648d88873bda693860424212471cdf3

echo "== 4/4 编译 =="
# shellcheck disable=SC2086
gcc -O1 -g -I "$HERE/stubs" -I "$HERE" -include "$HERE/protos_fold.h" -o "$HERE/harness-fold" \
  "$HERE/harness_fold.c" "$HERE/stubs.c" "$HERE/guard.c" \
  "$HERE/build/fe_helper.c" "$HERE/build/fe_params.c" "$HERE/build/fe_calc.c" \
  "$HERE/build/fe_req.c" "$HERE/build/fe_expand.c" "$HERE/build/fe_transpose.c" \
  "$HERE/build/up_calc.c" "$HERE/build/up_req.c" \
  "$HERE/build/etnaviv_ml_nn_v7.c" \
  -Wno-unused-function -Wno-unused-variable -Wno-unused-parameter -lm "$@"

echo "构建完成：$HERE/harness-fold"
