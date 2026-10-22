#pragma once
/* 跨 TU 声明（extract_funcs.py 每个函数生成一个 TU，调用方需要原型）。
 * 只声明**被提取出来的**函数；签名必须与源码逐字一致。 */
#include "etnaviv_ml_nn.h"
#include "pipe_ml.h"

unsigned och_fold_channel(const struct etna_operation *operation, unsigned i, bool depthwise);
void calc_quant_params(float min, float max, float *out_scale, uint8_t *out_zero_point);

/* Mesa 的 SWAP 来自 util/macros.h；stub 环境不拉那一整套，这里按原件语义补上。
 * 若哪天 Mesa 改了 SWAP 的语义，calc_quant_params 的提取版会与原件不同 ——
 * 但 body md5 已钉死（ce2291db…），原件一变就 fail-closed。 */
#ifndef SWAP
#define SWAP(a, b) do { __typeof__(a) tmp = (a); (a) = (b); (b) = tmp; } while (0)
#endif
