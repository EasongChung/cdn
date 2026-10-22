#pragma once
/* pipe 层的最小声明 —— **只保留 requantize_weights() 实际用到的字段**。
 *
 * 注意：这里故意不复刻整个 pipe 层。提取出的真实函数体一旦用到没声明的字段，
 * 编译会直接失败 ⇒ 不会静默漂移（这比「字段齐全但语义可能不同」更安全）。
 */

#include <stdbool.h>
#include <stdint.h>

struct pipe_tensor {
   unsigned dims[4];
   void *data;
   float scale;
   int zero_point;
   float *scales;        /* per-channel：长度 = dims[3]（或 depthwise 的 C） */
   uint8_t *zero_points; /* per-channel */
   bool is_signed;
};

struct pipe_ml_operation {
   struct {
      struct pipe_tensor *weight_tensor;
      struct pipe_tensor *bias_tensor;
      unsigned depthwise;
      unsigned pointwise;
      unsigned stride_x;
      unsigned stride_y;
      unsigned padding_top;
      unsigned padding_bottom;
      unsigned padding_left;
      unsigned padding_right;
   } conv;
};
