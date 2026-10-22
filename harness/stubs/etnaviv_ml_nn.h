#pragma once
/* struct etna_operation 与 Mesa 26.2.5 的 etnaviv_ml.h **逐字一致**（只去掉 list_head 成员，
 * 因为 v7 写入器不用它）。任何字段改名都会让真实源码编译失败，从而不会静默漂移。 */

#include "etnaviv_context.h"
#include "etnaviv_screen.h"

#define MAX_TENSORS 8

enum etna_job_type {
   ETNA_JOB_TYPE_NN = 0,
   ETNA_JOB_TYPE_ADD,
   ETNA_JOB_TYPE_CONCAT,
   ETNA_JOB_TYPE_SPLIT,
   ETNA_JOB_TYPE_TP,
};

enum etna_ml_tp_type {
   ETNA_ML_TP_NONE = 0,
};

struct etna_operation {
   enum etna_job_type type;
   enum etna_ml_tp_type tp_type;

   bool addition;
   bool depthwise;
   bool pointwise;
   bool fully_connected;
   bool pooling_first_pixel;
   bool padding_same;
   bool relu;

   unsigned stride;

   unsigned input_tensors[MAX_TENSORS];
   unsigned input_count;
   unsigned input_tensor_sizes[MAX_TENSORS];

   unsigned input_width;
   unsigned input_height;
   unsigned input_channels;
   uint8_t input_zero_point;
   float input_scale;

   unsigned output_tensors[MAX_TENSORS];
   unsigned output_count;
   unsigned output_tensor_sizes[MAX_TENSORS];

   unsigned output_width;
   unsigned output_height;
   unsigned output_channels;
   uint8_t output_zero_point;
   float output_scale;

   uint8_t *weight_tensor;
   unsigned weight_tensor_size;
   unsigned weight_width;
   unsigned weight_height;
   uint8_t weight_zero_point;
   float weight_scale;
   bool weight_signed;

   uint8_t addition_offset;

   uint8_t *bias_tensor;
   unsigned bias_tensor_size;

   unsigned pad_before_x;
   unsigned pad_after_x;
   unsigned pad_before_y;
   unsigned pad_after_y;
   unsigned pad_before_z;
   unsigned pad_after_z;
};

struct etna_ml_subgraph {
   struct etna_screen *screen;
};

void etna_ml_calc_addition_sizes(unsigned *input_width, unsigned *input_height, unsigned *input_channels,
                                 unsigned *output_width, unsigned *output_height, unsigned *output_channels);

unsigned etna_ml_calculate_tiling_v7(struct etna_screen *screen, const struct etna_operation *operation,
                                     unsigned *tile_width_out, unsigned *tile_height_out);

struct etna_bo *etna_ml_create_coeffs_v7(struct etna_ml_subgraph *subgraph,
                                        const struct etna_operation *operation, unsigned *cache_size);
