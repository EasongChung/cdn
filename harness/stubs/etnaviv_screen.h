#pragma once
/* 只保留 etnaviv_ml_nn_v7.c 真正用到的字段：screen->info->npu.* 与 core_info。 */

#include "etnaviv_context.h"

struct etna_core_npu_info {
   unsigned nn_core_count;
   unsigned nn_mad_per_core;
   unsigned tp_core_count;
   unsigned on_chip_sram_size;
   unsigned axi_sram_size;
   unsigned nn_zrl_bits;
   unsigned nn_input_buffer_depth;
   unsigned nn_accum_buffer_depth;
};

struct etna_core_info {
   struct etna_core_npu_info npu;
};

struct etna_screen {
   struct etna_core_info *info;
};

struct etna_core_npu_info *etna_ml_get_core_info(struct etna_screen *screen);
