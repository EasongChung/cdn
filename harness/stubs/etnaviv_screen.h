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

/* 真实 Mesa 的 etna_screen 有 `struct etna_specs specs` 成员（etnaviv_screen.h:67）。
 * 折叠/布局相关的源码（transpose 等）要读 specs.nn_core_version，故此处必须镜像它。
 * 只保留 ml_nn.c 用到的字段；缺字段会让真实源码**编译失败**（不会静默漂移）。 */
struct etna_specs {
   unsigned nn_core_version;
};

struct etna_screen {
   struct etna_core_info *info;
   struct etna_specs specs;
};

struct etna_core_npu_info *etna_ml_get_core_info(struct etna_screen *screen);
