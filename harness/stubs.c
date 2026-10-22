#include "etnaviv_context.h"
#include "etnaviv_screen.h"
#include "etnaviv_ml_nn.h"
#include "guard.h"
#include "stubs_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct etna_core_info g_core;
static struct etna_screen g_screen = { .info = &g_core };
static const char *g_case = "?";

void
stubs_set_case(const char *name)
{
   g_case = name;
}

void
stubs_config(unsigned cores, unsigned zrl_bits, unsigned accum_depth, unsigned input_depth)
{
   g_core.npu.nn_core_count = cores;
   g_core.npu.nn_zrl_bits = zrl_bits;
   g_core.npu.nn_accum_buffer_depth = accum_depth;
   g_core.npu.nn_input_buffer_depth = input_depth;
   /* 其余字段 v7 写入器不用 */
   g_core.npu.nn_mad_per_core = 1024;
   g_core.npu.tp_core_count = 0;
}

struct etna_core_npu_info *
etna_ml_get_core_info(struct etna_screen *screen)
{
   (void)screen;
   return &g_core.npu;
}

struct etna_screen *
stubs_screen(void)
{
   return &g_screen;
}

struct etna_bo *
etna_ml_create_bo(struct etna_screen *screen, size_t size)
{
   (void)screen;
   struct etna_bo *bo = calloc(1, sizeof(*bo));
   if (!bo)
      return NULL;

   char tag[160];
   snprintf(tag, sizeof(tag), "coeff_bo[%s]", g_case);
   bo->map = guard_alloc(size, tag);
   bo->size = size;
   bo->tag = strdup(tag); /* 泄漏是故意的：故障归因需要稳定字符串 */
   return bo;
}

void *
etna_bo_map(struct etna_bo *bo)
{
   return bo ? bo->map : NULL;
}

void
etna_bo_cpu_prep(struct etna_bo *bo, unsigned flags)
{
   (void)bo;
   (void)flags;
}

void
etna_bo_cpu_fini(struct etna_bo *bo)
{
   (void)bo;
}

/* map_resource() 在 v7 里只为 weight/bias 指针服务；harness 直接给指针，
 * 这个函数只为让真实源码链接通过。 */
static struct etna_buffer_resource g_bufres;

struct etna_buffer_resource *
etna_buffer_resource(struct pipe_resource *res)
{
   (void)res;
   return &g_bufres;
}

/* v7 里仅在 addition 路径调用；harness 不测 addition，但要满足链接。 */
void
etna_ml_calc_addition_sizes(unsigned *input_width, unsigned *input_height, unsigned *input_channels,
                            unsigned *output_width, unsigned *output_height, unsigned *output_channels)
{
   (void)input_width;
   (void)input_height;
   (void)input_channels;
   (void)output_width;
   (void)output_height;
   (void)output_channels;
}
