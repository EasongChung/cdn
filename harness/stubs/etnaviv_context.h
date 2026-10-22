#pragma once
/* 最小 stub：让真实的 etnaviv_ml_nn_v7.c 能在主机上编译运行。
 * 只提供类型/宏，不改变任何算法。 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* Mesa util/macros.h 的三个常用宏（逐字相同） */
#define align(x, y) (((x) + (y) - 1) & ~((y) - 1))
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))

struct pipe_resource {
   int dummy;
};

struct etna_buffer_resource {
   struct etna_bo *bo;
};

struct etna_bo {
   void *map;
   size_t size;
   const char *tag;
};

#define DRM_ETNA_PREP_WRITE 0x2

struct etna_screen;

void *etna_bo_map(struct etna_bo *bo);
void etna_bo_cpu_prep(struct etna_bo *bo, unsigned flags);
void etna_bo_cpu_fini(struct etna_bo *bo);
struct etna_bo *etna_ml_create_bo(struct etna_screen *screen, size_t size);
struct etna_buffer_resource *etna_buffer_resource(struct pipe_resource *res);
