#pragma once
/* 守页分配器：把缓冲区**末端**紧贴一页 PROT_NONE 内存。
 * 任何越界读/写（哪怕 1 字节）都会立刻 SIGSEGV，不依赖 libasan/valgrind。
 * 本机 Alpine 无 libasan，这条路比 ASan 更轻且确定性相同。 */

#include <stddef.h>

/* 分配 size 字节；缓冲区末尾对齐到守护页起点。tag 用于故障归因。 */
void *guard_alloc(size_t size, const char *tag);

/* addr 若落在某个守护页（或某个缓冲区起点之前），返回该缓冲区的 tag；否则 NULL。 */
const char *guard_tag(const void *addr);

/* 命中守护页时，返回该 buffer 的 (ptr,size)，用来说明「越界了多少字节」。 */
int guard_info(const void *addr, const void **ptr, size_t *size);

/* 已分配的缓冲区数量（用于报告）。 */
int guard_count(void);
