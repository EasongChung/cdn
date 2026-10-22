#include "guard.h"

#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define MAX_GUARDS 8192

struct guard_ent {
   char *ptr;        /* 用户可见的缓冲区起点 */
   size_t size;      /* 用户请求的大小 */
   char *guard;      /* 守护页起点 */
   char *base;       /* mmap 基址 */
   size_t maplen;
   char tag[96];
};

static struct guard_ent g_ents[MAX_GUARDS];
static int g_n;

static size_t page_size(void)
{
   static size_t ps;
   if (!ps)
      ps = (size_t)sysconf(_SC_PAGESIZE);
   return ps;
}

void *
guard_alloc(size_t size, const char *tag)
{
   size_t ps = page_size();
   size_t data = ((size + ps - 1) / ps) * ps;
   if (data == 0)
      data = ps;
   size_t maplen = data + ps; /* 末尾一页守护 */

   char *base = mmap(NULL, maplen, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (base == MAP_FAILED) {
      perror("mmap");
      return NULL;
   }
   if (mprotect(base + data, ps, PROT_NONE) != 0) {
      perror("mprotect");
      munmap(base, maplen);
      return NULL;
   }

   char *ptr = base + data - size; /* 末端紧贴守护页 */

   if (g_n < MAX_GUARDS) {
      struct guard_ent *e = &g_ents[g_n++];
      e->ptr = ptr;
      e->size = size;
      e->guard = base + data;
      e->base = base;
      e->maplen = maplen;
      snprintf(e->tag, sizeof(e->tag), "%s", tag ? tag : "?");
   }
   return ptr;
}

const char *
guard_tag(const void *addr)
{
   const char *a = addr;
   for (int i = 0; i < g_n; i++) {
      struct guard_ent *e = &g_ents[i];
      if (a >= e->guard && a < e->guard + page_size())
         return e->tag;
      if (a < e->ptr && a >= e->base)
         return e->tag; /* 起点之前（下越界） */
   }
   return NULL;
}

int
guard_info(const void *addr, const void **ptr, size_t *size)
{
   const char *a = addr;
   for (int i = 0; i < g_n; i++) {
      struct guard_ent *e = &g_ents[i];
      if ((a >= e->guard && a < e->guard + page_size()) ||
          (a < e->ptr && a >= e->base)) {
         *ptr = e->ptr;
         *size = e->size;
         return 1;
      }
   }
   return 0;
}

int
guard_count(void)
{
   return g_n;
}
