/* 系数写入器越界探测（真实源码 + 守页）
 *
 * 编译的是 third_party/mesa-src-26.2.5/.../etnaviv_ml_nn_v7.c **原文件**，
 * 不做任何复制/改写；只把它的外部依赖（screen/BO）stub 掉。
 *
 * 判据：
 *   - 权重缓冲、偏置缓冲、系数 BO 都用 guard_alloc 分配（末尾紧贴 PROT_NONE）
 *   - 任何越界读/写 ⇒ SIGSEGV，被 handler 捕获 ⇒ 记为该 case 的 FAIL
 *   - 除零 ⇒ SIGFPE，同样记为 FAIL（驱动里这会 abort/挂死）
 */

#include "etnaviv_context.h"
#include "etnaviv_screen.h"
#include "etnaviv_ml_nn.h"
#include "guard.h"
#include "stubs_api.h"
#include "cases.h"

#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static sigjmp_buf g_jmp;
static volatile sig_atomic_t g_in_case;
static volatile unsigned long g_fault_addr;
static volatile int g_fault_sig;

static void
fault_handler(int sig, siginfo_t *si, void *ctx)
{
   (void)ctx;
   if (!g_in_case) {
      signal(sig, SIG_DFL);
      raise(sig);
      return;
   }
   g_fault_sig = sig;
   g_fault_addr = (unsigned long)si->si_addr;
   siglongjmp(g_jmp, 1);
}

struct hwcfg {
   unsigned cores;
   unsigned zrl;
   unsigned accum;
   unsigned inbuf;
};

static int
run_case(const struct testcase *c, const char *hwtag, int verbose)
{
   unsigned ic_eff = c->ic ? c->ic : 1;
   size_t wsize = (size_t)c->oc * (size_t)c->kh * (size_t)c->kw * (size_t)ic_eff;
   size_t bsize = (size_t)c->oc * sizeof(int32_t);

   char wtag[160], btag[160];
   snprintf(wtag, sizeof(wtag), "weights[%s|%s]", hwtag, c->name);
   snprintf(btag, sizeof(btag), "biases[%s|%s]", hwtag, c->name);

   uint8_t *w = guard_alloc(wsize, wtag);
   uint8_t *b = guard_alloc(bsize, btag);
   if (!w || !b)
      return -1;

   for (size_t i = 0; i < wsize; i++)
      w[i] = (uint8_t)((i * 37u + 11u) & 0xFFu);
   for (size_t i = 0; i < bsize; i++)
      b[i] = (uint8_t)((i * 91u + 7u) & 0xFFu);

   struct etna_operation op;
   memset(&op, 0, sizeof(op));
   op.type = ETNA_JOB_TYPE_NN;
   op.depthwise = c->depthwise;
   op.pointwise = c->pointwise;
   op.stride = c->stride ? c->stride : 1;
   /* w/h 只影响系数数值，不影响越界；用输出尺寸即可 */
   op.input_width = c->ow;
   op.input_height = c->oh;
   op.input_channels = ic_eff;
   op.input_zero_point = 128;
   op.output_width = c->ow;
   op.output_height = c->oh;
   op.output_channels = c->oc;
   op.output_zero_point = 128;
   op.weight_tensor = w;
   op.weight_tensor_size = (unsigned)wsize;
   op.weight_width = c->kw;
   op.weight_height = c->kh;
   op.weight_zero_point = 0;
   op.weight_scale = 1.0f / 255.0f;
   op.weight_signed = true;
   op.bias_tensor = b;
   op.bias_tensor_size = (unsigned)bsize;

   stubs_set_case(c->name);

   unsigned cache_size = 0;
   g_fault_sig = 0;
   g_fault_addr = 0;
   g_in_case = 1;
   if (sigsetjmp(g_jmp, 1) == 0) {
      etna_ml_create_coeffs_v7((struct etna_ml_subgraph *)&(struct etna_ml_subgraph){
                                  .screen = stubs_screen()},
                              &op, &cache_size);
      g_in_case = 0;
      if (verbose)
         printf("  ok   %-52s cache=%u\n", c->name, cache_size);
      return 0;
   }
   g_in_case = 0;

   const void *gptr = NULL;
   size_t gsize = 0;
   int hit = guard_info((const void *)g_fault_addr, &gptr, &gsize);
   const char *tag = guard_tag((const void *)g_fault_addr);

   printf("  FAIL %-52s sig=%d addr=%#lx %s%s\n", c->name, g_fault_sig, g_fault_addr,
          tag ? "-> " : "", tag ? tag : "(非守护页)");
   if (hit && gptr) {
      long off = (long)((const char *)g_fault_addr - (const char *)gptr);
      printf("       命中缓冲区 tag=%s size=%zu 越界偏移=%ld\n", tag ? tag : "?", gsize, off);
   }
   return 1;
}

int
main(int argc, char **argv)
{
   int verbose = 0;
   int sweep = 0;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "-v"))
         verbose = 1;
      else if (!strcmp(argv[i], "--sweep"))
         sweep = 1;
   }

   struct sigaction sa;
   memset(&sa, 0, sizeof(sa));
   sa.sa_sigaction = fault_handler;
   sa.sa_flags = SA_SIGINFO | SA_NODEFER;
#if !defined(NO_FAULT_HANDLER)
   /* 开 ASan 的第二遍时用 -DNO_FAULT_HANDLER：让 ASan 自己接管 SIGSEGV，
      从而打印完整栈（本 harness 的 handler 只能报「哪个缓冲区越界」）。 */
   sigaction(SIGSEGV, &sa, NULL);
   sigaction(SIGBUS, &sa, NULL);
   sigaction(SIGFPE, &sa, NULL);
#endif

   /* A311D 的 cores_used=8 已由 NNDUMP 实测确认；accum/inbuf 未在本项目实测过，
    * 故对它们做扫描，确保结论不依赖某个猜测值。 */
   struct hwcfg cfgs[] = {
       {8, 4, 4096, 3072},
       {8, 0, 4096, 3072},
       {8, 4, 1024, 1024},
       {4, 4, 4096, 3072},
   };
   int ncfg = sweep ? (int)(sizeof(cfgs) / sizeof(cfgs[0])) : 1;

   printf("== 系数写入器越界探测（真实 etnaviv_ml_nn_v7.c + 守页）==\n");
   printf("   case 数=%zu  hw 配置数=%d\n", sizeof(CASES) / sizeof(CASES[0]), ncfg);

   int total_fail = 0;
   for (int k = 0; k < ncfg; k++) {
      char hwtag[64];
      snprintf(hwtag, sizeof(hwtag), "cores=%u,zrl=%u,acc=%u,in=%u",
               cfgs[k].cores, cfgs[k].zrl, cfgs[k].accum, cfgs[k].inbuf);
      stubs_config(cfgs[k].cores, cfgs[k].zrl, cfgs[k].accum, cfgs[k].inbuf);
      printf("\n-- hw %s --\n", hwtag);
      int fails = 0;
      for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
         int r = run_case(&CASES[i], hwtag, verbose);
         if (r != 0)
            fails++;
      }
      printf("   小结：%zu 个 case 中 %d 个越界/异常\n",
             sizeof(CASES) / sizeof(CASES[0]), fails);
      total_fail += fails;
   }

   printf("\n== 总计：%d 次越界/异常 ==\n", total_fail);
   return total_fail ? 1 : 0;
}
