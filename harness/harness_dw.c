/* depthwise × per-channel 折叠后的**后态**探测（真实源码 + 守页）
 *
 * 复现链条（全部来自未改写源码，行号见 verify/reports/54）：
 *
 *   1. lower_convolution()：operation->input_channels = 输入张量 dims[3] = C
 *   2. expand_depthwise()：权重展成 C×k×k×C，**input_channels 不变**
 *   3. transpose()：同上，尺寸 C·k²·C
 *   4. per-channel 分支：
 *        calc_quantization()   —— 内部强制 input_channels = 1 来遍历（稀疏采样）
 *        requantize_weights()  —— 强制 input_channels = 1，new_size = C·k²，
 *                                 free 掉展开后的缓冲区 ⇒ **缓冲区缩了 C 倍**，
 *                                 而 operation->input_channels 仍是 C
 *   5. create_coeffs_v7() 的写入器用 operation->input_channels(=C) 索引权重，
 *      即按 C·k²·C 读取一个只有 C·k² 字节的缓冲区 ⇒ **越界 C 倍**
 *
 * 本程序：
 *   · 前态用 malloc（因为 requantize_weights 内部会 free 它）
 *   · 调用**真实的** requantize_weights（extract_funcs.py 从源码提取，body md5 绑定）
 *   · 后态按驱动给出的 weight_tensor_size 复制进守页区 ⇒ 写入器一越界就 SIGSEGV
 *   · 对照组：同形状但非 depthwise（ic 不缩） / depthwise 但 per-tensor（不折叠）
 */

#include "etnaviv_context.h"
#include "etnaviv_screen.h"
#include "etnaviv_ml_nn.h"
#include "pipe_ml.h"
#include "guard.h"
#include "stubs_api.h"
#include "cases_dw.h"

#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 由 extract_funcs.py 生成 */
void requantize_weights(struct etna_ml_subgraph *subgraph,
                        const struct pipe_ml_operation *poperation,
                        struct etna_operation *operation);

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

/* 返回：0 = 干净；1 = 越界/异常；-1 = 准备失败 */
static int
run_case(const struct dwcase *c, const char *hwtag, int verbose)
{
   unsigned C = c->oc, k = c->k;
   size_t pre_size = (size_t)C * k * k * C; /* expand_depthwise + transpose 之后 */
   size_t bsize = (size_t)C * sizeof(int32_t);
   size_t post_size = 0;   /* 折叠后驱动报告的 weight_tensor_size */
   unsigned ic_after = 0;  /* 折叠后驱动留下的 input_channels */

   struct etna_operation op;
   memset(&op, 0, sizeof(op));
   op.type = ETNA_JOB_TYPE_NN;
   op.depthwise = c->depthwise;
   op.pointwise = 0;
   op.stride = c->stride ? c->stride : 1;
   op.input_width = op.output_width = c->out; /* 只影响 tile 数值，不影响越界 */
   op.input_height = op.output_height = c->out;
   op.input_channels = C; /* ★ 关键：展开后仍是 C（源码不改回） */
   op.input_zero_point = 128;
   op.output_channels = C;
   op.output_zero_point = 128;
   op.weight_width = k;
   op.weight_height = k;
   op.weight_zero_point = 0;
   op.weight_scale = 1.0f / 255.0f;
   op.weight_signed = true;

   /* 前态：requantize_weights 会 free 它 ⇒ 必须用 malloc */
   uint8_t *pre = malloc(pre_size);
   uint8_t *b = guard_alloc(bsize, "biases");
   if (!pre || !b) {
      free(pre);
      return -1;
   }
   for (size_t i = 0; i < pre_size; i++)
      pre[i] = (uint8_t)((i * 37u + 11u) & 0xFFu);
   for (size_t i = 0; i < bsize; i++)
      b[i] = (uint8_t)((i * 91u + 7u) & 0xFFu);

   op.weight_tensor = pre;
   op.weight_tensor_size = (unsigned)pre_size;
   op.bias_tensor = b;
   op.bias_tensor_size = (unsigned)bsize;

   const char *pcs = c->per_channel ? "per-channel" : "per-tensor";
   printf("  case %-34s C=%-4u k=%u dw=%d %s\n", c->name, C, k, c->depthwise, pcs);
   if (verbose)
      printf("       前态权重缓冲 = C·k²·C = %zu B\n", pre_size);

   if (c->per_channel) {
      /* per-channel 量化参数：只用于让真实 requantize_weights 走完整路径 */
      float *scales = malloc((size_t)C * sizeof(float));
      uint8_t *zps = malloc((size_t)C);
      if (!scales || !zps) {
         free(scales);
         free(zps);
         free(pre);
         return -1;
      }
      for (unsigned i = 0; i < C; i++) {
         scales[i] = 0.001f + 0.0001f * (float)(i % 97);
         zps[i] = (uint8_t)(i & 0x7F);
      }
      struct pipe_tensor wt;
      memset(&wt, 0, sizeof(wt));
      wt.scales = scales;
      wt.zero_points = zps;
      wt.is_signed = true;

      struct pipe_ml_operation pop;
      memset(&pop, 0, sizeof(pop));
      pop.conv.weight_tensor = &wt;
      pop.conv.depthwise = c->depthwise;

      struct etna_ml_subgraph sg = { .screen = stubs_screen() };
      requantize_weights(&sg, &pop, &op); /* ★ 真实源码 */
      free(scales);
      free(zps);

      post_size = op.weight_tensor_size;
      ic_after = op.input_channels;
      /* 写入器总读取量 = C(输出通道) × in_c × k² ；缓冲区 = weight_tensor_size */
      unsigned long need = (unsigned long)C * op.input_channels * k * k;
      printf("       后态：weight_tensor_size=%u  input_channels=%u  写入器将读 %lu B = %.1f× 缓冲区  %s\n",
             op.weight_tensor_size, op.input_channels, need,
             op.weight_tensor_size ? (double)need / (double)op.weight_tensor_size : 0.0,
             (need > op.weight_tensor_size) ? "★ 必然越界" : "在界内");

      /* 后态按驱动给出的尺寸复制进守页区 */
      size_t post = op.weight_tensor_size;
      uint8_t *w2 = guard_alloc(post ? post : 1, "post-requant");
      if (!w2) {
         free(op.weight_tensor);
         return -1;
      }
      memcpy(w2, op.weight_tensor, post);
      free(op.weight_tensor); /* 真实函数 malloc 的 */
      op.weight_tensor = w2;
   } else {
      /* per-tensor：不做折叠，缓冲区就是展开后的 C·k²·C —— 守页直接守着它 */
      size_t post = pre_size;
      uint8_t *w2 = guard_alloc(post, "expanded");
      if (!w2) {
         free(pre);
         return -1;
      }
      memcpy(w2, pre, post);
      free(pre);
      op.weight_tensor = w2;
      op.weight_tensor_size = (unsigned)post;
      post_size = post;
      ic_after = op.input_channels;
   }

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
         printf("       ok   cache=%u\n", cache_size);
      printf("       RESULT %s dw=%u pc=%u pre=%zu post=%zu ic=%u verdict=OK\n",
             c->name, c->depthwise, c->per_channel, pre_size, post_size, ic_after);
      return 0;
   }
   g_in_case = 0;

   const void *gptr = NULL;
   size_t gsize = 0;
   guard_info((const void *)g_fault_addr, &gptr, &gsize);
   const char *tag = guard_tag((const void *)g_fault_addr);
   long off = (gptr && gsize) ? (long)((const char *)g_fault_addr - (const char *)gptr) : -1;

   printf("       FAIL sig=%d 命中缓冲区=%s size=%zu 越界偏移=%ld (=%s)\n",
          g_fault_sig, tag ? tag : "(非守护页)", gsize, off,
          (off >= 0 && (size_t)off == gsize) ? "恰好在末尾" : "非末尾");
   printf("       RESULT %s dw=%u pc=%u pre=%zu post=%zu ic=%u verdict=OOB off=%ld size=%zu %s\n",
          c->name, c->depthwise, c->per_channel, pre_size, post_size, ic_after,
          off, gsize, (off >= 0 && (size_t)off == gsize) ? "at_end" : "not_end");
   return 1;
}

int
main(int argc, char **argv)
{
   int verbose = 0;
   for (int i = 1; i < argc; i++)
      if (!strcmp(argv[i], "-v"))
         verbose = 1;

   struct sigaction sa;
   memset(&sa, 0, sizeof(sa));
   sa.sa_sigaction = fault_handler;
   sa.sa_flags = SA_SIGINFO | SA_NODEFER;
#if !defined(NO_FAULT_HANDLER)
   sigaction(SIGSEGV, &sa, NULL);
   sigaction(SIGBUS, &sa, NULL);
   sigaction(SIGFPE, &sa, NULL);
#endif

   /* 与主 harness 相同的 8 核配置（A311D 实测 nn_core_count=8） */
   struct {
      unsigned cores, zrl, accum, inbuf;
   } cfgs[] = {{8, 4, 4096, 3072}, {4, 4, 4096, 3072}};

   printf("== depthwise × per-channel 后态探测（真实 requantize_weights + 真实写入器 + 守页）==\n");
   printf("   case 数=%zu  hw 配置数=%zu\n", sizeof(DW_CASES) / sizeof(DW_CASES[0]),
          sizeof(cfgs) / sizeof(cfgs[0]));

   int total = 0;
   for (size_t k = 0; k < sizeof(cfgs) / sizeof(cfgs[0]); k++) {
      char hwtag[64];
      snprintf(hwtag, sizeof(hwtag), "cores=%u,zrl=%u,acc=%u,in=%u",
               cfgs[k].cores, cfgs[k].zrl, cfgs[k].accum, cfgs[k].inbuf);
      stubs_config(cfgs[k].cores, cfgs[k].zrl, cfgs[k].accum, cfgs[k].inbuf);
      printf("\n-- dw hw %s --\n", hwtag);
      for (size_t i = 0; i < sizeof(DW_CASES) / sizeof(DW_CASES[0]); i++) {
         int r = run_case(&DW_CASES[i], hwtag, verbose);
         if (r != 0)
            total++;
      }
   }

   printf("\n== 总计：%d 次越界/异常 ==\n", total);
   return total ? 1 : 0;
}
