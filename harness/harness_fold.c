/* V4a：per-channel **折叠提前**（fold-early）的守页验证 —— 正控制 + 负控制
 *
 * 被测对象全部是**真实源码函数体**（extract_funcs.py 逐字节提取）：
 *
 *   打补丁后的 nn.c  ⇒ och_fold_channel / calc_quantization / requantize_weights
 *                       / expand_depthwise / transpose
 *   未打补丁的 nn.c  ⇒ upstream_calc_quantization / upstream_requantize_weights（对照）
 *
 * 三种模式（同一形状表 cases_dw.h）：
 *
 *   mode=fe  折叠提前管线： fold(原地) → expand → transpose → 写入器
 *            期望：缓冲区 = C·k²·C、input_channels = C、写入器**干净**
 *
 *   mode=up  上游管线：      expand → transpose → upstream fold（缩容 C 倍）→ 写入器
 *            期望：**越界**，且越界偏移恰好在缓冲区末尾（= 报告 54 的结论，此处作负控制）
 *
 *   mode=i2  非 depthwise 的逐字节等价：同一份原始权重分别走两条管线，
 *            比较**最终（transpose 之后）**缓冲 —— 必须 memcmp == 0（不变式 I2）
 *
 * 判据见 gate_fold.py：不是「全干净」，而是「实测必须逐字等于预期」。
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

/* ── 来自**打补丁后**的 nn.c ─────────────────────────────────────────────── */
unsigned och_fold_channel(const struct etna_operation *operation, unsigned i, bool depthwise);
void calc_quantization(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation,
                       struct etna_operation *operation, float *out_scale, uint8_t *out_zero_point);
void requantize_weights(struct etna_ml_subgraph *subgraph,
                        const struct pipe_ml_operation *poperation,
                        struct etna_operation *operation);
/* ── 上两项依赖的、保持不变的公共件 ──────────────────────────────────────── */
void calc_quant_params(float min, float max, float *out_scale, uint8_t *out_zero_point);
void expand_depthwise(struct etna_ml_subgraph *subgraph, struct etna_operation *operation);
void transpose(struct etna_ml_subgraph *subgraph, struct etna_operation *operation);
/* ── 来自**未打补丁**的 nn.c（对照）────────────────────────────────────── */
void upstream_calc_quantization(struct etna_ml_subgraph *subgraph,
                                const struct pipe_ml_operation *poperation,
                                struct etna_operation *operation,
                                float *out_scale, uint8_t *out_zero_point);
void upstream_requantize_weights(struct etna_ml_subgraph *subgraph,
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

static struct etna_ml_subgraph *
mk_sg(void)
{
   static struct etna_ml_subgraph sg;
   sg.screen = stubs_screen();
   return &sg;
}

/* 填一份「原始 TFLite 布局」的权重缓冲：[1][kh][kw][C]（depthwise）或 [oc][kh][kw][ic] */
static uint8_t *
mk_weights(size_t n)
{
   uint8_t *p = malloc(n ? n : 1);
   for (size_t i = 0; i < n; i++)
      p[i] = (uint8_t)((i * 37u + 11u) & 0xFFu);
   return p;
}

/* 分配 bias（写入器会写 biases[out_channel]；上游在 lower_convolution 里 memcpy 过一份）。
 * 用守页分配器 ⇒ bias 越界也会被抓到。*/
static void
mk_bias(struct etna_operation *op, unsigned C)
{
   size_t n = (size_t)C * sizeof(int32_t);
   uint8_t *b = guard_alloc(n, "biases");
   memset(b, 0, n);
   op->bias_tensor = b;
   op->bias_tensor_size = (unsigned)n;
}

/* 组装 op（与真实 lower_convolution 的前态一致） */
static void
mk_op(struct etna_operation *op, const struct dwcase *c)
{
   unsigned C = c->oc, k = c->k;
   memset(op, 0, sizeof(*op));
   op->type = ETNA_JOB_TYPE_NN;
   op->depthwise = c->depthwise;
   op->stride = c->stride ? c->stride : 1;
   op->input_width = op->output_width = c->out;
   op->input_height = op->output_height = c->out;
   op->input_channels = C;        /* ★ 展开后仍是 C（源码不改回） */
   op->input_zero_point = 128;
   op->output_channels = C;
   op->output_zero_point = 128;
   op->weight_width = k;
   op->weight_height = k;
   op->weight_zero_point = 0;
   op->weight_scale = 1.0f / 255.0f;
   op->weight_signed = true;
   mk_bias(op, C);
}

/* 造 per-channel 量化参数（真实 requantize 走完整路径所必需） */
static void
mk_pc(const struct dwcase *c, struct pipe_tensor *wt, float **scales_out, uint8_t **zps_out)
{
   unsigned C = c->oc;
   float *scales = malloc((size_t)C * sizeof(float));
   uint8_t *zps = malloc((size_t)C);
   for (unsigned i = 0; i < C; i++) {
      scales[i] = 0.001f + 0.0001f * (float)(i % 97);
      zps[i] = (uint8_t)(i & 0x7F);
   }
   memset(wt, 0, sizeof(*wt));
   wt->scales = scales;
   wt->zero_points = zps;
   wt->is_signed = true;
   *scales_out = scales;
   *zps_out = zps;
}

/* 把最终权重搬进守页区并跑写入器；返回 0=干净 1=越界/异常 */
static int
run_writer(const struct dwcase *c, struct etna_operation *op, const char *mode,
           unsigned long expect_bytes)
{
   size_t sz = op->weight_tensor_size;
   uint8_t *guarded = guard_alloc(sz ? sz : 1, "final-weights");
   if (!guarded)
      return -1;
   memcpy(guarded, op->weight_tensor, sz);
   op->weight_tensor = guarded;

   stubs_set_case(c->name);
   unsigned cache_size = 0;
   g_fault_sig = 0;
   g_fault_addr = 0;
   g_in_case = 1;
   long off = -1;
   size_t gsize = 0;
   const char *tag = NULL;

   if (sigsetjmp(g_jmp, 1) == 0) {
      etna_ml_create_coeffs_v7(mk_sg(), op, &cache_size);
      g_in_case = 0;
      printf("       RESULT %s mode=%s sz=%zu expect=%lu ic=%u verdict=OK cache=%u\n",
             c->name, mode, sz, expect_bytes, op->input_channels, cache_size);
      return 0;
   }
   g_in_case = 0;

   const void *gptr = NULL;
   guard_info((const void *)g_fault_addr, &gptr, &gsize);
   tag = guard_tag((const void *)g_fault_addr);
   off = (gptr && gsize) ? (long)((const char *)g_fault_addr - (const char *)gptr) : -1;
   printf("       RESULT %s mode=%s sz=%zu expect=%lu ic=%u verdict=OOB off=%ld size=%zu %s sig=%d\n",
          c->name, mode, sz, expect_bytes, op->input_channels, off, gsize,
          (off >= 0 && (size_t)off == gsize) ? "at_end" : "not_end", g_fault_sig);
   (void)tag;
   return 1;
}

/* mode=fe：折叠提前管线 */
static void
case_foldearly(const struct dwcase *c)
{
   const unsigned C = c->oc, k = c->k;
   const size_t orig = (size_t)C * k * k * (c->depthwise ? 1 : C); /* depthwise: [1][k][k][C] */
   struct etna_operation op;
   mk_op(&op, c);

   uint8_t *w = mk_weights(orig);
   op.weight_tensor = w;
   op.weight_tensor_size = (unsigned)orig;

   printf("  case %-28s mode=fe   C=%-4u k=%u dw=%d pc=%d\n",
          c->name, C, k, c->depthwise, c->per_channel);

   if (c->per_channel) {
      float *scales, *zs_unused = NULL;
      uint8_t *zps;
      struct pipe_tensor wt;
      mk_pc(c, &wt, &scales, &zps);
      struct pipe_ml_operation pop;
      memset(&pop, 0, sizeof(pop));
      pop.conv.weight_tensor = &wt;
      pop.conv.depthwise = c->depthwise;

      float scale = 0;
      uint8_t zp = 0;
      calc_quantization(mk_sg(), &pop, &op, &scale, &zp);
      op.weight_scale = scale;
      op.weight_zero_point = zp;
      requantize_weights(mk_sg(), &pop, &op);

      if (op.weight_tensor_size != orig) {
         printf("       FAIL I1：折叠后尺寸 %u != 原尺寸 %zu（必须原地）\n",
                op.weight_tensor_size, orig);
         free(scales);
         free(zps);
         return;
      }
      free(scales);
      free(zps);
      (void)zs_unused;
   }

   /* 布局变换链（与 lower_convolution 的顺序一致） */
   if (c->depthwise) {
      expand_depthwise(mk_sg(), &op);
   }
   if (op.input_channels > 1)
      transpose(mk_sg(), &op);

   size_t expect = (size_t)op.output_channels * op.weight_width * op.weight_height * op.input_channels;
   if (op.weight_tensor_size != expect) {
      printf("       FAIL 尺寸自洽：weight_tensor_size=%u 但写入器需要 %zu\n",
             op.weight_tensor_size, expect);
      return;
   }
   run_writer(c, &op, "fe", (unsigned long)expect);
}

/* mode=up：上游管线（负控制） */
static void
case_upstream(const struct dwcase *c)
{
   const unsigned C = c->oc, k = c->k;
   const size_t orig = (size_t)C * k * k;
   struct etna_operation op;
   mk_op(&op, c);
   op.weight_tensor = mk_weights(orig);
   op.weight_tensor_size = (unsigned)orig;

   printf("  case %-28s mode=up   C=%-4u k=%u dw=%d\n", c->name, C, k, c->depthwise);

   /* ★ 上游真实顺序：expand → transpose → 折叠（折叠在**布局变换之后**，这正是缺陷所在）。
    * 顺序写反会让 expand 去读已被缩容的缓冲区 ⇒ 变成另一个 bug，测的就不是上游了。 */
   if (c->depthwise)
      expand_depthwise(mk_sg(), &op);
   if (op.input_channels > 1)
      transpose(mk_sg(), &op);

   if (c->per_channel) {
      float *scales;
      uint8_t *zps;
      struct pipe_tensor wt;
      mk_pc(c, &wt, &scales, &zps);
      struct pipe_ml_operation pop;
      memset(&pop, 0, sizeof(pop));
      pop.conv.weight_tensor = &wt;
      pop.conv.depthwise = c->depthwise;

      float scale = 0;
      uint8_t zp = 0;
      upstream_calc_quantization(mk_sg(), &pop, &op, &scale, &zp);
      op.weight_scale = scale;
      op.weight_zero_point = zp;
      upstream_requantize_weights(mk_sg(), &pop, &op);
      free(scales);
      free(zps);
   }

   size_t expect = (size_t)op.output_channels * op.weight_width * op.weight_height * op.input_channels;
   run_writer(c, &op, "up", (unsigned long)expect);
}

/* mode=i2：非 depthwise 的两条管线必须逐字节相同（I2） */
static void
case_i2(const struct dwcase *c)
{
   const unsigned C = c->oc, k = c->k;
   const size_t orig = (size_t)C * k * k * C;
   struct etna_operation a, b;
   mk_op(&a, c);
   mk_op(&b, c);

   uint8_t *w = mk_weights(orig);
   uint8_t *wa = malloc(orig), *wb = malloc(orig);
   memcpy(wa, w, orig);
   memcpy(wb, w, orig);

   float *scales;
   uint8_t *zps;
   struct pipe_tensor wt;
   mk_pc(c, &wt, &scales, &zps);
   struct pipe_ml_operation pop;
   memset(&pop, 0, sizeof(pop));
   pop.conv.weight_tensor = &wt;
   pop.conv.depthwise = 0;

   /* A：上游 —— 先 transpose 再折叠 */
   a.weight_tensor = wa;
   a.weight_tensor_size = (unsigned)orig;
   transpose(mk_sg(), &a);
   float sA = 0;
   uint8_t zA = 0;
   upstream_calc_quantization(mk_sg(), &pop, &a, &sA, &zA);
   a.weight_scale = sA;
   a.weight_zero_point = zA;
   upstream_requantize_weights(mk_sg(), &pop, &a);

   /* B：fold-early —— 先折叠再 transpose */
   b.weight_tensor = wb;
   b.weight_tensor_size = (unsigned)orig;
   float sB = 0;
   uint8_t zB = 0;
   calc_quantization(mk_sg(), &pop, &b, &sB, &zB);
   b.weight_scale = sB;
   b.weight_zero_point = zB;
   requantize_weights(mk_sg(), &pop, &b);
   transpose(mk_sg(), &b);

   unsigned n = a.weight_tensor_size < b.weight_tensor_size ? a.weight_tensor_size : b.weight_tensor_size;
   int equal = (a.weight_tensor_size == b.weight_tensor_size) &&
               (memcmp(a.weight_tensor, b.weight_tensor, n) == 0) &&
               (sA == sB) && (zA == zB);
   long first = -1;
   if (!equal && a.weight_tensor_size == b.weight_tensor_size) {
      for (unsigned i = 0; i < n; i++)
         if (((uint8_t *)a.weight_tensor)[i] != ((uint8_t *)b.weight_tensor)[i]) {
            first = i;
            break;
         }
   }
   printf("       RESULT %s mode=i2 n=%u szA=%u szB=%u scale_eq=%d zp_eq=%d equal=%d first_diff=%ld\n",
          c->name, n, a.weight_tensor_size, b.weight_tensor_size,
          sA == sB, zA == zB, equal, first);

   free(scales);
   free(zps);
   free(w);
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

   struct {
      unsigned cores, zrl, accum, inbuf;
   } cfgs[] = {{8, 4, 4096, 3072}, {4, 4, 4096, 3072}};

   printf("== fold-early 守页验证（真实源码 + 真实写入器）==\n");
   printf("   case 数=%zu  hw 配置数=%zu\n", sizeof(DW_CASES) / sizeof(DW_CASES[0]),
          sizeof(cfgs) / sizeof(cfgs[0]));

   for (size_t ci = 0; ci < sizeof(cfgs) / sizeof(cfgs[0]); ci++) {
      char hwtag[64];
      snprintf(hwtag, sizeof(hwtag), "cores=%u,zrl=%u,acc=%u,in=%u",
               cfgs[ci].cores, cfgs[ci].zrl, cfgs[ci].accum, cfgs[ci].inbuf);
      stubs_config(cfgs[ci].cores, cfgs[ci].zrl, cfgs[ci].accum, cfgs[ci].inbuf);
      printf("\n-- fold hw %s --\n", hwtag);

      for (size_t i = 0; i < sizeof(DW_CASES) / sizeof(DW_CASES[0]); i++) {
         case_foldearly(&DW_CASES[i]);
         if (verbose)
            printf("       (verbose) fe 完成\n");
      }
      for (size_t i = 0; i < sizeof(DW_CASES) / sizeof(DW_CASES[0]); i++) {
         if (!DW_CASES[i].depthwise || !DW_CASES[i].per_channel)
            continue;
         case_upstream(&DW_CASES[i]);
      }
      for (size_t i = 0; i < sizeof(DW_CASES) / sizeof(DW_CASES[0]); i++) {
         if (DW_CASES[i].depthwise || !DW_CASES[i].per_channel)
            continue;
         case_i2(&DW_CASES[i]);
      }
   }
   return 0;
}
