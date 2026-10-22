/* depthwise 后态探测的 case 表。
 *
 * 形状取自 **挂死样本模型** `verify/artifacts/local-surgery/legacy-clean-nodepad.tflite`
 * 的真实 depthwise 层（17 层全部 per-channel）：
 *
 *   op#7    C=24   3×3  出 50×50   ⇒ %8 门 **放行**
 *   op#10   C=122  3×3  出 50×50   ⇒ %8 门 拦
 *   op#61   C=244  3×3  出 25×25   ⇒ %8 门 拦
 *   op#166  C=488  3×3  出 13×13   ⇒ %8 门 **放行**（op#181/194/207 同形）
 *
 * 前缀约定（gate_dw.py 依赖）：
 *   dw:  期望 **越界**（depthwise + per-channel，缓冲区被折叠缩了 C 倍）
 *   ctl: 期望 **干净**（负控制：同形状但非 depthwise / 或 per-tensor 不折叠）
 */
struct dwcase {
   const char *name;
   unsigned oc;   /* 通道数 C（depthwise 时 ic == oc） */
   unsigned k;    /* 卷积核边长 */
   unsigned out;  /* 输出高/宽（只影响 tiling 数值） */
   unsigned stride;
   unsigned depthwise;
   unsigned per_channel; /* 1 = 走 calc_quantization/requantize_weights */
};

static const struct dwcase DW_CASES[] = {
    /* --- 真实模型的 depthwise 层（per-channel） --- */
    {"dw:op7:C24_k3",      24,  3, 50, 1, 1, 1},
    {"dw:op10:C122_k3",    122, 3, 50, 1, 1, 1},
    {"dw:op61:C244_k3",    244, 3, 25, 1, 1, 1},
    {"dw:op166:C488_k3",   488, 3, 13, 1, 1, 1},

    /* --- 负控制 1：同形状但**非 depthwise** ⇒ requantize 用 ic=C，缓冲区不变 --- */
    {"ctl:C488_k3_nonDW_pc", 488, 3, 13, 1, 0, 1},

    /* --- 负控制 2：depthwise 但 **per-tensor** ⇒ 不折叠，缓冲区是展开后的 C·k²·C --- */
    {"ctl:C488_k3_dw_pertensor", 488, 3, 13, 1, 1, 0},
    {"ctl:C24_k3_dw_pertensor",  24,  3, 50, 1, 1, 0},
};
