#pragma once
/* ML_DBG 在真机上受 ETNA_MESA_DEBUG=ml_msgs 控制；harness 里默认静默，
 * 需要时用 -DML_DBG_PRINT 打开（只用于看流程，不影响任何计算）。 */
#ifdef ML_DBG_PRINT
#include <stdio.h>
#define ML_DBG(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#else
#define ML_DBG(...) do { } while (0)
#endif
