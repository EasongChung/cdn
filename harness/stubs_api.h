#pragma once
/* stub 层对 harness 暴露的配置入口（不属于 Mesa API） */

struct etna_screen;

void stubs_set_case(const char *name);
void stubs_config(unsigned cores, unsigned zrl_bits, unsigned accum_depth, unsigned input_depth);
struct etna_screen *stubs_screen(void);
