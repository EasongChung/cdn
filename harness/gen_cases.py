#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 .tflite 生成 harness 的 case 表（**折叠后/变换后**的真实形状）。

变换规则逐字对齐 `etna_ml_lower_conv`（nn_core_version = 7，A311D）：

  · pointwise 且 input_channels == 1  →  pointwise_to_2x2：权重变 2×2
  · depthwise 且 output_channels > 1  →  expand_depthwise：**展成稠密卷积**
        （output_channels 不变，input_channels 仍为原 C ⇒ 权重 oc×C×kh×kw）
  · 其余不变

⇒ 写入器真正看到的 (oc, ic, kh, kw) 与 TFLite 里的形状**不同**，必须按上述规则换算，
   否则 harness 测的是不存在的形状。

用法：python3 gen_cases.py out.h <tag>=<model.tflite> [<tag>=<model.tflite> ...]
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "verify"))

import tflite_schema as ts  # noqa: E402


def opcode_name(m, idx):
    oc = m.operator_codes[idx]
    return getattr(oc, "name", None) or str(getattr(oc, "builtin_code", idx))


def ops_of(path: str):
    with open(path, "rb") as fh:
        m = ts.parse_model(fh.read(), path)
    sg = m.subgraphs[0]
    for op in sg.operators:
        if len(op.inputs) < 2:
            continue
        # ★ 必须按**算子类型**过滤：CONCATENATION / ADD 之类也有 4D 的第二个输入，
        #   把那种常量张量当权重会造出模型里根本不存在的形状（2026-10-22 修）。
        if opcode_name(m, op.opcode_index) not in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            continue
        wt = sg.tensors[op.inputs[1]]
        shape = tuple(getattr(wt, "shape", []) or [])
        if len(shape) == 4 and shape[0] == 1 and shape[3] > 1:
            kind = "dw"
            oc, kh, kw, c = shape[3], shape[1], shape[2], shape[3]
        elif len(shape) == 4:
            kind = "conv"
            oc, kh, kw, c = shape[0], shape[1], shape[2], shape[3]
        else:
            continue  # FC / 其它不测

        it = sg.tensors[op.inputs[0]]
        ot = sg.tensors[op.outputs[0]]
        in_shape = tuple(getattr(it, "shape", []) or [])
        out_shape = tuple(getattr(ot, "shape", []) or [])
        ic = in_shape[3] if len(in_shape) >= 4 else c
        ow = out_shape[1] if len(out_shape) >= 4 else 1
        oh = out_shape[2] if len(out_shape) >= 4 else 1

        # ---- 逐字对齐 lower_conv 的变换 ----
        pointwise = (kh == 1 and kw == 1 and kind == "conv")
        depthwise = kind == "dw"
        if pointwise and ic == 1:
            kh = kw = 2          # pointwise_to_2x2
        if depthwise and oc > 1:
            depthwise = False    # expand_depthwise ⇒ 稠密卷积
            ic = c               # input_channels 不变（仍为 C）
        elif depthwise:
            ic = c

        yield dict(name="%s:op%d:%dx%d_k%dx%d%s"
                        % ("", op.index, oc, ic, kh, kw, "dw" if depthwise else ""),
                   oc=oc, ic=ic, kh=kh, kw=kw, ow=ow, oh=oh,
                   depthwise=int(depthwise), pointwise=int(pointwise and not depthwise))


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    out = argv[1]
    rows, seen = [], set()
    for spec in argv[2:]:
        tag, path = spec.split("=", 1)
        for r in ops_of(os.path.join(REPO, path)):
            key = (r["oc"], r["ic"], r["kh"], r["kw"], r["depthwise"], r["pointwise"])
            if key in seen:
                continue
            seen.add(key)
            r["name"] = ("%s:" % tag) + r["name"].lstrip(":")
            rows.append(r)

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("/* 由 gen_cases.py 生成，勿手改 */\n")
        fh.write("struct testcase { const char *name; unsigned oc, ic, kh, kw, ow, oh, "
                 "stride, depthwise, pointwise; };\n")
        fh.write("static const struct testcase CASES[] = {\n")
        for r in rows:
            fh.write('   {"%s", %d, %d, %d, %d, %d, %d, 1, %d, %d},\n'
                     % (r["name"], r["oc"], r["ic"], r["kh"], r["kw"],
                        r["ow"], r["oh"], r["depthwise"], r["pointwise"]))
        fh.write("};\n")
    print("写入 %s：%d 个去重后的形状" % (out, len(rows)))
    for r in rows:
        print("   %-42s oc=%-4d ic=%-4d k=%dx%d dw=%d pw=%d"
              % (r["name"], r["oc"], r["ic"], r["kh"], r["kw"], r["depthwise"], r["pointwise"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
