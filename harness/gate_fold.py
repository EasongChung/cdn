#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V4a 判定：fold-early 守页验证。

**判据不是「全干净」**，而是「实测必须逐字等于按缺陷模型算出的预期」。
本门同时区分**两个互不相同的缺陷**，混在一起判会得出错误结论：

  缺陷 A（fold 缺陷，报告 54）
      上游在 layout 变换**之后**折叠，depthwise 强压 in_c=1 并缩容 C 倍
      ⇒ 写入器按 C·k²·C 读一个 C·k² 的缓冲 ⇒ 越界 C 倍。
      本门用 `mode=up` 作**负控制**：必须越界，且 off == size == C·k²。

  缺陷 B（幻影通道，报告 53）
      写入器把 out_c 分摊到 cores_used 个核，`out_c % cores_used != 0` 时
      最后一个核写到描述符之外 ⇒ 越界。**与折叠无关**，
      现网由 TEFLON_PHANTOM_GATE 拦（该形状回落 CPU）。
      本门用 `mode=fe` 的预期表达它：`C % min(C,cores) == 0` 才期望干净。

⇒ fold-early 正确的判据是：
      mode=fe 的实测 == 「幻影模型」的预测（逐字相等，含 OK 与 OOB 两种）
   且 mode=up 的实测 == 「fold 缺陷模型」的预测（逐字相等）
   且 mode=i2 的两条管线逐字节相同（不变式 I2）

若哪天 fe 的 OOB 也消失了（例如幻影通道被修好），本门会红 —— 那是**升级信号**，
应当把预期改成「全干净」，而不是删掉本门。
"""

from __future__ import annotations

import re
import sys

RESULT = re.compile(
    r"RESULT (\S+) mode=(\w+) "
    r"(?:(?:sz=(\d+) expect=(\d+) ic=(\d+) verdict=OK)"
    r"|(?:sz=(\d+) expect=(\d+) ic=(\d+) verdict=OOB off=(-?\d+) size=(\d+) (\w+)))"
)
RESULT_I2 = re.compile(
    r"RESULT (\S+) mode=i2 n=(\d+) szA=(\d+) szB=(\d+) scale_eq=(\d) zp_eq=(\d) equal=(\d) first_diff=(-?\d+)"
)
HW = re.compile(r"^-- fold hw cores=(\d+),")

FAILS: list[str] = []
CHECKS = 0


def chk(cond, name, detail=""):
    global CHECKS
    CHECKS += 1
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s   %s" % (name, detail))
        FAILS.append(name)


def phantom_expected(C: int, cores: int) -> str:
    """幻影通道模型：cores_used = min(C, cores)；C % cores_used == 0 ⇒ 安全。"""
    cores_used = min(C, cores)
    return "OK" if C % cores_used == 0 else "OOB"


def main(argv):
    if len(argv) < 2:
        print("用法: gate_fold.py <harness 输出>", file=sys.stderr)
        return 2
    path = argv[1]

    cores = None
    fe: dict[tuple[int, str], dict] = {}
    up: dict[tuple[int, str], dict] = {}
    i2: dict[tuple[int, str], dict] = {}

    for line in open(path, encoding="utf-8", errors="replace"):
        m = HW.match(line)
        if m:
            cores = int(m.group(1))
            continue
        m = RESULT_I2.search(line)
        if m and cores is not None:
            i2[(cores, m.group(1))] = dict(n=int(m.group(2)), szA=int(m.group(3)),
                                           szB=int(m.group(4)), scale_eq=int(m.group(5)),
                                           zp_eq=int(m.group(6)), equal=int(m.group(7)),
                                           first=int(m.group(8)))
            continue
        m = RESULT.search(line)
        if not m or cores is None:
            continue
        name, mode = m.group(1), m.group(2)
        is_oob = m.group(6) is not None
        if mode == "fe":
            if is_oob:
                fe[(cores, name)] = dict(sz=int(m.group(6)), expect=int(m.group(7)),
                                         ic=int(m.group(8)), verdict="OOB")
            else:
                fe[(cores, name)] = dict(sz=int(m.group(3)), expect=int(m.group(4)),
                                         ic=int(m.group(5)), verdict="OK")
        elif mode == "up":
            if not is_oob:
                chk(False, "up %s 缺少 OOB 判决（解析到 OK 行）" % name)
                continue
            up[(cores, name)] = dict(sz=int(m.group(6)), expect=int(m.group(7)),
                                     ic=int(m.group(8)), verdict="OOB", off=int(m.group(9)),
                                     size=int(m.group(10)), at_end=(m.group(11) == "at_end"))

    # ── 1. mode=fe：必须逐字等于幻影模型预测 ──────────────────────────────────
    print("\n[1] mode=fe（fold-early 管线）vs 幻影通道模型预测")
    chk(len(fe) >= 14, "fe 结果数 ≥ 14（7 形状 × 2 配置）", "得到 %d" % len(fe))
    ok_cnt = oob_cnt = 0
    for (c, name), e in sorted(fe.items()):
        C = int(re.search(r"C(\d+)_", name).group(1))
        want = phantom_expected(C, c)
        chk(e["verdict"] == want, "fe %-30s cores=%d 预测=%s 实测=%s" % (name, c, want, e["verdict"]))
        chk(e["sz"] == e["expect"], "fe %-30s 尺寸自洽 sz=%d expect=%d" % (name, e["sz"], e["expect"]))
        chk(e["ic"] == C, "fe %-30s input_channels 保持 C（=%d）" % (name, C), "得到 %d" % e["ic"])
        ok_cnt += e["verdict"] == "OK"
        oob_cnt += e["verdict"] == "OOB"
    # 反作弊：两种判决都必须出现，否则本门没有分辨力
    chk(ok_cnt > 0 and oob_cnt > 0,
        "反作弊：fe 里同时出现 OK 与 OOB（否则门无分辨力）",
        "OK=%d OOB=%d" % (ok_cnt, oob_cnt))

    # ── 2. mode=up：fold 缺陷必须复现（负控制）───────────────────────────────
    print("\n[2] mode=up（上游管线）必须复现 fold 缺陷")
    chk(len(up) >= 8, "up 结果数 ≥ 8（4 个 depthwise×per-channel × 2 配置）", "得到 %d" % len(up))
    for (c, name), e in sorted(up.items()):
        C = int(re.search(r"C(\d+)_", name).group(1))
        k = int(re.search(r"_k(\d+)", name).group(1))
        shrink = C * k * k
        chk(e["verdict"] == "OOB", "up %-30s cores=%d 必须越界" % (name, c), "得到 %s" % e["verdict"])
        chk(e["sz"] == shrink, "up %-30s 缓冲区被缩成 C·k²=%d" % (name, shrink), "得到 %d" % e["sz"])
        chk(e["at_end"] and e["off"] == e["size"] == shrink,
            "up %-30s 越界偏移恰在末尾（off=%d size=%d）" % (name, e["off"], e["size"]))

    # ── 2b. 跨模式：同一形状下 fe 与 up 的缓冲尺寸必须不同（证明折叠没缩容）──
    print("\n[2b] 跨模式：fe 的缓冲必须是 up 的 C 倍")
    for key in sorted(set(fe) & set(up)):
        c, name = key
        C = int(re.search(r"C(\d+)_", name).group(1))
        chk(fe[key]["sz"] == up[key]["sz"] * C,
            "跨模式 %-30s cores=%d fe=%d == up=%d × C(%d)" % (name, c, fe[key]["sz"], up[key]["sz"], C))

    # ── 3. mode=i2：非 depthwise 两条管线逐字节相同 ──────────────────────────
    print("\n[3] mode=i2（非 depthwise）两条管线必须逐字节相同")
    chk(len(i2) >= 2, "i2 结果数 ≥ 2", "得到 %d" % len(i2))
    for (c, name), e in sorted(i2.items()):
        chk(e["equal"] == 1, "i2 %-30s cores=%d 缓冲逐字节相同" % (name, c),
            "first_diff=%d" % e["first"])
        chk(e["scale_eq"] == 1 and e["zp_eq"] == 1, "i2 %-30s 折叠参数也相同" % name)
        chk(e["szA"] == e["szB"], "i2 %-30s 尺寸相同" % name,
            "A=%d B=%d" % (e["szA"], e["szB"]))

    print("\n== %d 项检查，%d 项失败 ==" % (CHECKS, len(FAILS)))
    for f in FAILS:
        print("  FAILED: %s" % f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
