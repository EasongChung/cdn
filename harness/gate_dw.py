#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""depthwise × per-channel 后态探测的判定。

**判据不是「全部干净」** —— 未打布局补丁的上游源码**必然**越界，那正是要证明的事实。
判据是「实测模式必须逐字等于预期模式」：

  1. `dw:`  案（depthwise + per-channel）⇒ 必须 OOB，且越界偏移恰好在缓冲区末尾
     （off == size == post），并且 post == oc·k²（= 被 requantize_weights 缩掉 C 倍）
  2. `ctl:` 案（同形状但非 depthwise / 或 per-tensor 不折叠）⇒ 必须干净
  3. 两类都必须非空（反作弊：否则什么都没测）

若上游哪天修好了这条（例如把 requantize 移到 expand_depthwise 之前），
判据 1 会失败 —— 届时这是**升级信号**，应改为「dw 案也应干净」而不是删掉本门。
"""

from __future__ import annotations

import re
import sys


def parse(path: str):
    blocks: dict[str, dict[str, dict]] = {}
    cur = None
    pending = None  # (name, oc, k)
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^-- dw hw (.+) --\s*$", line)
        if m:
            cur = m.group(1)
            blocks[cur] = {}
            continue
        m = re.match(r"^\s+case (\S+)\s+C=(\d+)\s+k=(\d+)\s+dw=(\d+)\s+(\S+)", line)
        if m and cur is not None:
            pending = (m.group(1), int(m.group(2)), int(m.group(3)))
            blocks[cur].setdefault(m.group(1), {"oc": int(m.group(2)), "k": int(m.group(3))})
            continue
        m = re.search(r"RESULT (\S+) dw=(\d+) pc=(\d+) pre=(\d+) post=(\d+) ic=(\d+) verdict=(\w+)"
                      r"(?: off=(-?\d+) size=(\d+) (\w+))?", line)
        if m and cur is not None:
            name = m.group(1)
            e = blocks[cur].setdefault(name, {})
            e.update(dict(
                dw=int(m.group(2)), pc=int(m.group(3)), pre=int(m.group(4)),
                post=int(m.group(5)), ic=int(m.group(6)), verdict=m.group(7),
                off=int(m.group(8)) if m.group(8) is not None else None,
                size=int(m.group(9)) if m.group(9) is not None else 0,
                at_end=(m.group(10) == "at_end"),
            ))
            if pending and pending[0] == name:
                e.setdefault("oc", pending[1])
                e.setdefault("k", pending[2])
                pending = None
            continue
    return blocks


def main(path: str) -> int:
    blocks = parse(path)
    if not blocks:
        print("FAIL: 解析不到 hw 配置块（harness-dw 没跑？）")
        return 1

    ok = True
    for hw, cases in blocks.items():
        dw = [n for n in cases if n.startswith("dw:")]
        ctl = [n for n in cases if n.startswith("ctl:")]
        print("\nhw %s：dw 案 %d 个，控制案 %d 个" % (hw, len(dw), len(ctl)))
        if not dw or not ctl:
            ok = False
            print("  [反作弊] dw 或 ctl 案为空 —— 没测到东西")

        for n in sorted(dw):
            e = cases[n]
            expect_post = e.get("oc", 0) * e.get("k", 1) * e.get("k", 1)
            good = (e.get("verdict") == "OOB" and e.get("at_end")
                    and e.get("post") == expect_post)
            label = "OOB(末尾, post=oc·k²)" if good else "与预期不符"
            print("  %-28s %-24s post=%-8s ic=%-4s %s"
                  % (n, label, e.get("post"), e.get("ic"), "" if good else "★"))
            if not good:
                ok = False
                print("     明细: %s" % e)

        for n in sorted(ctl):
            e = cases[n]
            good = e.get("verdict") == "OK"
            print("  %-28s %-24s post=%-8s ic=%-4s %s"
                  % (n, "干净（负控制）" if good else "★ 意外越界", e.get("post"), e.get("ic"),
                     "" if good else "★"))
            if not good:
                ok = False
                print("     明细: %s" % e)

    print("\n== %s ==" % ("PASS：depthwise+per-channel 越界可复现（缓冲被缩 C 倍），负控制全干净"
                          if ok else "FAIL：实测模式与预期不符"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "dw.txt"))
