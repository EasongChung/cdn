#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不变式判定：越界只允许出现在 oc % 8 != 0 的形状上；且结论不依赖硬件参数猜测。

输入：harness --sweep 的输出（按 `-- hw ... --` 分块）。

判据：
  1) 每个 FAIL 的形状名必须能解出 oc，且 oc % 8 != 0        （正/负控制）
  2) 各 hw 配置的失败集合必须**完全一致**                   （不依赖 accum/zrl 猜测值）
  3) FAIL 总数 > 0                                          （反作弊：harness 真的跑了）
退出码 0 = 全部满足。
"""
import re
import sys


def parse(path):
    blocks, cur = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^-- hw (.+) --\s*$", line)
        if m:
            cur = m.group(1)
            blocks[cur] = []
            continue
        m = re.search(r"\bFAIL (\S+)", line)
        if m and cur:
            blocks[cur].append(m.group(1))
    return blocks


def oc_of(name):
    m = re.search(r":(\d+)x\d+_k\d+x\d+", name)
    return int(m.group(1)) if m else None


def main(path):
    blocks = parse(path)
    if not blocks:
        print("FAIL: 没有解析到任何 hw 配置块（harness 没跑？）")
        return 1

    bad, sets = [], {}
    for hw, fails in blocks.items():
        sets[hw] = set(fails)
        for n in fails:
            oc = oc_of(n)
            if oc is None:
                bad.append((hw, n, "形状名无法解析 oc"))
            elif oc % 8 == 0:
                bad.append((hw, n, "oc=%d 是 8 的倍数却越界（说明还有别的机制）" % oc))
        print("hw %-46s 失败 %2d 个" % (hw, len(fails)))

    all_fail = set().union(*sets.values()) if sets else set()
    print("\n去重后的越界形状 %d 个：" % len(all_fail))
    for n in sorted(all_fail):
        print("   %-40s oc=%s  oc%%8=%s" % (n, oc_of(n), (oc_of(n) or 0) % 8))

    ok = True
    if bad:
        ok = False
        print("\n[违反判据1]")
        for hw, n, why in bad:
            print("   %s | %s | %s" % (hw, n, why))

    if len({frozenset(s) for s in sets.values()}) != 1:
        ok = False
        print("\n[违反判据2] 不同 hw 配置的失败集合不一致")

    if not all_fail:
        ok = False
        print("\n[违反判据3] 一个越界都没有 —— 反作弊检查失败")

    print("\n== %s ==" % ("PASS：越界全部命中 oc%8!=0，且不随 hw 参数变化" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "out.txt"))
