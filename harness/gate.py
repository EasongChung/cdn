#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不变式判定：越界集合必须**逐字等于** `C % cores_used != 0` 的预测。

理论（v7 写入器）：
    cores_used      = MIN2(output_channels, nn_core_count)
    kernels_per_core = DIV_ROUND_UP(output_channels, cores_used)
    幻影通道        = cores_used * kernels_per_core - output_channels

    ⇒ 幻影 m > 0  ⇔  C % cores_used != 0
    ⇒ 当 C < cores 时 cores_used = C ⇒ m = 0（安全）

本脚本对每个 hw 配置独立预测，再与实测失败集合**逐元素比对**。
这比「所有配置失败集合相同」强得多：它验证的是**机制公式**本身。

反作弊：至少要有 1 个失败；否则说明 harness 没真正跑起来，判 FAIL。
"""
import re
import sys


def parse_cases(path):
    cases = []
    for line in open(path, encoding="utf-8"):
        m = re.match(r'\s*\{"([^"]+)",\s*(\d+),', line)
        if m:
            cases.append((m.group(1), int(m.group(2))))
    return cases


def parse_out(path):
    blocks, cur = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^-- hw (.+) --\s*$", line)
        if m:
            cur = m.group(1)
            blocks[cur] = set()
            continue
        m = re.search(r"\bFAIL (\S+)", line)
        if m and cur is not None:
            blocks[cur].add(m.group(1))
    return blocks


def main(out_path, cases_path):
    cases = parse_cases(cases_path)
    blocks = parse_out(out_path)
    if not cases:
        print("FAIL: 解析不到 cases.h 里的形状")
        return 1
    if not blocks:
        print("FAIL: 解析不到 hw 配置块（harness 没跑？）")
        return 1

    print("形状数=%d；hw 配置数=%d" % (len(cases), len(blocks)))
    ok = True
    total_fail = 0

    for hw, actual in blocks.items():
        m = re.search(r"cores=(\d+)", hw)
        if not m:
            print("FAIL: 配置块里没有 cores：%s" % hw)
            return 1
        cores = int(m.group(1))

        predicted = {name for name, oc in cases if oc >= cores and oc % cores != 0}
        total_fail += len(actual)

        # 预测漏了（实测有、理论无）
        extra = actual - predicted
        # 预测多了（理论有、实测无）
        missing = predicted - actual

        status = "OK" if not extra and not missing else "MISMATCH"
        print("\nhw %-46s cores=%d  实测=%2d 预测=%2d  %s"
              % (hw, cores, len(actual), len(predicted), status))
        for n in sorted(extra):
            ok = False
            print("   [理论未预测但实测越界] %s" % n)
        for n in sorted(missing):
            ok = False
            print("   [理论预测越界但实测干净] %s" % n)

    print("\n去重后的越界形状：%d 个" % len(set().union(*blocks.values())))
    for n in sorted(set().union(*blocks.values())):
        print("   %s" % n)

    if total_fail == 0:
        ok = False
        print("\n[反作弊] 一个越界都没有 —— harness 没跑对")

    print("\n== %s ==" % ("PASS：实测越界集合逐字等于 C % cores_used != 0 的预测" if ok
                          else "FAIL：实测与机制公式不符"))
    return 0 if ok else 1


if __name__ == "__main__":
    o = sys.argv[1] if len(sys.argv) > 1 else "out.txt"
    c = sys.argv[2] if len(sys.argv) > 2 else "cases.h"
    sys.exit(main(o, c))
