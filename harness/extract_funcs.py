#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从**未改写**的 Mesa 源码里按函数名提取函数体，生成可编译的 TU。

为什么要提取而不是复制：
    `requantize_weights()` 是 `static`，单独编译 `etnaviv_ml_nn.c` 需要 stub 掉上百个
    外部符号（pipe/p_*.h 全套 + 资源层）。提取只取**函数体原文**，再做 md5 绑定，
    既不需要 stub 整个文件，又保证「测的就是源码里的那段代码」。

防漂移：
    · 提取出的 body 与源码**逐字节相同**（只去掉外层 `static`）
    · `--expect-body-md5` 绑定 body 的 md5；上游一改就失败（CI 会红）
    · 生成的 .c 头部写入来源路径 + 行号 + md5，便于复核

用法：
    extract_funcs.py --src <etnaviv_ml_nn.c> --func requantize_weights \
                     --out build/requantize_extracted.c [--expect-body-md5 <md5>]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys


def find_body(text: str, func: str) -> tuple[str, int]:
    """返回 (函数体原文含大括号, 起始行号)。用花括号配对，跳过字符串/注释/字符常量。"""
    m = re.search(r"^%s\s*\(" % re.escape(func), text, re.MULTILINE)
    if not m:
        raise SystemExit("找不到函数 %s" % func)
    start_line = text.count("\n", 0, m.start()) + 1

    i = text.index("{", m.end())
    depth = 0
    j = i
    in_str = in_chr = in_line_c = in_blk_c = False
    while j < len(text):
        ch = text[j]
        nxt = text[j + 1] if j + 1 < len(text) else ""
        if in_line_c:
            if ch == "\n":
                in_line_c = False
        elif in_blk_c:
            if ch == "*" and nxt == "/":
                in_blk_c = False
                j += 1
        elif in_str:
            if ch == "\\":
                j += 1
            elif ch == '"':
                in_str = False
        elif in_chr:
            if ch == "\\":
                j += 1
            elif ch == "'":
                in_chr = False
        elif ch == "/" and nxt == "/":
            in_line_c = True
            j += 1
        elif ch == "/" and nxt == "*":
            in_blk_c = True
            j += 1
        elif ch == '"':
            in_str = True
        elif ch == "'":
            in_chr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i : j + 1], start_line
        j += 1
    raise SystemExit("花括号不配对：%s" % func)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--func", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--signature", required=True, help="生成文件里用的函数声明（不含分号）")
    ap.add_argument("--includes", default="", help="逗号分隔的 #include 名字")
    ap.add_argument("--expect-body-md5", default=None)
    args = ap.parse_args(argv[1:])

    text = open(args.src, encoding="utf-8", errors="replace").read()
    body, line = find_body(text, args.func)
    md5 = hashlib.md5(body.encode("utf-8")).hexdigest()

    print("提取 %s()  源码行 %d  body=%d 字节  md5=%s" % (args.func, line, len(body), md5))
    if args.expect_body_md5 and md5 != args.expect_body_md5:
        print("FAIL: body md5 与预期不符（上游源码改过了）", file=sys.stderr)
        print("  预期 %s" % args.expect_body_md5, file=sys.stderr)
        print("  实际 %s" % md5, file=sys.stderr)
        return 1

    inc = "".join('#include "%s"\n' % h for h in args.includes.split(",") if h)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("/* 自动生成，勿手改 —— extract_funcs.py\n")
        fh.write(" * 来源：%s 第 %d 行  %s()\n" % (args.src, line, args.func))
        fh.write(" * 函数体 md5（逐字节）= %s\n */\n" % md5)
        fh.write('#include <math.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <stdbool.h>\n#include <float.h>\n')
        fh.write(inc)
        fh.write("%s\n" % args.signature)
        fh.write(body)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
