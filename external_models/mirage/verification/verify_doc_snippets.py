#!/usr/bin/env python
"""
Independent check that every cited code block in a Mirage audit document is verbatim.

A citation is a line of the form  `path` La–Lb:   (or `path` La:) followed by a fenced code block.
The block must equal lines a..b of path (external_models/mirage/... or relative to external_models/mirage) exactly.
Also checks that inline citations `path` Ln point to existing lines.
Usage: python verification/verify_doc_snippets.py CODE_VS_PAPER_TRACE.md Gaps_Mirage.md ...
"""
import re
import sys
from pathlib import Path

from paths import resolve

MIRAGE = Path(__file__).resolve().parent.parent
HEAD = re.compile(r"^`([^`]+)` L(\d+)(?:–L(\d+))?:\s*$")
INLINE = re.compile(r"`((?:external_models/mirage/)?(?:repo|paper_src|ablation_study|diagnostics|verification)/[^`]+|(?:external_models/mirage/)?model_loader\.py)` L(\d+)")


def check(doc):
    lines = (MIRAGE / doc).read_text(encoding="utf-8").split("\n")
    blocks = bad = inline = 0
    i = 0
    while i < len(lines):
        m = HEAD.match(lines[i])
        if m:
            path, a = m.group(1), int(m.group(2))
            b = int(m.group(3) or a)
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            k = j + 1
            while k < len(lines) and not lines[k].startswith("```"):
                k += 1
            block = lines[j + 1:k]
            src = resolve(path).read_text(encoding="utf-8").split("\n")[a - 1:b]
            blocks += 1
            if block != src:
                bad += 1
                print(f"{doc}:{i + 1}: block for {path} L{a}-L{b} is not verbatim")
            i = k + 1
            continue
        for mm in INLINE.finditer(lines[i]):
            p, n = resolve(mm.group(1)), int(mm.group(2))
            inline += 1
            if not p.exists() or n > len(p.read_text(encoding="utf-8").split("\n")):
                bad += 1
                print(f"{doc}:{i + 1}: inline citation {mm.group(0)} does not exist")
        i += 1
    print(f"{doc}: {blocks} code blocks, {inline} inline citations, {bad} problems")
    return bad


if __name__ == "__main__":
    sys.exit(1 if sum(check(d) for d in sys.argv[1:]) else 0)
