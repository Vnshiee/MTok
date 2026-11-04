#!/usr/bin/env python
"""
Render the Mirage audit documents from templates so that every cited code block is the verbatim source.

Placeholders (paths relative to external_models/mirage; output paths are written as external_models/mirage/..., see paths.py):
  {{CODE path a b}}            -> heading "`path` La–Lb" + fenced block with lines a..b copied verbatim
  {{TEX path n "substring"}}   -> blockquote of `substring`, asserted to occur on line n of path
  {{LINE path n "substring"}}  -> inline citation "`path` Ln", asserted: substring occurs on line n
  {{SHA path}}                 -> full SHA-256 of the file (freezes analysis code in the pre-registration)

Rendering fails (non-zero exit) if a file is missing, a range is out of bounds, or an asserted substring is absent.
Usage: python verification/render_docs.py  (renders every verification/templates/*.tmpl.md to ../<name>.md)
"""
import hashlib
import re
import sys
from pathlib import Path

from paths import qualify, resolve

HERE = Path(__file__).resolve().parent
MIRAGE = HERE.parent
ERRORS = []


def lines_of(rel):
    p = resolve(rel)
    if not p.exists():
        raise FileNotFoundError(rel)
    return p.read_text(encoding="utf-8").split("\n")


def code(m):
    rel, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    L = lines_of(rel)
    if not (1 <= a <= b <= len(L)):
        ERRORS.append(f"range {rel} L{a}-L{b} outside 1..{len(L)}")
        return m.group(0)
    lang = "python" if rel.endswith(".py") else ("bash" if rel.endswith(".sh") else "")
    body = "\n".join(L[a - 1:b])
    rng = f"L{a}" if a == b else f"L{a}–L{b}"
    return f"`{rel}` {rng}:\n\n```{lang}\n{body}\n```"


def assert_on_line(rel, n, sub):
    L = lines_of(rel)
    if not (1 <= n <= len(L)) or sub not in L[n - 1]:
        ERRORS.append(f"substring not on {rel} L{n}: {sub[:70]!r}")
        return False
    return True


def tex(m):
    rel, n, sub = m.group(1), int(m.group(2)), m.group(3)
    assert_on_line(rel, n, sub)
    return f"> `{rel}` L{n}: “{sub}”"


def line(m):
    rel, n, sub = m.group(1), int(m.group(2)), m.group(3)
    assert_on_line(rel, n, sub)
    return f"`{rel}` L{n}"


def sha(m):
    return hashlib.sha256(resolve(m.group(1)).read_bytes()).hexdigest()


def render(tmpl: Path) -> str:
    s = tmpl.read_text(encoding="utf-8")
    s = re.sub(r"\{\{SHA (\S+)\}\}", sha, s)
    s = re.sub(r"\{\{CODE (\S+) (\d+) (\d+)\}\}", code, s)
    s = re.sub(r'\{\{TEX (\S+) (\d+) "((?:[^"\\]|\\.)*)"\}\}', lambda m: tex(type("M", (), {"group": lambda self, i: (m.group(0), m.group(1), m.group(2), m.group(3).replace('\\"', '"'))[i]})()), s)
    s = re.sub(r'\{\{LINE (\S+) (\d+) "((?:[^"\\]|\\.)*)"\}\}', lambda m: line(type("M", (), {"group": lambda self, i: (m.group(0), m.group(1), m.group(2), m.group(3).replace('\\"', '"'))[i]})()), s)
    left = re.findall(r"\{\{[A-Z]+ [^}]*\}\}", s)
    if left:
        ERRORS.extend(f"unrendered placeholder: {x[:80]}" for x in left)
    return s


def main():
    for tmpl in sorted((HERE / "templates").glob("*.tmpl.md")):
        out = MIRAGE / tmpl.name.replace(".tmpl.md", ".md")
        text = render(tmpl)
        if ERRORS:
            print("\n".join(ERRORS))
            sys.exit(1)
        out.write_text(qualify(text), encoding="utf-8")
        print(f"rendered {out.relative_to(MIRAGE)}")


if __name__ == "__main__":
    main()
