import re
import os

docs = [
    "/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md",
    "/home/devraj/MTok/external_models/lvr/Gaps_LVR.md"
]

# We want to extract every mention of file + line range in the document,
# e.g.: `src/train/monkey_patch_forward_lvr.py` lines 149–209
# or [`src/...`](...) lines 149-209
# or [`3_lvr.tex` L18](...)

for doc in docs:
    print(f"\n==========================================")
    print(f"AUDITING DETAILED CITATIONS IN: {os.path.basename(doc)}")
    print(f"==========================================")
    with open(doc, 'r') as f:
        text = f.read()

    # Find patterns like:
    # `path` lines X–Y or `path` line X or path L123
    patterns = [
        r'\[`?([a-zA-Z0-9_\-\./]+\.(?:py|sh|json|tex|bin|md|jinja))`?\](?:\([^)]+\))?\s*(?:lines?|L)\s*([0-9]+)(?:[–\-]([0-9]+))?',
        r'`([a-zA-Z0-9_\-\./]+\.(?:py|sh|json|tex|bin|md|jinja))`\s*(?:lines?|L)\s*([0-9]+)(?:[–\-]([0-9]+))?',
        r'([a-zA-Z0-9_\-\./]+\.(?:py|sh|json|tex|bin|md|jinja))\s*(?:lines?|L)\s*([0-9]+)(?:[–\-]([0-9]+))?'
    ]

    citations = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            fpath, s, e = m.groups()
            citations.append((m.start(), fpath, int(s), int(e) if e else int(s)))

    # Sort by start pos
    citations.sort(key=lambda x: x[0])
    
    # Deduplicate
    seen = set()
    unique_citations = []
    for pos, fp, s, e in citations:
        if (fp, s, e) not in seen:
            seen.add((fp, s, e))
            unique_citations.append((pos, fp, s, e))

    print(f"Total unique file+line citations found: {len(unique_citations)}")
    for pos, fp, s, e in unique_citations:
        # resolve fp to absolute path
        abs_path = None
        candidates = [
            os.path.join("/home/devraj/MTok/external_models/lvr/repo", fp),
            os.path.join("/home/devraj/MTok/external_models/lvr/paper_src", fp),
            os.path.join("/home/devraj/MTok/external_models/lvr/paper_src/sections", fp),
            os.path.join("/home/devraj/MTok/external_models/lvr/paper_src/tables", fp),
            os.path.join("/home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B", fp),
            os.path.join("/home/devraj/MTok/external_models/lvr", fp)
        ]
        for c in candidates:
            if os.path.exists(c):
                abs_path = c
                break
        if not abs_path:
            # try finding by basename
            base = os.path.basename(fp)
            for root_dir in ["/home/devraj/MTok/external_models/lvr/repo", "/home/devraj/MTok/external_models/lvr/paper_src", "/home/devraj/MTok/external_models/lvr/hf_metadata"]:
                for r, _, files in os.walk(root_dir):
                    if base in files:
                        abs_path = os.path.join(r, base)
                        break
                if abs_path:
                    break
        
        if not abs_path:
            print(f"COULD NOT RESOLVE FILE: {fp}")
            continue

        # Check line bounds
        if abs_path.endswith('.bin'):
            print(f"BINARY FILE CITED WITH LINE NUMBER: {fp} L{s}-L{e}")
            continue

        with open(abs_path, 'r', errors='ignore') as tf:
            flines = tf.readlines()
        total = len(flines)
        if s > total or e > total:
            print(f"OUT OF BOUNDS: {fp} ({abs_path}) has {total} lines, cited {s}-{e}")
        else:
            # print sample
            pass

