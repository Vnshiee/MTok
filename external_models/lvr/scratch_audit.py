import re
import os

repo_root = "/home/devraj/MTok/external_models/lvr/repo"
paper_root = "/home/devraj/MTok/external_models/lvr/paper_src"
hf_root = "/home/devraj/MTok/external_models/lvr/hf_metadata"

docs = [
    "/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md",
    "/home/devraj/MTok/external_models/lvr/Gaps_LVR.md"
]

# regex to find links like [text](file:///path#Lstart-Lend) or [text](file:///path#Lstart) or [text](file:///path)
link_regex = re.compile(r'\[([^\]]*)\]\(file://([^\)#]+)(?:#L(\d+)(?:-L(\d+))?)?\)')

for doc in docs:
    print(f"\n==========================================")
    print(f"AUDITING LINKS IN: {os.path.basename(doc)}")
    print(f"==========================================")
    with open(doc, 'r') as f:
        doc_lines = f.readlines()
    
    for line_idx, line in enumerate(doc_lines, 1):
        for m in link_regex.finditer(line):
            text, path, start_l, end_l = m.groups()
            if not os.path.exists(path):
                print(f"[Line {line_idx}] MISSING FILE: {path}")
                continue
            
            # Check line numbers
            if start_l:
                s = int(start_l)
                e = int(end_l) if end_l else s
                try:
                    with open(path, 'r', errors='ignore') as tf:
                        target_lines = tf.readlines()
                    total = len(target_lines)
                    if s > total or e > total or s < 1 or e < s:
                        print(f"[Line {line_idx}] OUT OF BOUNDS: {path} has {total} lines, cited L{s}-L{e}")
                except Exception as err:
                    print(f"[Line {line_idx}] ERROR READING {path}: {err}")

