import re
import os

repo_root = "/home/devraj/MTok/external_models/lvr/repo"

def audit_doc(doc_path):
    print(f"\n=======================================================")
    print(f"AUDITING SNIPPETS IN {os.path.basename(doc_path)}")
    print(f"=======================================================")
    with open(doc_path, 'r') as f:
        lines = f.readlines()
    
    # We will iterate line by line looking for citations followed by code blocks
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for file citation with line numbers
        m = re.search(r'([a-zA-Z0-9_\-\./]+\.(?:py|sh|json|tex|bin))\b.*?(?:lines?|L|L\.)\s*([0-9]+)(?:[–\-]([0-9]+))?', line)
        if m:
            file_ref, start_l, end_l = m.group(1), int(m.group(2)), int(m.group(3)) if m.group(3) else int(m.group(2))
            # see if there is a code block within the next 10 lines
            code_block = []
            in_code = False
            for j in range(i, min(len(lines), i + 15)):
                if lines[j].strip().startswith("```") and not in_code:
                    in_code = True
                    continue
                elif lines[j].strip().startswith("```") and in_code:
                    in_code = False
                    break
                elif in_code:
                    code_block.append(lines[j])
            
            if code_block:
                # verify code_block against file_ref
                # resolve path
                resolved_path = None
                for base in [repo_root, "/home/devraj/MTok/external_models/lvr/paper_src", "/home/devraj/MTok/external_models/lvr/hf_metadata/LVR-7B"]:
                    for r, _, fs in os.walk(base):
                        if os.path.basename(file_ref) in fs:
                            resolved_path = os.path.join(r, os.path.basename(file_ref))
                            break
                    if resolved_path:
                        break
                
                if not resolved_path:
                    print(f"L{i+1}: Could not find file {file_ref}")
                else:
                    with open(resolved_path, 'r', errors='ignore') as rf:
                        target_lines = rf.readlines()
                    
                    # Check whether snippet lines appear around start_l..end_l
                    actual_slice = "".join(target_lines[max(0, start_l - 1):min(len(target_lines), end_l)])
                    full_file = "".join(target_lines)
                    
                    # For each non-comment non-empty line in code_block:
                    for cl in code_block:
                        cl_str = cl.strip()
                        if not cl_str or cl_str.startswith("#") or cl_str.startswith("...") or cl_str.startswith("//"):
                            continue
                        if cl_str not in actual_slice:
                            # Is it in the file at all?
                            if cl_str in full_file:
                                # find where
                                actual_locs = [k+1 for k, l in enumerate(target_lines) if cl_str in l]
                                print(f"L{i+1}: Snippet line '{cl_str[:40]}...' NOT in cited range {file_ref} L{start_l}-L{end_l}, but at L{actual_locs}")
                            else:
                                print(f"L{i+1}: Snippet line '{cl_str[:40]}...' NOT FOUND AT ALL in {file_ref}")
        i += 1

audit_doc("/home/devraj/MTok/external_models/lvr/CODE_VS_PAPER_TRACE.md")
audit_doc("/home/devraj/MTok/external_models/lvr/Gaps_LVR.md")
