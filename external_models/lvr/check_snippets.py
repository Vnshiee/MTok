import re
import os

def check_snippet(doc_name, citation_desc, file_path, start_line, end_line, snippet_text):
    if not os.path.exists(file_path):
        print(f"[{doc_name}] File not found: {file_path}")
        return
    with open(file_path, 'r', errors='ignore') as f:
        lines = f.readlines()
    
    # Extract actual lines (1-indexed)
    actual_lines = lines[start_line - 1 : end_line]
    actual_block = "".join(actual_lines)
    
    # Clean whitespace for comparison
    clean_snippet = "\n".join(line.strip() for line in snippet_text.strip().splitlines() if line.strip() and not line.strip().startswith("#"))
    clean_actual = "\n".join(line.strip() for line in actual_block.strip().splitlines() if line.strip() and not line.strip().startswith("#"))
    
    # Check if lines match closely or if there's any discrepancy
    print(f"[{doc_name}] Checking {citation_desc} ({file_path} L{start_line}-L{end_line}):")
    # Check if snippet lines appear in actual lines
    snippet_lines = [l.strip() for l in snippet_text.strip().splitlines() if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("...")]
    missing = []
    for sl in snippet_lines:
        found = any(sl in al for al in actual_lines)
        if not found:
            # check within +/- 15 lines of range
            extended = lines[max(0, start_line - 20) : min(len(lines), end_line + 20)]
            found_ext = any(sl in el for el in extended)
            if found_ext:
                missing.append(f"Line '{sl}' not in L{start_line}-L{end_line}, but found nearby!")
            else:
                missing.append(f"Line '{sl}' NOT FOUND in file near L{start_line}-L{end_line}!")
    
    if missing:
        for m in missing:
            print(f"  DISCREPANCY: {m}")
    else:
        print(f"  OK: All {len(snippet_lines)} code lines verified.")

# Let's test the snippets cited in CODE_VS_PAPER_TRACE.md
