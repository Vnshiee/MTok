import re

with open('/home/devraj/MTok/slides/presentation.tex', 'r') as f:
    content = f.read()

# 1. Add \definecolor{emerald}{HTML}{10B981}
if r'\definecolor{emerald}' not in content:
    content = content.replace(
        r'\definecolor{success}{HTML}{059669}',
        "\\definecolor{success}{HTML}{059669}\n\\definecolor{emerald}{HTML}{10B981}"
    )

# 2. Fix unescaped % inside \note{...}
# Find all \note{...} and escape % that are not already escaped
def fix_note(match):
    note_content = match.group(1)
    # replace unescaped % with \%
    fixed = re.sub(r'(?<!\\)%', r'\%', note_content)
    return f"\\note{{{fixed}}}"

# Match \note{...} allowing multiline
content = re.sub(r'\\note\{([^}]*)\}', fix_note, content, flags=re.DOTALL)

# Also check for any unescaped % in normal text outside comments
# In TeX, lines starting with % are comments.
lines = content.split('\n')
new_lines = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith('%'):
        new_lines.append(line)
        continue
    # If line has %, check if it's a comment or in text
    # Specifically, check if there's any unescaped % before a comment
    new_lines.append(line)

content = '\n'.join(new_lines)

with open('/home/devraj/MTok/slides/presentation.tex', 'w') as f:
    f.write(content)

print("Fixed presentation.tex")
