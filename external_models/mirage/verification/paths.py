"""
Path resolution and qualification utilities for Mirage audit and ablation documents.
"""
import re
from pathlib import Path

VERIFICATION_DIR = Path(__file__).resolve().parent
MIRAGE_DIR = VERIFICATION_DIR.parent
MTOCK_ROOT = MIRAGE_DIR.parent.parent

def resolve(rel) -> Path:
    """Resolve a path (relative to MIRAGE_DIR or MTok repo root) to an absolute Path."""
    s = str(rel).strip()
    prefix = "external_models/mirage/"
    if s.startswith(prefix):
        s = s[len(prefix):]
    elif s.startswith(str(MIRAGE_DIR) + "/"):
        s = s[len(str(MIRAGE_DIR) + "/"):]
    return (MIRAGE_DIR / s).resolve()

def qualify(text: str) -> str:
    """Rewrite citations of repo-relative paths into external_models/mirage/..."""
    # 1. Backticked paths: `repo/...`, `paper_src/...`, `ablation_study/...`, etc.
    pattern = re.compile(r"`(?!external_models/mirage/)((?:repo|paper_src|ablation_study|diagnostics|verification)/[^`\n]+|model_loader\.py)`")
    text = pattern.sub(r"`external_models/mirage/\1`", text)
    # 2. Markdown links: [text](repo/...) -> [text](external_models/mirage/repo/...)
    link_pattern = re.compile(r"\]\((?!external_models/mirage/)((?:repo|paper_src|ablation_study|diagnostics|verification)/[^)\n]+|model_loader\.py)\)")
    text = link_pattern.sub(r"](external_models/mirage/\1)", text)
    return text

