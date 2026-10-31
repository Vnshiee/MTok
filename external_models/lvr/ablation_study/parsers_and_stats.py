"""
Answer Extraction Parsers, Prompt Formatters, Circular Permutation Generator,
and Paired Statistical Testing Engine for the LVR Ablation Study.

Implements:
- Axis 5 Factor A: Official `evaluation.py` first-char parser vs. Robust regex/option parser.
- Axis 5 Factor B: Official prompt formatting vs. SFT bbox-prompt / clean prompt formatting.
- Axis 5 Factor C: Circular option permutation (`CircularEval`).
- Section 4.3: Paired McNemar's test, Exact Binomial test (b + c < 25),
  B=10,000 Paired Bootstrap 95% CIs, and Holm-Bonferroni step-down correction.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def accuracy_reward_official(response: str, ground_truth: str) -> bool:
    """
    Exact line-for-line replica of `accuracy_reward` in `repo/evaluation/evaluation.py` L52-L61:
        given_answer = response.split('<answer>')[-1]
        given_answer = given_answer.split('</answer')[0].strip()
        if " " in given_answer:
            given_answer = given_answer.split(" ")[0]
        if len(given_answer) >1:
            given_answer = given_answer[0]
        return given_answer == ground_truth
    """
    given_answer = response.split("<answer>")[-1]
    given_answer = given_answer.split("</answer")[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) > 1:
        given_answer = given_answer[0]
    return bool(given_answer == ground_truth)


def extract_robust_letter(
    response: str,
    options_dict: Optional[Dict[str, str]] = None,
) -> str:
    """
    Cascaded robust answer extractor (Axis 5 Factor A2: `Parser-Robust-Regex`).
    Eliminates the first-character extraction bug (`"Answer: B"` -> `'A'`) while
    strictly respecting `<answer>...</answer>` tags when present.
    """
    if not response:
        return ""

    # 1. Strip special tokens
    cleaned = re.sub(
        r"<\|lvr_start\|>|<\|lvr\|>|<\|lvr_latent_end\|>|<\|lvr_end\|>|<\|im_end\|>|<\|endoftext\|>",
        " ",
        response,
    ).strip()

    # 2. If <answer>...</answer> is present, prioritize its interior
    ans_match = re.search(r"<answer>(.*?)(?:</answer>|$)", cleaned, flags=re.DOTALL | re.IGNORECASE)
    target_text = ans_match.group(1).strip() if ans_match else cleaned

    valid_letters = set(options_dict.keys()) if options_dict else {"A", "B", "C", "D", "E", "F"}
    letter_pattern = "|".join(sorted(valid_letters))

    # 3. Direct single letter or parenthesized/dotted letter at start: e.g. "B", "(B)", "B.", "B) cotton"
    m_start = re.match(
        rf"^\s*[\(\[]?({letter_pattern})[\)\]\.:]?(?:\s|$)",
        target_text,
        flags=re.IGNORECASE,
    )
    if m_start:
        return m_start.group(1).upper()

    # 4. Explicit phrase: "The answer is (B)", "Answer: B", "Option C is correct"
    m_phrase = re.search(
        rf"(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?({letter_pattern})[\)\]\.:]?(?:\b|$)",
        target_text,
        flags=re.IGNORECASE,
    )
    if m_phrase:
        return m_phrase.group(1).upper()

    # 5. Parenthesized letter anywhere in target_text: "(A)", "(B)"
    m_paren = re.findall(rf"\(({letter_pattern})\)", target_text, flags=re.IGNORECASE)
    if len(m_paren) == 1:
        return m_paren[0].upper()

    # 6. Standalone uppercase option letter token
    m_standalone = re.findall(rf"\b({letter_pattern})\b", target_text)
    if len(m_standalone) == 1:
        return m_standalone[0].upper()

    # 7. Option text substring match against options_dict
    if options_dict:
        matched_letters = []
        target_lower = target_text.lower()
        for letter, opt_str in options_dict.items():
            opt_clean = opt_str.strip().lower().rstrip(".")
            if opt_clean and re.search(rf"\b{re.escape(opt_clean)}\b", target_lower):
                matched_letters.append(letter)
        if len(matched_letters) == 1:
            return matched_letters[0]

    # 8. Final fallback to first character of target_text if it is a valid letter
    if target_text and target_text[0].upper() in valid_letters:
        return target_text[0].upper()

    return ""


def accuracy_reward_robust(
    response: str,
    ground_truth: str,
    options_dict: Optional[Dict[str, str]] = None,
) -> bool:
    pred_letter = extract_robust_letter(response, options_dict=options_dict)
    return bool(pred_letter == ground_truth.strip().upper())


def build_circular_shifted_item(item: Dict[str, Any], shift: int) -> Dict[str, Any]:
    """
    Constructs a cyclic option shift of `item` by `shift` positions for `CircularEval` (Axis 5 Factor C2).
    If original choices are A: opt0, B: opt1, C: opt2, D: opt3 and shift=1,
    new choices assign A: opt1, B: opt2, C: opt3, D: opt0, and update `label` accordingly.
    """
    options_dict = item.get("options_dict", {})
    letters = sorted(options_dict.keys())
    n_opts = len(letters)
    if n_opts < 2 or shift % n_opts == 0:
        return dict(item)

    shift = shift % n_opts
    new_options_dict = {}
    old_to_new_letter = {}
    for idx, new_letter in enumerate(letters):
        old_letter = letters[(idx + shift) % n_opts]
        new_options_dict[new_letter] = options_dict[old_letter]
        old_to_new_letter[old_letter] = new_letter

    new_label = old_to_new_letter[item["label"]]

    # Rebuild question prompt with shifted options
    bench = item["benchmark"]
    raw_text = item["text_clean"]
    if bench == "vstar":
        # Replace (A) ... (B) ... lines
        lines = raw_text.splitlines()
        new_lines = []
        for line in lines:
            m = re.match(r"^\(([A-D])\)\s*(.+)$", line)
            if m and m.group(1) in new_options_dict:
                let = m.group(1)
                new_lines.append(f"({let}) {new_options_dict[let]}")
            else:
                new_lines.append(line)
        new_clean = "\n".join(new_lines)
        new_official = new_clean + "\nAnswer with the option's letter from the given choices directly."
    elif bench == "mmvp":
        q_part = raw_text.split("\nOptions:\n")[0]
        opt_line = f"(a) {new_options_dict['A']} (b) {new_options_dict['B']}"
        new_clean = (
            q_part
            + "\nOptions:\n"
            + opt_line
            + "\nAnswer with the option's letter from the given choices directly."
        )
        new_official = (
            q_part
            + "\nOptions:\n"
            + opt_line.replace("(a)", "A.").replace("(b)", "B.")
            + "\nAnswer with the option's letter from the given choices directly."
        )
    else:  # blink
        q_part = raw_text.split("\nOptions:\n")[0]
        opt_lines = "".join([f"{let}. {new_options_dict[let]}\n" for let in letters])
        new_official = (
            q_part
            + "\nOptions:\n"
            + opt_lines
            + "\nAnswer with the option's letter from the given choices directly."
        )
        new_clean = new_official

    shifted_item = dict(item)
    shifted_item["options_dict"] = new_options_dict
    shifted_item["label"] = new_label
    shifted_item["text_official_eval"] = new_official
    shifted_item["text_clean"] = new_clean
    shifted_item["circular_shift"] = shift
    return shifted_item


def compute_paired_statistics(
    correct_a: List[bool],
    correct_b: List[bool],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes exact paired statistical metrics comparing Condition A vs. Condition B (Section 4.3):
    - Acc(A), Acc(B), Delta Acc(A - B) in percentage points.
    - Contingency table (n00, n01, n10, n11) where b = n10 (A correct, B wrong), c = n01 (A wrong, B correct).
    - Paired McNemar's test (Edwards' continuity-corrected chi2 for b+c >= 25; exact two-sided Binomial for b+c < 25).
    - B=10,000 paired bootstrap 95% confidence interval on Delta Acc(A - B).
    """
    arr_a = np.asarray(correct_a, dtype=np.int32)
    arr_b = np.asarray(correct_b, dtype=np.int32)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {arr_a.shape} vs {arr_b.shape}")
    n = len(arr_a)
    if n == 0:
        raise ValueError("Empty input arrays")

    n11 = int(np.sum((arr_a == 1) & (arr_b == 1)))
    n10 = int(np.sum((arr_a == 1) & (arr_b == 0)))  # b: A right, B wrong
    n01 = int(np.sum((arr_a == 0) & (arr_b == 1)))  # c: A wrong, B right
    n00 = int(np.sum((arr_a == 0) & (arr_b == 0)))

    b, c = n10, n01
    discordant = b + c

    if discordant == 0:
        p_val = 1.0
        stat_val = 0.0
        test_used = "identical_predictions"
    elif discordant < 25:
        res = stats.binomtest(b, n=discordant, p=0.5, alternative="two-sided")
        p_val = float(res.pvalue)
        stat_val = float(b)
        test_used = "exact_binomial"
    else:
        chi2 = ((abs(b - c) - 1.0) ** 2) / float(discordant)
        p_val = float(stats.chi2.sf(chi2, df=1))
        stat_val = float(chi2)
        test_used = "mcnemar_continuity_corrected"

    acc_a = float(np.mean(arr_a)) * 100.0
    acc_b = float(np.mean(arr_b)) * 100.0
    delta_acc = acc_a - acc_b

    # Paired bootstrap on diff = (arr_a - arr_b) * 100.0
    diffs = (arr_a - arr_b).astype(np.float64) * 100.0
    rng = np.random.default_rng(seed=seed)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(diffs[boot_indices], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "n": n,
        "acc_a_pct": round(acc_a, 4),
        "acc_b_pct": round(acc_b, 4),
        "delta_acc_pct": round(delta_acc, 4),
        "ci95_pct": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "contingency": {
            "n11_both_correct": n11,
            "n10_a_only_b": b,
            "n01_b_only_c": c,
            "n00_both_wrong": n00,
            "discordant_b_plus_c": discordant,
        },
        "test_used": test_used,
        "statistic": round(stat_val, 4),
        "p_value": p_val,
    }


def holm_bonferroni_correction(
    p_values: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """
    Applies Holm-Bonferroni step-down family-wise error rate control to a dictionary
    of {hypothesis_id: raw_p_value}.
    """
    sorted_items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(sorted_items)
    results: Dict[str, Dict[str, Any]] = {}
    rejected_so_far = True
    running_max_adj_p = 0.0

    for rank_idx, (hid, raw_p) in enumerate(sorted_items):
        multiplier = m - rank_idx
        adj_p = min(1.0, raw_p * multiplier)
        running_max_adj_p = max(running_max_adj_p, adj_p)
        threshold = alpha / multiplier
        if rejected_so_far and raw_p <= threshold:
            reject = True
        else:
            reject = False
            rejected_so_far = False

        results[hid] = {
            "rank": rank_idx + 1,
            "raw_p": raw_p,
            "adjusted_p": round(running_max_adj_p, 6),
            "holm_threshold": round(threshold, 6),
            "reject_null": reject,
        }
    return results


if __name__ == "__main__":
    print("Running unit tests for parsers_and_stats.py...")
    # 1. Test first-char bug vs robust parser
    resp_instruct = "Answer: B. The flag is red."
    assert accuracy_reward_official(resp_instruct, "B") is False, "Official parser must fail on 'Answer: B'"
    assert accuracy_reward_official(resp_instruct, "A") is True
    assert accuracy_reward_robust(resp_instruct, "B", {"A": "white", "B": "red"}) is True

    resp_lvr = "<|lvr_start|><|lvr_end|><answer>B</answer><|im_end|>"
    assert accuracy_reward_official(resp_lvr, "B") is True
    assert accuracy_reward_robust(resp_lvr, "B") is True

    # 2. Test paired statistics
    a = [True] * 80 + [False] * 20
    b = [True] * 70 + [False] * 30
    stats_out = compute_paired_statistics(a, b)
    print("Paired stats sample:", stats_out)
    assert stats_out["contingency"]["n10_a_only_b"] == 10
    assert stats_out["contingency"]["n01_b_only_c"] == 0
    assert stats_out["p_value"] < 0.01
    print("All unit tests in parsers_and_stats.py PASSED!")
