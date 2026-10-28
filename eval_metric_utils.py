#!/usr/bin/env python3
"""
Evaluation Metric Utilities: Strict Sample Validation & Metric Extraction.

Ensures that only full evaluation runs (matching exact expected sample sizes)
are aggregated, preventing smoke tests or partial runs from polluting tables.
Fails loudly if expected tasks or samples are missing.
"""

import os
import glob
import json
import re

# Expected sample counts for BLINK spatial benchmarks
BLINK_EXPECTED_N = {
    "blink_sprel": 143,
    "blink_reldepth": 124,
    "blink_jigsaw": 150,
    "blink_mv": 133,
    "blink_iqtest": 150,
}

BLINK_METRIC_KEY = "blink_score_overall,flexible-extract"


def extract_blink_metrics(dir_path, require_all=True):
    """
    Extracts BLINK benchmark metrics from a directory with strict sample count validation.

    Args:
        dir_path: Directory path to search recursively for results.json files.
        require_all: If True, raises RuntimeError if any of the 5 BLINK tasks are missing.

    Returns:
        dict of {task_name: score_percentage} rounded to 2 decimal places.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    all_files = sorted(glob.glob(os.path.join(dir_path, "**/*result*.json"), recursive=True))
    # Exclude any files in smoke_archive
    files = [f for f in all_files if "smoke_archive" not in f]

    task_scores = {}
    task_files = {}

    for f_path in files:
        try:
            with open(f_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to parse JSON {f_path}: {e}")
            continue

        results = data.get("results", {})
        n_samples = data.get("n-samples", {})

        for task_name, expected_n in BLINK_EXPECTED_N.items():
            if task_name not in results:
                continue

            sample_info = n_samples.get(task_name, {})
            eff = sample_info.get("effective")
            orig = sample_info.get("original")

            # Strict sample validation: must match exact expected N
            if eff != expected_n or orig != expected_n:
                # Skip invalid / smoke run
                continue

            task_metrics = results[task_name]
            if BLINK_METRIC_KEY not in task_metrics:
                raise KeyError(
                    f"Expected metric key '{BLINK_METRIC_KEY}' not found in {f_path} for task {task_name}. "
                    f"Available keys: {list(task_metrics.keys())}"
                )

            val = task_metrics[BLINK_METRIC_KEY]
            if not isinstance(val, (int, float)):
                raise ValueError(f"Metric value for {task_name} in {f_path} is non-numeric: {val}")

            score = round(float(val) * 100.0, 2)

            # Check for conflicting duplicates
            if task_name in task_scores:
                prev_score = task_scores[task_name]
                if abs(prev_score - score) > 1e-4:
                    raise RuntimeError(
                        f"Conflicting duplicate results found for {task_name} in {dir_path}:\n"
                        f"  File 1: {task_files[task_name]} -> {prev_score}%\n"
                        f"  File 2: {f_path} -> {score}%"
                    )
            else:
                task_scores[task_name] = score
                task_files[task_name] = f_path

    if require_all:
        missing = [t for t in BLINK_EXPECTED_N if t not in task_scores]
        if missing:
            raise RuntimeError(
                f"Missing required BLINK tasks in {dir_path}: {missing}.\n"
                f"Found tasks: {list(task_scores.keys())}\n"
                f"Files searched: {files}"
            )

    return task_scores
