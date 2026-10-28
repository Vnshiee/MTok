#!/usr/bin/env python3
"""
watchdog_e1_matrix.py: Background Watchdog & Telemetry for E1 Matrix Evaluation.

Monitors:
- Active row, task, and sample count
- Process health (lmms_eval PID, CPU%, memory)
- GPU VRAM utilization and external process status
- Automatically regenerates WATCHDOG_STATUS.md and calls summarize_e1_matrix.py
"""

import os
import sys
import time
import glob
import json
import subprocess
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJECT_DIR, ".mtok", "bin", "python")
EVAL_BASE = os.path.join(PROJECT_DIR, "eval_outputs", "e1_matrix")
STATUS_FILE = os.path.join(EVAL_BASE, "WATCHDOG_STATUS.md")
LOG_FILE = os.path.join(EVAL_BASE, "watchdog.log")

ROWS = [
    {"row": 1, "dir": "row1_mull_ref", "desc": "Mull | latents | think + 20 latents (Ref baseline)"},
    {"row": 2, "dir": "row2_mull_empty", "desc": "Mull | latents | empty (K=0 repeat)"},
    {"row": 3, "dir": "row3_mull_empty_think", "desc": "Mull | latents | <think></think>\\n (Tag-only)"},
    {"row": 4, "dir": "row4_mull_dot_pause", "desc": "Mull | latents | think + 20x' .' (Pause tokens)"},
    {"row": 5, "dir": "row5_mull_latent_end", "desc": "Mull | latents | think + 20x<|latent_end|> (Special token)"},
    {"row": 6, "dir": "row6_mull_sft_empty", "desc": "Mull | sft | empty (Prompt confound test)"},
    {"row": 7, "dir": "row7_mull_sft_latents", "desc": "Mull | sft | think + 20 latents (Mull w/ SFT prompt)"},
    {"row": 8, "dir": "row8_sat_sft_empty", "desc": "SAT | sft | empty (Direct SAT SFT baseline)"},
    {"row": 9, "dir": "row9_sat_latents_empty", "desc": "SAT | latents | empty (SAT w/ latent prompt)"},
    {"row": 10, "dir": "row10_mull_grpo_k20", "desc": "Mull-GRPO | latents | think + 20 latents (Stage 3 GRPO)"},
    {"row": 11, "dir": "row11_mull_k1", "desc": "Mull | latents | think + 1 latent (K=1)"},
    {"row": 12, "dir": "row12_mull_k5", "desc": "Mull | latents | think + 5 latents (K=5)"},
    {"row": 13, "dir": "row13_mull_k10", "desc": "Mull | latents | think + 10 latents (K=10)"},
    {"row": 14, "dir": "row14_mull_k40", "desc": "Mull | latents | think + 40 latents (K=40)"},
    {"row": 15, "dir": "row15_mull_warm_k20", "desc": "Mull | latents | think + 20 latents (Warm GPU Ref)"},
]

TASKS = ["blink_sprel", "blink_reldepth", "blink_jigsaw", "blink_mv", "blink_iqtest"]
TASK_TOTALS = {
    "blink_sprel": 143,
    "blink_reldepth": 124,
    "blink_jigsaw": 150,
    "blink_mv": 133,
    "blink_iqtest": 150,
}

def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(EVAL_BASE, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        used, total, util = int(parts[0]), int(parts[1]), int(parts[2])
        return {"used_mib": used, "total_mib": total, "free_mib": total - used, "util_pct": util}
    except Exception:
        return {"used_mib": 0, "total_mib": 40960, "free_mib": 40960, "util_pct": 0}

def check_eval_process():
    try:
        out = subprocess.check_output(["pgrep", "-f", "lmms_eval"], text=True).strip()
        pids = [int(p) for p in out.splitlines() if p.strip()]
        return pids
    except Exception:
        return []

def count_completed_samples(row_dir, task):
    sample_file = os.path.join(row_dir, "per_sample_logs", f"per_sample_details_{task}.jsonl")
    if not os.path.exists(sample_file):
        return 0
    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0

def check_task_completion(row_dir, task):
    expected_n = TASK_TOTALS[task]
    pattern = os.path.join(row_dir, "**", "*result*.json")
    for f in glob.glob(pattern, recursive=True):
        if "_summary.json" in f or "samples_" in f or "smoke_archive" in f:
            continue
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
            sample_info = data.get("n-samples", {}).get(task, {})
            if sample_info.get("effective") == expected_n and sample_info.get("original") == expected_n:
                return True
        except Exception:
            pass
    return False

def update_status():
    gpu = get_gpu_info()
    pids = check_eval_process()
    is_running = len(pids) > 0

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_tasks = len(ROWS) * len(TASKS)
    completed_tasks = 0
    active_row = None
    active_task = None
    active_sample_count = 0

    row_statuses = []

    for r in ROWS:
        row_dir = os.path.join(EVAL_BASE, r["dir"])
        task_done_count = 0
        current_active = None

        for t in TASKS:
            if check_task_completion(row_dir, t):
                task_done_count += 1
            elif current_active is None:
                # First non-completed task in row
                s_count = count_completed_samples(row_dir, t)
                if s_count > 0 or (active_row is None and is_running):
                    current_active = (t, s_count)

        completed_tasks += task_done_count

        if task_done_count == len(TASKS):
            status = "COMPLETED"
        elif current_active is not None and is_running and active_row is None:
            status = f"IN_PROGRESS ({current_active[0]}: {current_active[1]}/{TASK_TOTALS[current_active[0]]})"
            active_row = r["row"]
            active_task = current_active[0]
            active_sample_count = current_active[1]
        elif task_done_count > 0:
            status = f"PARTIAL ({task_done_count}/{len(TASKS)} tasks)"
        else:
            status = "PENDING"

        row_statuses.append({
            "row": r["row"],
            "desc": r["desc"],
            "done_tasks": task_done_count,
            "status": status,
        })

    overall_pct = (completed_tasks / total_tasks) * 100.0

    lines = [
        "# E1 Matrix Evaluation Watchdog Status",
        f"**Last Heartbeat:** `{now_str}`  ",
        f"**Process Status:** `{'RUNNING (PID ' + str(pids[0]) + ')' if is_running else 'IDLE / COMPLETED'}`  ",
        f"**GPU Utilization:** `{gpu['util_pct']}%` | VRAM Used: `{gpu['used_mib']} MiB` / `{gpu['total_mib']} MiB` (Free: `{gpu['free_mib']} MiB`)  ",
        f"**Overall Progress:** `{completed_tasks}/{total_tasks} tasks ({overall_pct:.1f}%)`  \n",
    ]

    if is_running and active_row:
        lines.append(f"### Currently Evaluating: Row {active_row} — Task `{active_task}` ({active_sample_count}/{TASK_TOTALS[active_task]} samples)\n")

    lines.append("| Row | Description | Tasks Completed | Status |")
    lines.append("|:---:|---|:---:|---|")
    for s in row_statuses:
        lines.append(f"| **{s['row']}** | {s['desc']} | {s['done_tasks']}/5 | {s['status']} |")

    lines.append("\n---\n*Status updates automatically every 30 seconds. Attach to live console via `tmux attach -t e1_matrix_eval`.*")

    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return completed_tasks, total_tasks, is_running

def main():
    log_msg("E1 Matrix Watchdog started.")
    prev_completed = -1

    prev_running = True
    while True:
        try:
            completed, total, is_running = update_status()

            if completed != prev_completed:
                log_msg(f"Progress update: {completed}/{total} tasks complete.")
                prev_completed = completed
                # Trigger summary update if new task completed
                if os.path.exists(VENV_PYTHON):
                    subprocess.run([VENV_PYTHON, os.path.join(PROJECT_DIR, "summarize_e1_matrix.py")], capture_output=True)

            if is_running != prev_running:
                if is_running:
                    log_msg("lmms_eval process active. Evaluating tasks...")
                else:
                    log_msg("Notice: lmms_eval process is currently IDLE / stopped.")
                prev_running = is_running

            if completed == total:
                log_msg("ALL 9 ROWS COMPLETED SUCCESSFULLY!")
                update_status()
                break

            time.sleep(30)
        except KeyboardInterrupt:
            log_msg("Watchdog stopped by user.")
            break
        except Exception as e:
            log_msg(f"Watchdog error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
