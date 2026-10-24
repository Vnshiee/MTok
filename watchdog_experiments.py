#!/usr/bin/env python3
"""
Watchdog for Experiments 2 to 5 Overnight Run.
Monitors the tmux session 'exp_overnight', tracks GPU memory usage,
checks for CUDA Out-Of-Memory (OOM) or any fatal crashes, and logs
real-time status to a lightweight JSON file.
"""

import os
import sys
import time
import json
import subprocess
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "eval_outputs/experiments_2_to_5_execution.log")
STATUS_FILE = os.path.join(BASE_DIR, "eval_outputs/watchdog_status.json")
ERROR_FILE = os.path.join(BASE_DIR, "eval_outputs/watchdog_error.log")
TMUX_SESSION = "exp_overnight"

def get_gpu_memory():
    """Returns (used_mb, total_mb, free_mb) for GPU 0, or None if failed."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        parts = [int(p.strip()) for p in out.split(",")]
        return parts[0], parts[1], parts[2]
    except Exception:
        return None, None, None

def is_tmux_running(session_name):
    """Checks if the tmux session is still active."""
    try:
        out = subprocess.check_output(["tmux", "has-session", "-t", session_name], stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def check_log_for_events(last_pos):
    """Reads new lines from LOG_FILE since last_pos and looks for OOM or milestones."""
    if not os.path.exists(LOG_FILE):
        return last_pos, [], False, None

    new_lines = []
    oom_detected = False
    oom_msg = None

    with open(LOG_FILE, "r", errors="ignore") as f:
        f.seek(last_pos)
        lines = f.readlines()
        new_pos = f.tell()

    for line in lines:
        new_lines.append(line.strip())
        lower = line.lower()
        if "outofmemoryerror" in lower or "cuda out of memory" in lower:
            oom_detected = True
            oom_msg = line.strip()

    return new_pos, new_lines, oom_detected, oom_msg

def main():
    print(f"[{datetime.now().isoformat()}] Watchdog started for session '{TMUX_SESSION}'.")
    last_log_pos = 0
    min_free_mb = 999999
    current_exp = "Initializing"
    current_task = "Starting"
    completed_tasks = 0
    start_time = time.time()

    # Initial status
    status_data = {
        "status": "RUNNING",
        "start_time": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "current_experiment": current_exp,
        "current_task": current_task,
        "completed_tasks": completed_tasks,
        "gpu_used_mb": 0,
        "gpu_free_mb": 0,
        "min_gpu_free_mb": 0,
        "oom_detected": False,
        "last_log_snippet": "",
    }

    with open(STATUS_FILE, "w") as f:
        json.dump(status_data, f, indent=2)

    consecutive_inactive_checks = 0

    while True:
        now_str = datetime.now().isoformat()
        used_mb, total_mb, free_mb = get_gpu_memory()

        if free_mb is not None:
            if free_mb < min_free_mb:
                min_free_mb = free_mb

        # Check log updates
        last_log_pos, new_lines, oom_found, oom_msg = check_log_for_events(last_log_pos)

        if oom_found:
            status_data["status"] = "OOM_ERROR"
            status_data["oom_detected"] = True
            status_data["error_message"] = oom_msg
            status_data["last_update"] = now_str
            with open(ERROR_FILE, "a") as f:
                f.write(f"[{now_str}] CRITICAL: CUDA Out of Memory detected!\n{oom_msg}\n")
            with open(STATUS_FILE, "w") as f:
                json.dump(status_data, f, indent=2)
            print(f"[{now_str}] OOM Detected: {oom_msg}", file=sys.stderr)

        # Parse milestones
        for line in new_lines:
            if ">>> [EXPERIMENT 2]" in line:
                current_exp = "Exp 2: Stage 3 GRPO vs SFT"
            elif ">>> [EXPERIMENT 3]" in line:
                current_exp = "Exp 3: Direct SAT SFT Baseline"
            elif ">>> [EXPERIMENT 4]" in line:
                current_exp = "Exp 4: Thinking Budget Scaling"
            elif ">>> [EXPERIMENT 5]" in line:
                current_exp = "Exp 5: Latent Dynamics Analysis"
            elif ">>> Compiling Master Summary" in line:
                current_exp = "Compiling Master Summary"
            elif "ALL EXPERIMENTS 2 TO 5 PIPELINE COMPLETE!" in line:
                current_exp = "ALL EXPERIMENTS COMPLETE"
                status_data["status"] = "COMPLETED"

            if "Running Task:" in line:
                m = re.search(r"Running Task:\s*(\S+)", line)
                if m:
                    current_task = m.group(1)
            elif "Value" in line and "|" in line:
                completed_tasks += 1

        tmux_active = is_tmux_running(TMUX_SESSION)

        if not tmux_active:
            consecutive_inactive_checks += 1
            if consecutive_inactive_checks >= 3:
                # If finished successfully
                if status_data.get("status") != "COMPLETED":
                    status_data["status"] = "TERMINATED"
                status_data["last_update"] = now_str
                with open(STATUS_FILE, "w") as f:
                    json.dump(status_data, f, indent=2)
                print(f"[{now_str}] Tmux session '{TMUX_SESSION}' has exited. Watchdog terminating.")
                break
        else:
            consecutive_inactive_checks = 0

        # Update status file
        status_data["last_update"] = now_str
        status_data["current_experiment"] = current_exp
        status_data["current_task"] = current_task
        status_data["completed_tasks"] = completed_tasks
        status_data["gpu_used_mb"] = used_mb
        status_data["gpu_free_mb"] = free_mb
        status_data["min_gpu_free_mb"] = min_free_mb
        status_data["elapsed_minutes"] = round((time.time() - start_time) / 60.0, 1)
        if new_lines:
            status_data["last_log_snippet"] = new_lines[-1][:200]

        with open(STATUS_FILE, "w") as f:
            json.dump(status_data, f, indent=2)

        # Sleep interval (20 seconds between checks)
        time.sleep(20)

if __name__ == "__main__":
    main()
