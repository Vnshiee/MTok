#!/bin/bash
# ==============================================================================
# Environment Setup Script for Mull-Tokens (MTok)
# ==============================================================================
# Sets up Python dependencies, custom transformers fork, and evaluation harness.
# Compatible with Ubuntu Linux, CUDA 12+, Python 3.10+.
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

echo "=============================================================================="
echo "Setting up Mull-Tokens (MTok) Research Environment"
echo "Project Directory: ${PROJECT_DIR}"
echo "=============================================================================="

# 1. Virtual Environment Setup (Optional if already inside an active virtualenv)
if [ -z "${VIRTUAL_ENV:-}" ]; then
    VENV_NAME=".venv"
    if [ ! -d "${VENV_NAME}" ] && [ ! -d ".mtok" ]; then
        echo "[1/5] Creating Python virtual environment in ./${VENV_NAME}..."
        python3 -m venv "${VENV_NAME}"
        source "${VENV_NAME}/bin/activate"
    elif [ -d ".mtok" ]; then
        echo "[1/5] Found existing .mtok environment. Activating..."
        source ".mtok/bin/activate"
    else
        echo "[1/5] Found existing ./${VENV_NAME}. Activating..."
        source "${VENV_NAME}/bin/activate"
    fi
else
    echo "[1/5] Active virtual environment detected: ${VIRTUAL_ENV}"
fi

# 2. Upgrade pip and build tools
echo "[2/5] Upgrading build tools..."
pip install --upgrade pip setuptools wheel

# 3. Clone and install required custom Transformers fork (Video-R1)
if [ ! -d "Video-R1" ]; then
    echo "[3/5] Cloning custom transformers fork from arijitray1993/Video-R1..."
    git clone https://github.com/arijitray1993/Video-R1.git
else
    echo "[3/5] Video-R1 already present."
fi

echo "Installing custom transformers fork in editable mode..."
pip install -e Video-R1/transformers-main

# 4. Install lmms-eval in editable mode
echo "[4/5] Installing patched lmms-eval harness..."
pip install -e mull/lmms-eval

# 5. Install required evaluation and inference dependencies
echo "[5/5] Installing auxiliary dependencies..."
pip install qwen-vl-utils==0.0.14 \
    accelerate>=1.0.0 \
    datasets \
    evaluate \
    pytablewriter \
    decord \
    sqlitedict \
    tiktoken \
    pydantic \
    loguru \
    tenacity \
    openai \
    sacrebleu \
    scikit-learn

echo "=============================================================================="
echo "Environment setup complete!"
echo "To verify inference, run:"
echo "    python mull_infer.py"
echo "To run the core spatial benchmarks, run:"
echo "    ./run_core_benchmarks.sh"
echo "=============================================================================="
