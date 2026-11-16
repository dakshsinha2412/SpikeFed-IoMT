#!/bin/bash
# =============================================================================
# pi_install.sh – SpikeFed-IoMT Raspberry Pi Dependency Installer
# =============================================================================
# Run this ON THE PI after deploy_to_pi.ps1 has copied the project files.
# The deploy script calls this automatically; you can also run it manually:
#
#   bash ~/spikefed-iomt/pi_setup/pi_install.sh ~/spikefed-iomt
#
# The script:
#   1. Creates (or reuses) a venv at <project_dir>/.venv
#   2. Installs PyTorch (CPU build for ARM)
#   3. Installs snnTorch and Flower
#   4. Verifies the imports work
# =============================================================================

set -euo pipefail

PROJECT_DIR="${1:-$HOME/spikefed-iomt}"
VENV_DIR="$PROJECT_DIR/.venv"
REQ_FILE="$PROJECT_DIR/requirements_pi.txt"

echo ""
echo "====================================================="
echo "  SpikeFed-IoMT  –  Raspberry Pi Setup"
echo "  Project dir : $PROJECT_DIR"
echo "  Venv        : $VENV_DIR"
echo "====================================================="
echo ""

# ── system Python check ───────────────────────────────────────────────────
PY=$(command -v python3 || true)
if [ -z "$PY" ]; then
    echo "ERROR: python3 not found.  Install it with:"
    echo "  sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    exit 1
fi
echo "[1/4] Python3 found: $($PY --version)"

# ── virtual environment ───────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/4] Creating virtual environment at $VENV_DIR …"
    python3 -m venv "$VENV_DIR"
else
    echo "[2/4] Virtual environment already exists – reusing."
fi

# Activate venv for the rest of this script
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── PyTorch (CPU / ARM) ───────────────────────────────────────────────────
echo "[3/4] Installing packages …"

# Raspberry Pi 3 (ARMv8/aarch64 with 64-bit OS) or ARMv7 (32-bit OS).
# The official PyTorch index carries ARM64 wheels; fall back to --no-cache-dir
# to avoid memory issues during build-from-source, then try piwheels.
ARCH=$(uname -m)
echo "  Architecture detected: $ARCH"

install_torch() {
    if pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
        --quiet 2>/dev/null; then
        echo "  PyTorch installed from pytorch.org/whl/cpu"
        return 0
    fi

    echo "  pytorch.org wheel not available – trying piwheels …"
    if pip install torch torchvision \
        --extra-index-url https://www.piwheels.org/simple \
        --no-cache-dir --quiet 2>/dev/null; then
        echo "  PyTorch installed from piwheels"
        return 0
    fi

    echo "  piwheels failed – installing from PyPI (may build from source, slow) …"
    pip install torch torchvision --no-cache-dir --quiet
}

install_torch

# snnTorch and Flower (no heavy build needed)
if [ -f "$REQ_FILE" ]; then
    pip install -r "$REQ_FILE" --quiet
else
    echo "  requirements_pi.txt not found – installing defaults …"
    pip install snntorch flwr --quiet
fi

# ── verify imports ────────────────────────────────────────────────────────
echo "[4/4] Verifying package imports …"
python3 - <<'PYCHECK'
import torch, snntorch, flwr
print(f"  torch   = {torch.__version__}")
print(f"  snntorch= {snntorch.__version__}")
print(f"  flwr    = {flwr.__version__}")
PYCHECK

echo ""
echo "====================================================="
echo "  Setup complete!"
echo "  To start the client:"
echo "    source $VENV_DIR/bin/activate"
echo "    bash $PROJECT_DIR/pi_setup/start_client.sh <SERVER_IP> <CLIENT_ID>"
echo "====================================================="
