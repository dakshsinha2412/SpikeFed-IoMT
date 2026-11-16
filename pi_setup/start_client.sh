#!/bin/bash
# =============================================================================
# start_client.sh – SpikeFed-IoMT Pi Client Launcher
# =============================================================================
# Convenience wrapper that activates the venv and starts the Flower client.
#
# Usage (run from ~/spikefed-iomt on the Pi):
#   bash pi_setup/start_client.sh <SERVER_IP> <CLIENT_ID> [THRESHOLD] [DATA_DIR]
#
# Examples:
#   bash pi_setup/start_client.sh 192.168.137.1 1
#   bash pi_setup/start_client.sh 192.168.137.1 2 1.38 data
#
# Arguments:
#   SERVER_IP   IP address of the PC running the Flower server (required)
#   CLIENT_ID   Integer ID of this Pi, e.g. 1 or 2  (required)
#   THRESHOLD   Uncertainty threshold (optional, default: 1.38)
#   DATA_DIR    Path to local MRI shard folder (optional, default: data/)
# =============================================================================

set -euo pipefail

SERVER_IP="${1:-}"
CLIENT_ID="${2:-}"
THRESHOLD="${3:-1.38}"
DATA_DIR="${4:-data}"
PORT="${PORT:-8080}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

# ── validate required args ────────────────────────────────────────────────
if [ -z "$SERVER_IP" ] || [ -z "$CLIENT_ID" ]; then
    echo "Usage: bash pi_setup/start_client.sh <SERVER_IP> <CLIENT_ID> [THRESHOLD] [DATA_DIR]"
    echo "Example: bash pi_setup/start_client.sh 192.168.137.1 1"
    exit 1
fi

# ── activate venv ─────────────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    echo "Run the setup script first:"
    echo "  bash $PROJECT_DIR/pi_setup/pi_install.sh"
    exit 1
fi
source "$VENV_DIR/bin/activate"
echo "Venv activated: $VENV_DIR"

# ── sanity check data dir ─────────────────────────────────────────────────
DATA_PATH="$PROJECT_DIR/$DATA_DIR"
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Data directory not found at $DATA_PATH"
    echo "  Copy your MRI shard first:"
    echo "    scp -r data_shard_${CLIENT_ID} pi@<PI_IP>:$PROJECT_DIR/data"
    echo "  The Pi deployment intentionally has no kagglehub dependency."
    exit 1
else
    DATA_ARG=(--data_dir "$DATA_PATH")
fi

# ── launch client ─────────────────────────────────────────────────────────
echo ""
echo "====================================================="
echo "  SpikeFed-IoMT  –  Pi Client ${CLIENT_ID}"
echo "  Server : ${SERVER_IP}:${PORT}"
echo "  Threshold : ${THRESHOLD}"
if [ -n "${DATA_ARG}" ]; then
    echo "  Data   : ${DATA_PATH}"
fi
echo "====================================================="
echo ""

cd "$PROJECT_DIR"
python3 client.py \
    --server "${SERVER_IP}:${PORT}" \
    --client_id "${CLIENT_ID}" \
    --data_dir "${DATA_PATH}"
