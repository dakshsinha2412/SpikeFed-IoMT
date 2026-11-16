#!/usr/bin/env bash
# verify_pi_environment.sh — Smoke test on Raspberry Pi 3B+
set -e

echo "=========================================================="
echo "  SPIKEFED-IOMT: RASPBERRY PI ENVIRONMENT VERIFICATION"
echo "=========================================================="

echo "[1/4] Checking Hardware & System:"
uname -a
python3 --version
free -m

echo "[2/4] Checking Throttling & CPU Clock:"
vcgencmd get_throttled || echo "get_throttled N/A"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq || echo "Cur freq N/A"

echo "[3/4] Activating Virtual Environment:"
source /home/pi/spikefed-iomt/.venv/bin/activate

echo "[4/4] Verifying ML Packages & LIF-SNN Forward Pass:"
python3 -c "
import torch
import snntorch as snn
import flwr as fl
print('Torch Version:   ', torch.__version__)
print('snnTorch Version:', snn.__version__)
print('Flower Version:  ', fl.__version__)
from model import SpikingCNN
m = SpikingCNN(num_classes=2)
x = torch.rand(2, 1, 28, 28)
spk, mem = m(x, num_steps=5)
print('Smoke test forward pass OK: spk shape', list(spk.shape))
"

echo "Raspberry Pi Environment Verified Successfully!"
