# SpikeFed-IoMT – Raspberry Pi Deployment Guide

## Architecture summary

| Node | Role | What it runs |
|------|------|--------------|
| **PC** | FL Server | `main.py --mode server` – aggregates model updates from clients |
| **Pi 1** | FL Client 1 | `start_client.sh <PC_IP> 1` – trains on its own MRI shard |
| **Pi 2** | FL Client 2 | `start_client.sh <PC_IP> 2` – trains on its own MRI shard |

One Pi can still demonstrate the end-to-end networked workflow; two Pis allow comparing participating vs. absent devices and measuring real communication savings.

---

## Part 1 – First-time Pi OS Setup

Before preparing hardware, run this from the project folder on the PC to
verify the Python source and required deployment files:

```powershell
python self_check.py
```

### 1.1 Flash the SD card

1. Download [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2. Choose **Raspberry Pi OS Lite (64-bit)** as the OS.
3. Click the ⚙️ gear icon **before** writing:
   - Set hostname, e.g. `pi1`
   - Set username/password (e.g. `pi` / `spikefed`)
   - Configure Wi-Fi (needed if not using Ethernet ICS)
   - **Enable SSH**
4. Flash the card.

### 1.2 Static IP via SD card (Ethernet ICS alternative)

If you connect the Pi directly to the PC via Ethernet and share the PC's internet:

```powershell
# Insert the SD card and run from the project folder:
.\pi_setup\setup_sdcard.ps1
```

This sets the Pi's static IP to `192.168.137.2`.  
The PC's Ethernet adapter must be configured to `192.168.137.1` (Windows ICS sets this automatically when you share Wi-Fi over Ethernet).

### 1.3 Boot and verify SSH

```powershell
# Wait ~60 s after powering on, then:
ssh <YOUR_PI_USER>@192.168.137.2          # Static-IP / direct Ethernet
# or
ssh pi@<PI_IP_FROM_ROUTER>    # Router-assigned IP via Wi-Fi
```

---

## Part 2 – Dataset Preparation

Do **not** share a single dataset between Pis. Each Pi must have its own shard.

### 2.1 Create shards on the PC

```powershell
# Download and split into 2 shards (one per Pi)
python prepare_data.py --download --num_shards 2
# Creates: data_shard_1/  and  data_shard_2/
```

Each shard keeps the ImageFolder class structure:
```
data_shard_1/
  glioma/
  meningioma/
  notumor/
  pituitary/
```

### 2.2 Copy shards to each Pi

```powershell
scp -r data_shard_1 pi@192.168.137.2:~/spikefed-iomt/data
scp -r data_shard_2 pi@<PI_2_IP>:~/spikefed-iomt/data
```

---

## Part 3 – Deploy Client Code to Pi

Run the deployment script from the project folder on the PC:

```powershell
# Basic deploy (no data copy)
.\deploy_to_pi.ps1 -PiHost 192.168.137.2 -PiUser pi

# Deploy AND copy data shard in one step
.\deploy_to_pi.ps1 -PiHost 192.168.137.2 -PiUser pi -DataDir data_shard_1
```

The script:
1. SSH-connects and creates `~/spikefed-iomt/`
2. Copies `client.py`, `model.py`, `utils.py`, `requirements_pi.txt`
3. Copies `pi_setup/pi_install.sh` and `pi_setup/start_client.sh`
4. Optionally copies a local data shard
5. Runs `pi_install.sh` on the Pi (ARM-aware PyTorch install, ~5–15 min)

---

## Part 4 – Running the Experiment

### 4a. PC-side orchestrator (recommended)

```powershell
.\run_experiment.ps1 -NumPis 2 -Rounds 3
```

This opens the server in a new window and prints the exact SSH commands for each Pi.

### 4b. Manual start

**On the PC** — find your IP first:
```powershell
ipconfig    # look for Ethernet or Wi-Fi adapter IPv4 address
```

Start the server:
```powershell
python server.py --host 0.0.0.0 --port 8080 --rounds 3 --min_clients 2
```

**On Pi 1** (after server prints `Waiting for incoming connections`):
```bash
cd ~/spikefed-iomt
bash pi_setup/start_client.sh <PC_IP> 1 1.38
```

**On Pi 2:**
```bash
bash pi_setup/start_client.sh <PC_IP> 2 1.38
```

> If Windows Firewall prompts, click **Allow** for Python on private networks.  
> If port 8080 is blocked, either open it in Windows Firewall or change `--port`.

---

## Part 5 – After the Experiment

Results are automatically saved on the PC to:

```
results/
  round_results.json     ← per-round: client decisions, savings %
  simulation_summary.json  ← for simulation mode
  evaluation_<ts>.json   ← if you ran evaluate.py
  model.pt               ← final global model checkpoint
```

Run standalone evaluation on any dataset:
```powershell
python evaluate.py --data_dir data_shard_1 --checkpoint results/model.pt
```

Fill in **[RESULTS.md](RESULTS.md)** with your evidence for the project review.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ssh: connect to host … port 22: Connection refused` | SSH not enabled. Re-flash with SSH enabled in Imager settings. |
| `Connection timed out` to Pi | Check Ethernet cable; run `ping 192.168.137.2`; verify ICS is enabled. |
| `pip install torch` hangs / OOM | Pi 3 has 1 GB RAM. Use `--no-cache-dir`; `pi_install.sh` handles this. |
| `flwr` version mismatch | Ensure both PC and Pi have the same `flwr` version: `pip install flwr==1.x.x`. |
| `Data directory not found` on Pi | Copy data shard to `~/spikefed-iomt/data` before starting the client. The Pi launcher stops instead of attempting a download. |
| Port 8080 refused from Pi | Open firewall on PC: `netsh advfirewall firewall add rule name="Flower" dir=in action=allow protocol=TCP localport=8080`. |
| `torch` import fails on Pi | Run `pi_install.sh` again; it tries three ARM-compatible install paths. |
