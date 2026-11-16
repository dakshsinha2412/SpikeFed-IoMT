import os
import sys
import time
import paramiko
from pathlib import Path

PI_HOST = "192.168.137.2"
PI_USER = "pi"
PI_PASS = "spikefed123"
REMOTE_DIR = "/home/pi/spikefed-iomt"
PROXY = "http://192.168.137.1:8888"

FILES_TO_TRANSFER = [
    "client.py",
    "model.py",
    "metrics.py",
    "utils.py",
    "server.py",
    "main.py",
    "federated.py",
    "dataset.py",
    "visualization.py",
    "evaluate.py",
    "tests.py",
    "self_check.py",
    "benchmark.py",
    "requirements_pi.txt",
    "medical_classwise_dataset.zip",
]

SETUP_FILES = [
    "pi_setup/pi_install.sh",
    "pi_setup/start_client.sh",
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run_ssh(client, cmd, timeout=None):
    log(f"EXEC: {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    # Stream output
    for line in iter(stdout.readline, ""):
        print("  " + line.rstrip(), flush=True)
    err = stderr.read().decode()
    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0 and err:
        print(f"  [STDERR]: {err.strip()}", flush=True)
    return exit_status, err

def main():
    log("=== SpikeFed-IoMT: Raspberry Pi Complete Migration & Execution ===")
    log(f"Connecting to {PI_USER}@{PI_HOST}...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
    log("Connected successfully to Raspberry Pi.")
    
    # 0. Sync system clock so SSL certificates validate properly
    import datetime
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log(f"Synchronizing system clock on Pi to {now_str}...")
    run_ssh(ssh, f"echo {PI_PASS} | sudo -S date -s '{now_str}'")
    run_ssh(ssh, "date")
    
    # 1. Create remote directory layout
    log("Creating remote directories...")
    ssh.exec_command(f"mkdir -p {REMOTE_DIR}/pi_setup {REMOTE_DIR}/checkpoints {REMOTE_DIR}/results")
    
    # 2. Transfer files via SFTP
    log("Starting SFTP transfer of complete project files...")
    sftp = ssh.open_sftp()
    
    for f in FILES_TO_TRANSFER:
        if os.path.exists(f):
            log(f"  Uploading {f} ({os.path.getsize(f):,} bytes)...")
            sftp.put(f, f"{REMOTE_DIR}/{f}")
        else:
            log(f"  WARNING: Local file {f} not found!")
            
    for sf in SETUP_FILES:
        if os.path.exists(sf):
            fname = os.path.basename(sf)
            log(f"  Uploading {sf}...")
            sftp.put(sf, f"{REMOTE_DIR}/pi_setup/{fname}")
            
    sftp.close()
    log("All files transferred successfully.")
    
    # Make shell scripts executable
    ssh.exec_command(f"chmod +x {REMOTE_DIR}/pi_setup/*.sh")
    
    # 3. Extract medical dataset
    log("Extracting medical dataset on Raspberry Pi...")
    code, err = run_ssh(ssh, f"python3 -m zipfile -e {REMOTE_DIR}/medical_classwise_dataset.zip {REMOTE_DIR}/medical_dataset")
    if code != 0:
        log(f"Error extracting dataset: {err}")
    run_ssh(ssh, f"ls -l {REMOTE_DIR}/medical_dataset")
    
    # 4. Virtual environment & Dependency Setup
    log("Setting up Python virtual environment on Pi...")
    run_ssh(ssh, f"python3 -m venv {REMOTE_DIR}/.venv")
    
    log("Installing PyTorch and ML dependencies (via bridge proxy)...")
    TRUSTED = "--trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host www.piwheels.org"
    install_cmd = (
        f"export http_proxy={PROXY}; export https_proxy={PROXY}; "
        f"source {REMOTE_DIR}/.venv/bin/activate && "
        f"pip install {TRUSTED} --upgrade pip && "
        f"pip install {TRUSTED} torch torchvision --index-url https://download.pytorch.org/whl/cpu && "
        f"pip install {TRUSTED} snntorch flwr Pillow numpy psutil"
    )
    code, err = run_ssh(ssh, install_cmd)
    if code != 0:
        log(f"Dependency installation notice/error: {err}")
        
    # 5. Verify Environment
    log("Verifying environment and running unit tests...")
    test_cmd = (
        f"source {REMOTE_DIR}/.venv/bin/activate && "
        f"cd {REMOTE_DIR} && "
        f"python3 tests.py"
    )
    code, _ = run_ssh(ssh, test_cmd)
    if code == 0:
        log("Unit tests PASSED on Raspberry Pi!")
    else:
        log("Unit tests failed!")
        
    # 6. Execute complete Federated Learning pipeline on Medical Dataset
    log("Executing Federated SNN Training with Medical Dataset on Pi...")
    fl_cmd = (
        f"source {REMOTE_DIR}/.venv/bin/activate && "
        f"cd {REMOTE_DIR} && "
        f"python3 main.py --data-dir medical_dataset --rounds 3 --num-clients 2 --label pi_execution"
    )
    code, _ = run_ssh(ssh, fl_cmd)
    
    # 7. Execute Hardware Performance & Resource Benchmark
    log("Executing Hardware Performance and Resource Benchmark on Pi...")
    bench_cmd = (
        f"source {REMOTE_DIR}/.venv/bin/activate && "
        f"cd {REMOTE_DIR} && "
        f"python3 benchmark.py --data-dir medical_dataset --rounds 3 --num-clients 2 --label pi_benchmark"
    )
    code, _ = run_ssh(ssh, bench_cmd)
    
    # 8. Retrieve generated results from Pi to PC
    log("Retrieving generated results and checkpoints from Pi...")
    sftp = ssh.open_sftp()
    try:
        remote_results = sftp.listdir(f"{REMOTE_DIR}/results")
        for rf in remote_results:
            local_dest = f"results/{rf}"
            log(f"  Downloading results/{rf}...")
            sftp.get(f"{REMOTE_DIR}/results/{rf}", local_dest)
    except Exception as e:
        log(f"  Error reading remote results: {e}")
        
    try:
        remote_ckpts = sftp.listdir(f"{REMOTE_DIR}/checkpoints")
        for cf in remote_ckpts:
            local_dest = f"checkpoints/{cf}"
            log(f"  Downloading checkpoints/{cf}...")
            sftp.get(f"{REMOTE_DIR}/checkpoints/{cf}", local_dest)
    except Exception as e:
        log(f"  Error reading remote checkpoints: {e}")
        
    sftp.close()
    ssh.close()
    log("=== Raspberry Pi Execution and Transfer Complete! ===")

if __name__ == "__main__":
    main()
