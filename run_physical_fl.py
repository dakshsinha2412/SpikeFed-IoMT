"""
run_physical_fl.py
Orchestrates physical Federated Learning execution across:
- HP Victus Laptop (192.168.137.1) acting as central FedAvg server
- Raspberry Pi 3B+ (192.168.137.2) acting as edge client
"""
import os
import sys
import time
import socket
import threading
import subprocess
import paramiko

PI_HOST = "192.168.137.2"
PI_USER = "pi"
PI_PASS = "spikefed123"
SERVER_HOST = "192.168.137.1"
PORT = 8080
ROUNDS = 2
NUM_CLASSES = 2

combined_logs = []

def log(tag, message):
    line = f"[{time.strftime('%H:%M:%S')}] [{tag}] {message}"
    print(line, flush=True)
    combined_logs.append(line)

def wait_for_port(host, port, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False

def stream_process(proc, tag):
    for line in iter(proc.stdout.readline, ''):
        cleaned = line.rstrip()
        if cleaned:
            log(tag, cleaned)
    proc.stdout.close()

def main():
    print("=" * 70)
    print("  PHYSICAL FEDERATED LEARNING: HP VICTUS SERVER <--> RASPBERRY PI CLIENT")
    print(f"  Central Server: HP Victus Laptop @ {SERVER_HOST}:{PORT}")
    print(f"  Edge Client:    Raspberry Pi 3B+ @ {PI_HOST}")
    print(f"  Communication Rounds: {ROUNDS}")
    print("=" * 70)

    # 1. Start Server on HP Victus
    server_cmd = [
        sys.executable,
        "server.py",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--rounds", str(ROUNDS),
        "--min_clients", "1",
        "--num_classes", str(NUM_CLASSES)
    ]
    log("ORCHESTRATOR", f"Starting HP Victus FL server on 0.0.0.0:{PORT}...")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    server_thread = threading.Thread(target=stream_process, args=(server_proc, "HP-VICTUS-SERVER"), daemon=True)
    server_thread.start()

    # Wait for server to bind port
    log("ORCHESTRATOR", "Waiting for Flower server to bind port 8080...")
    if not wait_for_port("127.0.0.1", PORT, timeout=15):
        log("ERROR", "Flower server failed to start within 15 seconds.")
        server_proc.terminate()
        return 1

    log("ORCHESTRATOR", f"Flower server is listening on port {PORT}. Connecting to Raspberry Pi via SSH...")

    # 2. Connect to Raspberry Pi via SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)
    log("ORCHESTRATOR", f"SSH connected to {PI_USER}@{PI_HOST}. Launching edge client...")

    client_cmd = (
        "source /home/pi/spikefed-iomt/.venv/bin/activate && "
        "cd /home/pi/spikefed-iomt && "
        f"python3 -u client.py --server {SERVER_HOST}:{PORT} --client_id 1 --data_dir data/chest_xray/train --batch_size 6"
    )

    stdin, stdout, stderr = ssh.exec_command(client_cmd)

    def stream_ssh(stream, tag):
        for line in iter(stream.readline, ''):
            cleaned = line.rstrip()
            if cleaned:
                log(tag, cleaned)

    client_stdout_thread = threading.Thread(target=stream_ssh, args=(stdout, "PI-EDGE-CLIENT"), daemon=True)
    client_stderr_thread = threading.Thread(target=stream_ssh, args=(stderr, "PI-EDGE-STDERR"), daemon=True)
    client_stdout_thread.start()
    client_stderr_thread.start()

    # Wait for client to complete
    client_exit_code = stdout.channel.recv_exit_status()
    log("ORCHESTRATOR", f"Raspberry Pi edge client completed with exit code: {client_exit_code}")
    ssh.close()

    # Wait for server process to finish
    server_proc.wait(timeout=30)
    log("ORCHESTRATOR", f"HP Victus FL server exited with code: {server_proc.returncode}")

    # Save log output
    os.makedirs("results", exist_ok=True)
    log_path = os.path.join("results", "physical_fl_execution.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_logs) + "\n")
    log("ORCHESTRATOR", f"Complete physical execution log saved to: {log_path}")

    # Verify model checkpoint
    model_path = os.path.join("results", "model.pt")
    if os.path.exists(model_path):
        size_kb = os.path.getsize(model_path) / 1024
        log("ORCHESTRATOR", f"VERIFIED: Global model checkpoint saved at {model_path} ({size_kb:.2f} KB)")
    else:
        log("WARNING", "results/model.pt not found!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
