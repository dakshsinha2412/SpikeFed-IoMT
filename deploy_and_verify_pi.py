"""
deploy_and_verify_pi.py
Deploys updated multi-modal dataset modules, SNN model, configs, and datasets
to the physical Raspberry Pi 3B+ over SFTP and executes verification.
"""
import os
import paramiko
from pathlib import Path

PI_HOST = "192.168.137.2"
PI_USER = "pi"
PI_PASS = "spikefed123"
REMOTE_DIR = "/home/pi/spikefed-iomt"

def sftp_put_dir(sftp, local_dir, remote_dir):
    local_dir = Path(local_dir)
    try:
        sftp.mkdir(remote_dir)
    except Exception:
        pass
    for item in local_dir.rglob("*"):
        rel = item.relative_to(local_dir)
        rem_path = f"{remote_dir}/{rel.as_posix()}"
        if item.is_dir():
            try:
                sftp.mkdir(rem_path)
            except Exception:
                pass
        else:
            sftp.put(str(item), rem_path)

def main():
    print(f"Connecting to Raspberry Pi at {PI_USER}@{PI_HOST} via SSH...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=10)

    print("Opening SFTP session...")
    sftp = ssh.open_sftp()

    # Ensure remote dirs
    for d in ["config", "data"]:
        try:
            sftp.mkdir(f"{REMOTE_DIR}/{d}")
        except Exception:
            pass

    # Transfer files
    files_to_sync = [
        "client.py",
        "model.py",
        "dataset.py",
        "dataset_preprocessing.py",
        "dataset_registry.py",
        "dataset_loader.py",
        "dataset_validation.py",
        "dataset_statistics.py",
        "federated_partition.py",
        "pi_requirements.txt",
        "verify_pi_environment.sh"
    ]

    print("Transferring core Python modules and SNN scripts...")
    for f in files_to_sync:
        if os.path.exists(f):
            sftp.put(f, f"{REMOTE_DIR}/{f}")
            print(f"  Synced: {f}")

    if os.path.exists("config/pi_client_config.json"):
        sftp.put("config/pi_client_config.json", f"{REMOTE_DIR}/config/pi_client_config.json")
        print("  Synced: config/pi_client_config.json")

    print("Selectively transferring datasets to Pi (chest_xray & ultrasound)...")
    if os.path.exists("data/chest_xray"):
        sftp_put_dir(sftp, "data/chest_xray", f"{REMOTE_DIR}/data/chest_xray")
        print("  Synced data/chest_xray")

    if os.path.exists("data/ultrasound"):
        sftp_put_dir(sftp, "data/ultrasound", f"{REMOTE_DIR}/data/ultrasound")
        print("  Synced data/ultrasound")

    sftp.close()

    print("\nExecuting verification script on Raspberry Pi...")
    cmd = (
        f"cd {REMOTE_DIR} && chmod +x verify_pi_environment.sh && "
        f"source {REMOTE_DIR}/.venv/bin/activate && ./verify_pi_environment.sh"
    )
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()

    if out:
        print(out)
    if err:
        print("STDERR:", err)

    ssh.close()
    print("\nDeployment and Pi Verification Complete!")

if __name__ == "__main__":
    main()
