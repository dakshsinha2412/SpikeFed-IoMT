"""
audit_pi.py
Audits Raspberry Pi 3B+ hardware, OS, disk, RAM, CPU temperature, throttling, and Python ML packages.
"""
import paramiko

def audit():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.137.2', username='pi', password='spikefed123', timeout=5)

    cmds = [
        ("OS / Kernel", "uname -a"),
        ("Python Version", "python3 --version"),
        ("Memory (MB)", "free -m"),
        ("Disk Space", "df -h /"),
        ("CPU Temperature", "vcgencmd measure_temp 2>/dev/null || cat /sys/class/thermal/thermal_zone0/temp"),
        ("Throttled Status", "vcgencmd get_throttled 2>/dev/null"),
        ("CPU Frequency (kHz)", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
        ("ML Environment", "source /home/pi/spikefed-iomt/.venv/bin/activate && python3 -c 'import torch, snntorch, flwr; print(\"Torch:\", torch.__version__, \"snnTorch:\", snntorch.__version__, \"Flower:\", flwr.__version__)'")
    ]

    print("=" * 60)
    print("RASPBERRY PI 3B+ HARDWARE & RUNTIME AUDIT")
    print("=" * 60)
    for label, cmd in cmds:
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        print(f"[{label}]")
        if out:
            print(out)
        if err:
            print(f"  (stderr): {err}")
        print("-" * 60)

    ssh.close()

if __name__ == "__main__":
    audit()
