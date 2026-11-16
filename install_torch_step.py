import paramiko
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('192.168.137.2', username='pi', password='spikefed123', timeout=10)
    
    # Enable more swap temporarily if needed: 1GB swap is already active
    TRUSTED = "--trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org --trusted-host pypi.org --trusted-host files.pythonhosted.org"
    cmd = (
        "export http_proxy=http://192.168.137.1:8888; export https_proxy=http://192.168.137.1:8888; "
        f"/home/pi/spikefed-iomt/.venv/bin/pip install {TRUSTED} --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    )
    log(f"Running: {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd)
    for line in iter(stdout.readline, ""):
        print("  " + line.rstrip(), flush=True)
    err = stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    log(f"Exit code: {code}")
    if err:
        log(f"STDERR: {err}")
    client.close()

if __name__ == "__main__":
    main()
