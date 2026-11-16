import paramiko
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.2', username='pi', password='spikefed123', timeout=10)

cmd = (
    "source /home/pi/spikefed-iomt/.venv/bin/activate && "
    "cd /home/pi/spikefed-iomt && "
    "python3 benchmark.py --data-dir medical_dataset --rounds 3 --num-clients 2 --label pi_benchmark"
)
log(f"Executing: {cmd}")
stdin, stdout, stderr = client.exec_command(cmd)
for line in iter(stdout.readline, ""):
    print("  " + line.rstrip(), flush=True)
err = stderr.read().decode()
code = stdout.channel.recv_exit_status()
log(f"Finished with exit code: {code}")
if err:
    print("STDERR:\n", err)
client.close()
