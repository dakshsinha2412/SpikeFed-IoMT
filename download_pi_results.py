import os
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.2', username='pi', password='spikefed123', timeout=10)

sftp = client.open_sftp()
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

remote_results = sftp.listdir("/home/pi/spikefed-iomt/results")
print("Found remote results:", remote_results)
for f in remote_results:
    remote_path = f"/home/pi/spikefed-iomt/results/{f}"
    local_path = f"results/{f}"
    print(f"Downloading {remote_path} -> {local_path}...")
    sftp.get(remote_path, local_path)

remote_ckpts = sftp.listdir("/home/pi/spikefed-iomt/checkpoints")
print("Found remote checkpoints:", remote_ckpts)
for f in remote_ckpts:
    remote_path = f"/home/pi/spikefed-iomt/checkpoints/{f}"
    local_path = f"checkpoints/{f}"
    print(f"Downloading {remote_path} -> {local_path}...")
    sftp.get(remote_path, local_path)

sftp.close()
client.close()
print("All Pi artifacts successfully synced to PC!")
