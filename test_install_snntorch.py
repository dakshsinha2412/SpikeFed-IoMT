import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.2', username='pi', password='spikefed123', timeout=10)

TRUSTED = "--trusted-host pypi.org --trusted-host files.pythonhosted.org"
cmd = (
    "export http_proxy=http://192.168.137.1:8888; export https_proxy=http://192.168.137.1:8888; "
    f"/home/pi/spikefed-iomt/.venv/bin/pip install {TRUSTED} --no-compile snntorch"
)
stdin, stdout, stderr = client.exec_command(cmd)
for line in iter(stdout.readline, ""):
    print("  " + line.rstrip(), flush=True)
err = stderr.read().decode()
code = stdout.channel.recv_exit_status()
print(f"Exit code: {code}")
if err:
    print(f"STDERR: {err}")
client.close()
