import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.2', username='pi', password='spikefed123', timeout=10)
stdin, stdout, stderr = client.exec_command("uptime; free -m; /home/pi/spikefed-iomt/.venv/bin/python -m pip list")
print(stdout.read().decode())
client.close()
