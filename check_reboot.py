import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.2', username='pi', password='spikefed123', timeout=10)
stdin, stdout, stderr = client.exec_command('uptime; journalctl -b -1 -e -n 20 2>/dev/null || uptime')
print(stdout.read().decode())
client.close()
