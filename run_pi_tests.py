import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.2', username='pi', password='spikefed123', timeout=10)

cmd = "source /home/pi/spikefed-iomt/.venv/bin/activate && cd /home/pi/spikefed-iomt && python3 tests.py"
stdin, stdout, stderr = client.exec_command(cmd)
print("TESTS OUTPUT:\n", stdout.read().decode())
err = stderr.read().decode()
if err:
    print("TESTS STDERR:\n", err)
print("Exit code:", stdout.channel.recv_exit_status())
client.close()
