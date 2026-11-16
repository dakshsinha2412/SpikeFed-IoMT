import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.2', username='pi', password='spikefed123', timeout=10)

cmd = """
/home/pi/spikefed-iomt/.venv/bin/python3 -c "
import torch, torchvision, snntorch, flwr, numpy, PIL, psutil
print('  torch       =', torch.__version__)
print('  torchvision =', torchvision.__version__)
print('  snntorch    =', snntorch.__version__)
print('  flwr        =', flwr.__version__)
print('  numpy       =', numpy.__version__)
print('  PIL         =', PIL.__version__)
print('  psutil      =', psutil.__version__)
"
"""
stdin, stdout, stderr = client.exec_command(cmd)
print("VERIFICATION OUTPUT:\n", stdout.read().decode())
err = stderr.read().decode()
if err:
    print("ERROR:\n", err)
client.close()
