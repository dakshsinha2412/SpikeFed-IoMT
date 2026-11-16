import socket
import threading
import time
import paramiko

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('192.168.137.1', 8080))
s.listen(1)
print("Laptop listening on 192.168.137.1:8080...")

received_flag = []
def accept_conn():
    try:
        conn, addr = s.accept()
        print(f"Accepted connection from Pi: {addr}")
        conn.sendall(b"OK_FROM_HP_VICTUS\n")
        conn.close()
        received_flag.append(True)
    except Exception as e:
        print(f"Socket accept error: {e}")

t = threading.Thread(target=accept_conn, daemon=True)
t.start()

# SSH into Pi and probe port 8080
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.2', username='pi', password='spikefed123', timeout=5)

cmd = "python3 -c \"import socket; s = socket.create_connection(('192.168.137.1', 8080), timeout=5); print(s.recv(1024).decode())\""
stdin, stdout, stderr = client.exec_command(cmd)
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print(f"Pi probe output: '{out}'")
if err:
    print(f"Pi probe error: '{err}'")

client.close()
s.close()
if received_flag:
    print("Port 8080 communication test: SUCCESS!")
else:
    print("Port 8080 communication test: FAILED")
