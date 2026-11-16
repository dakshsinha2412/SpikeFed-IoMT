import paramiko
import sys

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("192.168.137.2", username="pi", password="spikefed123")
    
    cmd = "uname -a; python3 --version; which pip3 || echo 'no pip'; free -h; df -h /"
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    print("OUTPUT:\n", out)
    if err:
        print("ERROR:\n", err)
    client.close()

if __name__ == "__main__":
    main()
