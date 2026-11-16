import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.2', username='pi', password='spikefed123', timeout=10)

# Set CPU power management: run on 2 cores to eliminate brownouts
cmd = """
echo spikefed123 | sudo -S sh -c '
echo 0 > /sys/devices/system/cpu/cpu3/online 2>/dev/null || true
echo 0 > /sys/devices/system/cpu/cpu2/online 2>/dev/null || true
echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
echo powersave > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor 2>/dev/null || true
'
cat /sys/devices/system/cpu/online
vcgencmd get_throttled 2>/dev/null || echo "no vcgencmd"
"""
stdin, stdout, stderr = client.exec_command(cmd)
print("CPU STATUS:\n", stdout.read().decode())
client.close()
