import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('192.168.137.2', username='pi', password='spikefed123', timeout=10)

cmd = """
echo spikefed123 | sudo -S bash -c '
echo 600000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
echo 600000 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_max_freq
echo powersave > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo powersave > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
echo 0 > /sys/devices/system/cpu/cpu2/online 2>/dev/null || true
echo 0 > /sys/devices/system/cpu/cpu3/online 2>/dev/null || true
'
cat /sys/devices/system/cpu/online
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
"""
stdin, stdout, stderr = client.exec_command(cmd)
print("THROTTLE OUTPUT:\n", stdout.read().decode())
client.close()
