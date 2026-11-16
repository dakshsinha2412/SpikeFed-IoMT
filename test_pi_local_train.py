"""
test_pi_local_train.py
Connects to Pi and runs a single epoch of local SNN training on genuine medical data.
"""
import paramiko

def test_local_train():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('192.168.137.2', username='pi', password='spikefed123', timeout=5)

    remote_script = (
        "import torch, time\n"
        "from model import SpikingCNN\n"
        "from dataset import load_image_folder\n"
        "from torch.utils.data import DataLoader\n"
        "import snntorch.functional as SF\n\n"
        "ds = load_image_folder('data/ultrasound/train', normalize=False)\n"
        "loader = DataLoader(ds, batch_size=8, shuffle=True)\n"
        "model = SpikingCNN(num_classes=2, threshold=0.5)\n"
        "opt = torch.optim.Adam(model.parameters(), lr=0.003)\n"
        "crit = SF.ce_rate_loss()\n\n"
        "start = time.time()\n"
        "tot_loss, tot_spk, count = 0, 0, 0\n"
        "for x, y in loader:\n"
        "    opt.zero_grad()\n"
        "    spk, _ = model(x, num_steps=12)\n"
        "    loss = crit(spk, y)\n"
        "    loss.backward()\n"
        "    opt.step()\n"
        "    tot_loss += loss.item() * len(y)\n"
        "    tot_spk += spk.sum().item()\n"
        "    count += len(y)\n\n"
        "dur = time.time() - start\n"
        "print(f'On-Pi Ultrasound Training (threshold=0.5): {count} samples in {dur:.2f}s | Avg Loss: {tot_loss/count:.4f} | Total Spikes Fired: {tot_spk:.0f}')\n"
    )

    cmd = f"source /home/pi/spikefed-iomt/.venv/bin/activate && cd /home/pi/spikefed-iomt && python3 -c \"{remote_script}\""
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()

    print("STDOUT:", out)
    if err:
        print("STDERR:", err)

    ssh.close()

if __name__ == "__main__":
    test_local_train()
