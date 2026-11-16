"""
launch_dashboard.py
Ultra-fast dual-stack launcher for SpikeFed-IoMT Research Platform.
Binds IPv6 + IPv4 dual-stack socket to eliminate Windows localhost TCP fallback latency.
"""
import socket
import uvicorn
import webbrowser
import threading
import time
from web.backend_app import app

def create_fast_socket(port=8000):
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('::', port))
        sock.listen(128)
        return sock
    except Exception:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        sock.listen(128)
        return sock

def open_browser():
    time.sleep(1.0)
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except Exception:
        pass

if __name__ == "__main__":
    print("=" * 70)
    print("  SPIKEFED-IOMT RESEARCH DEMONSTRATION PLATFORM (HIGH-SPEED DUAL-STACK)")
    print("  Brain-Inspired Federated Spiking Neural Networks for Medical IoMT")
    print("=" * 70)
    print("  Server URL (Fast Local): http://127.0.0.1:8000")
    print("  Server URL (Localhost):  http://localhost:8000")
    print("  Server URL (Network):    http://192.168.137.1:8000")
    print("  Reviewer Mode:           http://127.0.0.1:8000#reviewer")
    print("=" * 70)
    print("  Press Ctrl+C to terminate the dashboard.\n")

    threading.Thread(target=open_browser, daemon=True).start()
    sock = create_fast_socket(8000)
    config = uvicorn.Config(app, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    server.run(sockets=[sock])
