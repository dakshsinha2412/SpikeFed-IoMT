import socket
import select
import threading

def handle_client(client_socket):
    try:
        request = client_socket.recv(8192)
        if not request:
            client_socket.close()
            return
        
        first_line = request.split(b'\n')[0]
        words = first_line.split()
        if len(words) < 2:
            client_socket.close()
            return
            
        method = words[0].decode('utf-8', errors='ignore')
        target = words[1].decode('utf-8', errors='ignore')
        
        if method.upper() == 'CONNECT':
            host, port = target.split(':')
            port = int(port)
            remote = socket.create_connection((host, port), timeout=10)
            client_socket.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            
            # Forward data bidirectionally
            sockets = [client_socket, remote]
            while True:
                r, _, _ = select.select(sockets, [], sockets, 30)
                if not r:
                    break
                for s in r:
                    data = s.recv(16384)
                    if not data:
                        return
                    out = remote if s is client_socket else client_socket
                    out.sendall(data)
        else:
            # Plain HTTP
            if target.startswith('http://'):
                target = target[7:]
            host_part = target.split('/')[0]
            if ':' in host_part:
                host, port = host_part.split(':')
                port = int(port)
            else:
                host, port = host_part, 80
            remote = socket.create_connection((host, port), timeout=10)
            remote.sendall(request)
            
            sockets = [client_socket, remote]
            while True:
                r, _, _ = select.select(sockets, [], sockets, 30)
                if not r:
                    break
                for s in r:
                    data = s.recv(16384)
                    if not data:
                        return
                    out = remote if s is client_socket else client_socket
                    out.sendall(data)
    except Exception:
        pass
    finally:
        try:
            client_socket.close()
        except Exception:
            pass

def run_proxy(host='0.0.0.0', port=8888):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(100)
    print(f'Proxy listening on {host}:{port}')
    while True:
        client, _ = server.accept()
        t = threading.Thread(target=handle_client, args=(client,), daemon=True)
        t.start()

if __name__ == '__main__':
    run_proxy()
