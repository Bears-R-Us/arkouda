import atexit
import contextlib
import logging
import os
import socket
import subprocess
import sys
import time

from collections import namedtuple

import arkouda

def get_arkouda_home():
    arkouda_home = os.getenv('ARKOUDA_HOME')
    if not arkouda_home:
        dirname = os.path.dirname
        arkouda_home = dirname(dirname(dirname(os.path.realpath(__file__))))
    return arkouda_home

def get_arkouda_server():
    return os.path.join(get_arkouda_home(), 'arkouda_server')

def is_multilocale_arkouda():
    out = subprocess.check_output([get_arkouda_server(), '--about'], encoding='utf-8')
    return 'CHPL_COMM: none' not in out

def get_arkouda_numlocales():
    if is_multilocale_arkouda():
        return int(os.getenv('ARKOUDA_NUMLOCALES', 2))
    else:
        return 1

def get_arkouda_server_info_file():
    dflt = os.path.join(get_arkouda_home(), 'ak-server-info')
    return os.getenv('ARKOUDA_SERVER_CONNECTION_INFO', dflt)

def read_server_and_port_from_file(server_connection_info):
    while True:
        try:
            with open(server_connection_info, 'r') as f:
                (hostname, port) = f.readline().split(' ')
                port = int(port)
                if hostname == socket.gethostname():
                    hostname = 'localhost'
                return (hostname, port)
        except (ValueError, FileNotFoundError) as e:
            time.sleep(1)
            continue


ServerInfo = namedtuple('ServerInfo', 'host port process')
_server_info = ServerInfo(None, None, None)

def set_server_info(info):
    global _server_info
    _server_info = info

def get_server_info():
    if _server_info.host is None or _server_info.process.poll() is not None:
        raise ValueError('Arkouda server is not running')
    return _server_info

def kill_server(server_process):
    if server_process.poll() is None:
        try:
            logging.info('Attemping clean server shutdown')
            stop_arkouda_server()
        except ValueError as e:
            pass

        if server_process.poll() is None:
            logging.warn('Attemping dirty server shutdown')
            server_process.kill()

def start_arkouda_server(numlocales, verbose=False, log=False):
    connection_file = get_arkouda_server_info_file()
    with contextlib.suppress(FileNotFoundError):
        os.remove(connection_file)

    cmd = [get_arkouda_server(),
           '--logging={}'.format('true' if log else 'false'),
           '--v={}'.format('true' if verbose else 'false'),
           '--serverConnectionInfo={}'.format(connection_file),
           '-nl {}'.format(numlocales)]

    logging.info('Starting "{}"'.format(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
    atexit.register(kill_server, process)

    (host, port) = read_server_and_port_from_file(connection_file)
    set_server_info(ServerInfo(host, port, process))

def stop_arkouda_server():
    server_info = get_server_info()
    logging.info('Stopping the arkouda server')
    arkouda.connect(server_info.host, server_info.port)
    arkouda.shutdown()
    server_info.process.wait(20)


def run_client(client_args):
    server_info = get_server_info()
    cmd = client_args + [server_info.host, str(server_info.port)]
    logging.info('Running client "{}"'.format(cmd))
    out = subprocess.check_output(cmd, encoding='utf-8')
    return out

def run_client_live(client_args):
    server_info = get_server_info()
    cmd = client_args + [server_info.host, str(server_info.port)]
    logging.info('Running client "{}"'.format(cmd))
    try:
        subprocess.check_call(cmd, encoding='utf-8')
        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode
