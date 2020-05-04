"""
Utility functions for running the Arkouda server and clients. Typical usage
would be something like:

    start_arkouda_server(get_arkouda_numlocales())
    ret = run_client{_live}(client, client_args)
    stop_arkouda_server()

This will start the arkouda server (using 2 locales if Arkouda was built with
multilocale support, 1 otherwise) and wait for the server to start running. The
client must take `hostname port` as its first positional arguments, and there
are versions that either print the output live or return it. The server will be
stopped atexit or with a call to stop_arkouda_server().
"""

import atexit
import contextlib
import logging
import os
import socket
import subprocess
import sys
import time

from collections import namedtuple

from context import arkouda

util_dir = os.path.dirname(os.path.realpath(__file__))

##################
# Misc utilities #
##################

def get_arkouda_home():
    """
    Returns the path to Arkouda's root/home directory. Either walks up the tree
    or uses ARKOUDA_HOME if set.
    """
    arkouda_home = os.getenv('ARKOUDA_HOME')
    if not arkouda_home:
        dirname = os.path.dirname
        arkouda_home = dirname(dirname(util_dir))
    return arkouda_home

def get_arkouda_server():
    """
    Returns the path to the Arkouda server (ARKOUDA_HOME + arkouda_server).
    """
    return os.path.join(get_arkouda_home(), 'arkouda_server')

def is_multilocale_arkouda():
    """
    Checks if the arkouda server was compiled for multiple locales (runs
    `arkouda_server --about` and parses the CHPL_COMM value.
    """
    out = subprocess.check_output([get_arkouda_server(), '--about'], encoding='utf-8')
    return 'CHPL_COMM: none' not in out

def get_arkouda_numlocales():
    """
    Returns a default number of locales to use. 2 if Arkouda has multilocale
    support, 1 otherwise. Can be overridden with ARKOUDA_NUMLOCALES.
    """
    if is_multilocale_arkouda():
        return int(os.getenv('ARKOUDA_NUMLOCALES', 2))
    else:
        return 1

def get_arkouda_server_info_file():
    """
    Returns the name of a file to store connection information for the server.
    Defaults to ARKOUDA_HOME + ak-server-info, but can be overridden with
    ARKOUDA_SERVER_CONNECTION_INFO
    """
    dflt = os.path.join(get_arkouda_home(), 'ak-server-info')
    return os.getenv('ARKOUDA_SERVER_CONNECTION_INFO', dflt)

def read_server_and_port_from_file(server_connection_info):
    """
    Reads the server hostname and port from a file, which must contain
    'hostname port'. Sleeps if the file doesn't exist or formatting was off (so
    you can have clients wait for the server to start running.)
    """
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

####################
# Server utilities #
####################

"""
Global information about the server's hostname, port, and subprocess. This is
global so that the clients don't have to pass the information around manually.
"""
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
    """
    Kill a running server. Tries to shutdown cleanly with a call to
    `arkouda.shutdown()`, but if that fails calls `kill()` on the subprocess.
    """
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
    """
    Start the Arkouda server and wait for it to start running. Connection info
    is written to `get_arkouda_server_info_file()`.
    Returns ServerInfo(host, port, process)
    """
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
    server_info = ServerInfo(host, port, process)
    set_server_info(server_info)
    return server_info

def stop_arkouda_server():
    """
    Shutdown the Arkouda server.
    """
    _, _, server_process = get_server_info()
    try:
        run_client(os.path.join(util_dir, 'shutdown.py'), timeout=60)
        server_process.wait(5)
    except subprocess.TimeoutExpired as e:
        server_process.kill()

####################
# Client utilities #
####################

def get_client_timeout():
    """
    Get the timeout for clients. $ARKOUDA_CLIENT_TIMEOUT if set, otherwise None
    """
    if os.getenv('ARKOUDA_CLIENT_TIMEOUT'):
        return int(os.getenv('ARKOUDA_CLIENT_TIMEOUT'))
    return None

def run_client(client, client_args=None, timeout=get_client_timeout()):
    """
    Run a client program using an already started server and return the output.
    This is a thin wrapper over subprocess.check_output.
    """
    server_info = get_server_info()
    cmd = ['python3'] + [client] + [server_info.host, str(server_info.port)]
    if client_args:
        cmd += client_args
    logging.info('Running client "{}"'.format(cmd))
    out = subprocess.check_output(cmd, encoding='utf-8', timeout=timeout)
    return out

def run_client_live(client, client_args=None, timeout=get_client_timeout()):
    """
    Run a client program using an already started server. Output is sent to the
    terminal, and the returncode is returned. This is a subprocess.check_call
    shim that returns the returncode instead of raising an exception.
    """
    server_info = get_server_info()
    cmd = ['python3'] + [client] + [server_info.host, str(server_info.port)]
    if client_args:
        cmd += client_args
    logging.info('Running client "{}"'.format(cmd))
    try:
        subprocess.check_call(cmd, encoding='utf-8', timeout=timeout)
        return 0
    except subprocess.TimeoutExpired as e:
        return 1
    except subprocess.CalledProcessError as e:
        return e.returncode
