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
import time

from collections import namedtuple
from enum import Enum


class TestRunningMode(Enum):
    """Enum indicating the running mode of the test harness."""

    CLIENT = "CLIENT"
    CLASS_SERVER = "CLASS_SERVER"
    GLOBAL_SERVER = "GLOBAL_SERVER"


util_dir = os.path.dirname(os.path.realpath(__file__))

##################
# Misc utilities #
##################


def get_arkouda_home():
    """
    Returns the path to Arkouda's root/home directory. Either walks up the tree
    or uses ARKOUDA_HOME if set.

    :return: string of path to arkouda home directory, defaults to ARKOUDA_HOME
    :rtype: str
    """
    arkouda_home = os.getenv("ARKOUDA_HOME")
    if not arkouda_home:
        dirname = os.path.dirname
        arkouda_home = dirname(dirname(util_dir))
    return arkouda_home


def get_arkouda_server():
    """
    Returns the path to the Arkouda server (ARKOUDA_HOME + arkouda_server).

    :return: string of path to arkouda_server binary
    :rtype: str
    """
    return os.path.join(get_arkouda_home(), "arkouda_server")


def is_multilocale_arkouda():
    """
    Checks if the arkouda server was compiled for multiple locales (runs
    `arkouda_server --about` and parses the CHPL_COMM value.

    :return: boolean indicating whether arkouda_server is multilocale
    :rtype: bool
    """
    out = subprocess.check_output([get_arkouda_server(), "--about"], encoding="utf-8")
    return "CHPL_COMM: none" not in out


def get_arkouda_numlocales():
    """
    Returns a default number of locales to use. 2 if Arkouda has multilocale
    support, 1 otherwise. Can be overridden with ARKOUDA_NUMLOCALES.

    :return: number of locales
    :rtype: int
    """
    if is_multilocale_arkouda():
        return int(os.getenv("ARKOUDA_NUMLOCALES", 2))
    else:
        return 1


def get_arkouda_server_info_file():
    """
    Returns the name of a file to store connection information for the server.
    Defaults to ARKOUDA_HOME + ak-server-info, but can be overridden with
    ARKOUDA_SERVER_CONNECTION_INFO

    :return: server connection info file name as a string
    :rtype: str
    """
    dflt = os.path.join(get_arkouda_home(), "ak-server-info")
    return os.getenv("ARKOUDA_SERVER_CONNECTION_INFO", dflt)


def _server_output_to_string(p):
    """Returns the annotated stdout and stderr, when available from `p`, as a string."""

    def s2s(stream, name):
        return (
            f"\n  server {name} is shown between angle brackets: <<<\n"
            + f"{stream.read().decode(errors='backslashreplace')}>>>"
        )

    if p.stdout:
        if p.stderr:
            return s2s(p.stdout, "stdout") + s2s(p.stderr, "stderr")
        else:
            return s2s(p.stdout, "output")
    else:
        if p.stderr:
            return s2s(p.stderr, "output")
        else:
            return ""


def read_server_and_port_from_file(server_connection_info, process=None, server_cmd=None):
    """
    Reads the server hostname and port from a file, which must contain
    'hostname port'. Sleeps if the file doesn't exist or formatting was off (so
    you can have clients wait for the server to start running.)

    :return: tuple containing hostname and port
    :rtype: Tuple
    :raise: RuntimeError if Arkouda server is not running
    """
    while True:
        try:
            with open(server_connection_info, "r") as f:
                (hostname, port, connect_url) = f.readline().split(" ")
                port = int(port)
                if hostname == socket.gethostname():
                    hostname = "localhost"
                return (hostname, port, connect_url)
        except (ValueError, FileNotFoundError):
            time.sleep(1)
            if process is not None and process.poll() is not None:
                logging.error(
                    "Arkouda server exited without creating the connection file"
                    + f"\n  exit code: {str(process.returncode)}"
                    + (f"\n  launch command was: {str(server_cmd)}" if server_cmd else "")
                    + _server_output_to_string(process)
                )
                raise RuntimeError("Arkouda server exited without creating the connection file")
            continue


def get_default_temp_directory():
    """Get the default temporary directory for arkouda server and client."""
    dflt = os.getcwd()
    return os.getenv("ARKOUDA_DEFAULT_TEMP_DIRECTORY", dflt)


####################
# Server utilities #
####################

"""
Global information about the server's hostname, port, and subprocess. This is
global so that the clients don't have to pass the information around manually.
"""
ServerInfo = namedtuple("ServerInfo", "host port process")
_server_info = ServerInfo(None, None, None)


def set_server_info(info):
    """
    Sets global _server_info attribute

    :return: None
    """
    global _server_info
    _server_info = info


def get_server_info():
    """
    Returns the ServerInfo tuple encapsulating arkouda_server process state

    :return: arkouda_server process state
    :rtype: ServerInfo
    :raise: ValueError if Arkouda server is not running
    """
    if _server_info.host is None or _server_info.process.poll() is not None:
        raise ValueError("Arkouda server is not running")
    return _server_info


def kill_server(server_process):
    """
    Kill a running server. Tries to shutdown cleanly with a call to
    `arkouda.shutdown()`, but if that fails calls `kill()` on the subprocess.

    :return: None
    """
    if server_process.poll() is None:
        try:
            logging.info("Attempting clean server shutdown")
            stop_arkouda_server()
        except ValueError:
            pass

        if server_process.poll() is None:
            logging.warn("Attempting dirty server shutdown")
            server_process.kill()


def get_server_launch_cmd(numlocales):
    """Get an srun command to launch ./arkouda_server_real directly."""
    import re

    # get the srun command for 'arkouda_server_real'
    p = subprocess.Popen(["./arkouda_server", f"-nl{numlocales}", "--dry-run"], stdout=subprocess.PIPE)
    srun_cmd, err = p.communicate()
    srun_cmd = srun_cmd.decode()

    if err is not None:
        raise RuntimeError("failed to capture arkouda srun command: ", err)

    # remove and capture the '--constraint=' argument if present
    constraint_setting = None
    m = re.search(r"--constraint=[\w,]*\s", srun_cmd)
    if m is not None:
        constraint_setting = srun_cmd[m.start() : m.end()]
        srun_cmd = srun_cmd[: m.start()] + srun_cmd[m.end() + 1 :]

    # extract evironment variable settings specified in the command
    # and include them in the executing environment
    env = os.environ.copy()
    max_env_idx = 0
    for match in re.finditer(r"([A-Z_]+)=(\S+)", srun_cmd):
        max_env_idx = max(max_env_idx, match.end())
        env.update({match.group(1): match.group(2)})

    # remove the environment variables from the command string
    srun_cmd = srun_cmd[max_env_idx:]

    return (srun_cmd, env, constraint_setting)


def start_arkouda_server(
    numlocales,
    trace=False,
    port=5555,
    host=None,
    server_args=None,
    within_slurm_alloc=False,
):
    """
    Start the Arkouda server and wait for it to start running. Connection info
    is written to `get_arkouda_server_info_file()`.

    :param int numlocals: the number of arkouda_server locales
    :param bool trace: indicates whether to start the arkouda_server with tracing
    :param int port: the desired arkouda_server port, defaults to 5555
    :param str host: the host that arkouda_server will run on, if known, None (default) otherwise
    :param list server_args: additional arguments to pass to the server
    :param within_slurm_alloc: whether the current script is running within a slurm allocation.
                               in which case, special care needs to be taken when launching the server.
    :return: tuple containing server host, port, and process
    :rtype: ServerInfo(host, port, process)
    """
    connection_file = get_arkouda_server_info_file()
    with contextlib.suppress(FileNotFoundError):
        os.remove(connection_file)

    launch_prefix = os.getenv("ARKOUDA_SERVER_LAUNCH_PREFIX", default="")

    if within_slurm_alloc:
        raw_server_cmd, env, _ = get_server_launch_cmd(numlocales)
        raw_server_cmd = raw_server_cmd.strip().strip().split(" ")
    else:
        raw_server_cmd = [
            get_arkouda_server(),
        ]
        env = None

    cmd = (
        launch_prefix.split()
        + raw_server_cmd
        + [
            "--trace={}".format("true" if trace else "false"),
            "--serverConnectionInfo={}".format(connection_file),
            "-nl",
            "{}".format(numlocales),
            "--ServerPort={}".format(port),
            "--logChannel=LogChannel.FILE",
        ]
    )

    if server_args:
        cmd += server_args

    logging.info('Starting "{}"'.format(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    atexit.register(kill_server, process)

    if not host:
        """
        If host is None, this means the host and port are to be retrieved
        via the read_server_and_port_from_file method
        """
        requested_port = port
        host, port, connect_url = read_server_and_port_from_file(
            connection_file, process=process, server_cmd=cmd
        )
        if str(port) != str(requested_port):
            logging.error(f"requested port {requested_port}, got {port}")

    server_info = ServerInfo(host, port, process)
    set_server_info(server_info)
    return server_info


def stop_arkouda_server():
    """
    Shutdown the Arkouda server.

    :return: None
    """
    server_process = get_server_info().process
    try:
        run_client(os.path.join(util_dir, "shutdown.py"), timeout=60)
        server_process.wait(5)
    except subprocess.TimeoutExpired:
        server_process.kill()


####################
# Client utilities #
####################


def get_client_timeout():
    """
    Get the timeout for clients. $ARKOUDA_CLIENT_TIMEOUT if set, otherwise None

    :return: client timeout in seconds
    :rtype: int
    """
    if os.getenv("ARKOUDA_CLIENT_TIMEOUT"):
        return int(os.getenv("ARKOUDA_CLIENT_TIMEOUT"))
    return None


def run_client(client, client_args=None, timeout=get_client_timeout()):
    """
    Run a client program using an already started server and return the output.
    This is a thin wrapper over subprocess.check_output.

    :return: stdout string
    :rtype: str
    """
    server_info = get_server_info()
    cmd = ["python3", client, server_info.host, str(server_info.port)]
    if client_args:
        cmd += client_args
    logging.info('Running client "{}"'.format(cmd))
    out = subprocess.check_output(cmd, encoding="utf-8", timeout=timeout)
    return out


def run_client_live(client, client_args=None, timeout=get_client_timeout()):
    """
    Run a client program using an already started server. Output is sent to the
    terminal and the exit code is returned. This is a subprocess.check_call
    shim that returns the error.returncode instead of raising an exception.

    :return: exit code or CallProcessError.returncode
    :rtype: int
    """
    server_info = get_server_info()
    cmd = ["python3", client, server_info.host, str(server_info.port)]
    if client_args:
        cmd += client_args
    logging.info('Running client "{}"'.format(cmd))
    try:
        subprocess.check_call(cmd, encoding="utf-8", timeout=timeout)
        return 0
    except subprocess.TimeoutExpired:
        return 1
    except subprocess.CalledProcessError as e:
        return e.returncode
