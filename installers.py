import os
import shutil
import sys
from subprocess import PIPE, Popen, TimeoutExpired

from setuptools.command.build_py import build_py


class ArkoudaBuildError(Exception):
    pass


def chpl_installed():
    """Check to see if chapel is installed and sourced"""
    try:
        if os.environ["CHPL_HOME"]:
            print("Existing Chapel install detected")
            return True
        return False
    except KeyError:
        return False


def make_arkouda_server():
    """
    Call make to build to Arkouda server
    has a timeout of an hour.

    """
    proc = Popen(["make"], shell=True, stdout=PIPE, stderr=PIPE)
    print("Installing Arkouda server...")

    try:
        out, err = proc.communicate(timeout=3600)
        exitcode = proc.returncode
        if exitcode != 0:
            print(err.decode("utf-8"))
            raise ArkoudaBuildError("Error building Arkouda")

    # if build does not complete in an hour
    except TimeoutExpired:
        out, errs = proc.communicate()
        proc.kill()
        print(err.decode("utf-8"))
        raise ArkoudaBuildError("Error building Arkouda")


def install_in_py_prefix():
    """Move the chpl compiled arkouda_server executable to the current python prefix"""
    prefix_target = os.path.join(os.path.abspath(sys.prefix), "bin", "arkouda_server")
    if os.path.isfile("arkouda_server"):
        # Overwrite existing executable, if any
        if os.path.isfile(prefix_target):
            os.remove(prefix_target)
        shutil.move("arkouda_server", prefix_target)
        print("Installing Arkouda server to " + prefix_target)
    else:
        raise ArkoudaBuildError("Build passed but arkouda_server not found")


def installarkouda(setup_subcommand):
    """
    Decorate a setuptools command to build and install the Arkouda server.

    This decorator wraps the `run` method of a setuptools command class to
    ensure that Chapel is installed, build the Arkouda server, and install it
    into the Python environment before executing the original command.

    Parameters
    ----------
    setup_subcommand : class
        A setuptools command class (e.g., `build_py` or `install`) whose
        `run` method will be wrapped to include Arkouda server installation.

    Returns
    -------
    class
        The same command class with its `run` method replaced by a version
        that performs the Arkouda build steps first.

    Raises
    ------
    ArkoudaBuildError
        If Chapel is not installed, or if building or installing the Arkouda
        server fails.

    """
    run_subcommand = setup_subcommand.run

    def custom_run(self):
        if not chpl_installed():
            raise ArkoudaBuildError("Chapel is not installed, Arkouda cannot be built")
        try:
            make_arkouda_server()
            install_in_py_prefix()
            run_subcommand(self)
        except ArkoudaBuildError:
            print("Exception raised in the process of building Arkouda")
            raise

    setup_subcommand.run = custom_run
    return setup_subcommand


@installarkouda
class ArkoudaInstall(build_py):
    """
    Will replace the `python setup.py install` command
    Is called when user invokes pip install Arkouda

    """

    pass
