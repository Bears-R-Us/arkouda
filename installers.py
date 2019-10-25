import os

from setuptools.command.install import install
from setuptools.command.develop import develop
from subprocess import PIPE, Popen


# define errors for build process
class ChplBuildError(Exception):
    pass

class ArkoudaBuildError(Exception):
    pass


class ChplInstaller():
    """Functions to assist in the installation for Chapel for the purposes
       of programmatically building/installing a Chapel application. """

    def install_chpl(self):
        if not self.chpl_installed():
            print("Downloading Chapel from Github...")
            self.setup_chpl()

    def chpl_installed(self):
        # assumes chapel user has configured chapel
        if os.environ["CHPL_HOME"]:
            print("Existing Chapel install detected")
            return True
        return False

    def setup_chpl(self):
        # TODO: symlink or print path to user
        process = Popen(["source scripts/setup_chpl.sh"], shell=True, stdout=PIPE, stderr=PIPE)
        try:
            out, errs = proc.communicate(timeout=40)
        except TimeoutExpired:
            proc.kill()
            out, errs = proc.communicate()
            raise ChplBuildError("Error building Chapel for Arkouda.")



def installarkouda(setup_subcommand):
   """Decorator for installing arkouda"""
   run_subcommand = setup_subcommand.run

   def make_arkouda_server():
       process = Popen(["make"], shell=True, stdout=PIPE, stderr=PIPE)
       print("Installing Arkouda server...")

       out, err = process.communicate()
       if err:
           print(err.decode("utf-8"))

   def custom_run(self):
       installer = ChplInstaller()
       installer.install_chpl()
       make_arkouda_server()
       run_subcommand(self)

   setup_subcommand.run = custom_run
   return setup_subcommand


@installarkouda
class ArkoudaDevelop(develop):
    pass

@installarkouda
class ArkoudaInstall(install):
    pass

