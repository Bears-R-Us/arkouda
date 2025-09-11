# setup.py (compat shim for Versioneer 0.19)
from setuptools import setup

import versioneer

setup(
    name="arkouda",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
