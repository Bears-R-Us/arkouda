"""Provide setuptools entry point for Versioneer's hooks (PEP 517 project)."""

from setuptools import setup

import versioneer


setup(
    name="arkouda",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
