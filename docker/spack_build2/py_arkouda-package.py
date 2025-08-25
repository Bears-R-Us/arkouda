# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)s

from spack.package import *
from spack_repo.builtin.build_systems.python import PythonPackage


class PyArkouda(PythonPackage):
    """This is the python client for Arkouda."""

    homepage = "https://github.com/Bears-R-Us/arkouda"

    # Updating the arkouda PyPI package is future work
    url = "https://github.com/Bears-R-Us/arkouda/archive/refs/tags/v2024.10.02.tar.gz"
    git = "https://github.com/Bears-R-Us/arkouda.git"

    # See https://spdx.org/licenses/ for a list.
    license("MIT")

    test_requires_compiler = True

    # A list of GitHub accounts to notify when the package is updated.
    # TODO: add arkouda devs github account
    maintainers("ajpotts", "arezaii")

    version("main", branch="main")

    version("2025.08.20", sha256="3e305930905397ff3a7a28a5d8cc2c9adca4194ca7f6ee51f749f427a2dea92c")
    version("2025.07.03", sha256="eb888fac7b0eec6b4f3bfa0bfe14e5c8f15b449286e84c45ba95c44d8cd3917a")
    version("2025.01.13", sha256="bb53bab92fedf43a47aadd9195eeedebe5f806d85887fa508fb5c69f2a4544ea")
    version("2024.12.06", sha256="92ca11319a9fdeeb8879afbd1e0c9c1b1d14aa2496781c1481598963d3c37b46")
    version("2024.10.02", sha256="00671a89a08be57ff90a94052f69bfc6fe793f7b50cf9195dd7ee794d6d13f23")
    version("2024.06.21", sha256="ab7f753befb3a0b8e27a3d28f3c83332d2c6ae49678877a7456f0fcfe42df51c")

    variant("dev", default=False, description="Include arkouda developer extras")

    depends_on("python@3.8:", type=("build", "run"), when="@:2024.06.21")
    depends_on("python@3.9:3.12.3", type=("build", "run"), when="@2024.10.02:")
    depends_on("py-setuptools", type="build")
    depends_on("py-numpy@1.24.1:1.99", type=("build", "run"))
    depends_on("py-pandas@1.4.0:", type=("build", "run"))
    conflicts("^py-pandas@2.2.0", msg="arkouda client not compatible with pandas 2.2.0")

    depends_on("py-pyarrow", type=("build", "run"))
    depends_on("py-pyzmq@20:", type=("build", "run"))
    depends_on("py-scipy@:1.13.1", type=("build", "run"), when="@2024.06.21:")
    depends_on("py-tables@3.7.0: +lzo +bzip2", type=("build", "run"), when="@:2024.06.21")
    depends_on("py-tables@3.8.0: +lzo +bzip2", type=("build", "run"), when="@2024.10.02:")
    depends_on("py-h5py@3.7.0:", type=("build", "run"))
    depends_on("py-matplotlib@3.3.2:", type=("build", "run"))
    depends_on("py-versioneer", type=("build"))
    depends_on("py-pyfiglet", type=("build", "run"))
    depends_on("py-typeguard@2.10:2.12", type=("build", "run"))
    depends_on("py-tabulate", type=("build", "run"))
    depends_on("py-pytest@6.0:", type=("build", "run"), when="@2024.10.02")
