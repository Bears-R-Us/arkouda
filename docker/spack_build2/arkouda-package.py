# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)


from spack.package import *
from spack_repo.builtin.build_systems.makefile import MakefilePackage


class Arkouda(MakefilePackage):
    """Arkouda is a NumPy-like library for distributed data with a focus on
    large-scale data science applications."""

    homepage = "https://github.com/Bears-R-Us/arkouda"

    # Arkouda does not have a current PyPI package, so we use the GitHub tarball
    url = (
        self.url_for_version()
    )  # "https://github.com/Bears-R-Us/arkouda/archive/refs/tags/v2024.10.02.tar.gz"
    git = "https://github.com/Bears-R-Us/arkouda.git"

    # See https://spdx.org/licenses/ for a list.
    license("MIT")

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

    variant(
        "distributed",
        default=False,
        description="Build Arkouda for multi-locale execution on a cluster or supercomputer",
    )

    # For Arkouda releases before July 2025, support up to Chapel 2.4.x
    depends_on(
        "chapel@2.0.0:2.4.99 +hdf5 +zmq",
        when="@:2025.01.13",
        type=("build", "link", "run", "test"),
    )

    # For Arkouda 2025.07.03 and later, allow Chapel 2.5.x too
    depends_on(
        "chapel@2.0.0:2.5.99 +hdf5 +zmq",
        when="@2025.07.03:",
        type=("build", "link", "run", "test"),
    )
    depends_on("cmake@3.13.4:", type="build")
    depends_on("python@3.9:", type=("build", "link", "run", "test"))
    depends_on("libzmq@4.2.5:", type=("build", "link", "run", "test"))
    depends_on("hdf5+hl~mpi", type=("build", "link", "run", "test"))
    depends_on("libiconv", type=("build", "link", "run", "test"))
    depends_on("libidn2", type=("build", "link", "run", "test"))
    depends_on(
        "arrow +parquet +snappy +zlib +brotli +bz2 ~lz4 +zstd",
        type=("build", "link", "run", "test"),
    )

    requires("^chapel comm=none", when="~distributed")
    requires("^chapel +python-bindings", when="@2024.10.02:")
    requires(
        "^chapel comm=gasnet",
        "^chapel comm=ugni",
        "^chapel comm=ofi",
        policy="one_of",
        when="+distributed",
    )

    # Some systems need explicit -fPIC flag when building the Arrow functions
    patch("makefile-fpic-2024.06.21.patch", when="@2024.06.21")
    patch("makefile-fpic-2024.10.02.patch", when="@2024.10.02:")

    sanity_check_is_file = [join_path("bin", "arkouda_server")]

    def check(self):
        # skip b/c we need the python client
        pass

    # override the default edit method to apply the patch
    def edit(self, spec, prefix):
        self.update_makefile_paths(spec, prefix)

    def update_makefile_paths(self, spec, prefix):
        # add to the Makefile.paths file for all of the dependencies installed by spack
        # in the form $(eval $(call add-path,<path-to-dep-aka-prefix>))
        with open("Makefile.paths", "w") as f:
            f.write("$(eval $(call add-path,{0}))\n".format(spec["hdf5"].prefix))
            f.write("$(eval $(call add-path,{0}))\n".format(spec["libzmq"].prefix))
            f.write("$(eval $(call add-path,{0}))\n".format(spec["arrow"].prefix))
            f.write("$(eval $(call add-path,{0}))\n".format(spec["libiconv"].prefix))
            f.write("$(eval $(call add-path,{0}))\n".format(spec["libidn2"].prefix))

    def build(self, spec, prefix):
        # Detect distributed builds and skip the dependency checks built into
        # the Arkouda Makefile. These checks will try to spawn multiple jobs which may
        # cause the build to fail in situations where the user is constrained
        # to a limited number of simultaneous jobs.
        if spec.satisfies("+distributed"):
            with set_env(ARKOUDA_SKIP_CHECK_DEPS="1"):
                tty.warn("Distributed build detected. Skipping dependency checks")
                make()
        else:
            make()

    # Arkouda does not have an install target in its Makefile
    def install(self, spec, prefix):
        mkdir(prefix.bin)
        install("arkouda_server", prefix.bin)
        # Arkouda can have two executables depending on if Chapel is compiled in
        # single-locale or multi-locale mode
        if spec.satisfies("+distributed"):
            install("arkouda_server_real", prefix.bin)
