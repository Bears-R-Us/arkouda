"""
Setup script for the Arkouda Python package.

This script uses setuptools to configure installation of the Arkouda Python client,
which provides a NumPy-like interface for distributed data analytics backed by
a Chapel-based server.

Backend Build Requirement
-------------------------
Before installing the Python package, you must first build the Chapel backend:

    make

This compiles the Arkouda server binary, which is required to connect from the Python client.
Failure to do so will result in runtime errors when attempting to use Arkouda.

Usage
-----
To install the client package after building the backend:

    pip install .

Or for development mode (editable install):

    pip install -e .

Functionality
-------------
- Defines metadata (name, version, author, etc.)
- Specifies package requirements
- Registers console scripts
- Packages Python modules and data files

See Also
--------
- Makefile       : Used to build the Chapel backend

"""

from os import path

from setuptools import find_packages, setup

import versioneer

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
    # https://packaging.python.org/specifications/core-metadata/#name
    name="arkouda",  # Required
    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=versioneer.get_version(),  # Required
    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description="Parallel, distributed NumPy-like arrays backed by Chapel",  # Optional
    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,  # Optional
    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructuredText (rst) but
    # required for plain-text or Markdown; if unspecified, "applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst" (see link below)
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url="https://github.com/Bears-R-Us/arkouda",  # Optional
    # This should be your name or the name of the organization which owns the
    # project.
    author="U.S. Government",  # Optional
    # This should be a valid email address corresponding to the author listed
    # above.
    author_email="",  # Optional
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.
        "Programming Language :: Python :: 3",
    ],
    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords="HPC workflow exploratory analysis parallel distribute arrays Chapel",  # Optional
    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=find_packages(),  # Required
    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. If you
    # do not support Python 2, you can simplify this to '>=3.5' or similar, see
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.9",
    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "numpy>=2.0",
        "pandas>=1.4.0,!=2.2.0",
        "pyzmq>=20.0.0",
        "typeguard==2.10.0",
        "tabulate",
        "pyfiglet",
        "versioneer",
        "matplotlib>=3.3.2",
        "h5py>=3.7.0",
        "pip",
        "types-tabulate",
        "tables>=3.10.0",
        "pyarrow",
        "scipy",
        "cloudpickle",
        # chapel-py
    ],
    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
        "dev": [
            "pexpect",
            "pytest>=6.0",
            "pytest-env",
            "Sphinx>=5.1.1",
            "sphinx-argparse",
            "sphinx-autoapi",
            "mypy>=0.931",
            "black==25.1.0",
            "ruff==0.11.2",
            "isort==5.13.2",
            "flake8",
            "furo",
            "myst-parser",
            "linkify-it-py",
            "mathjax",
            "sphinx-autopackagesummary",
            "sphinx-design",
            "sphinx-autodoc-typehints",
            "pandas-stubs",
            "types-python-dateutil",
            "ipython",
            "pydocstyle>=6.3.0",
            "pre-commit",
            "darglint>=1.8.1",
            "pydoclint[flake8]==0.6.6",
            "pytest-subtests",
            "numba",
            "pytest-timeout",
        ],
    },
    # replace original install command with version that also builds
    # chapel and the arkouda server.
    # cmdclass={
    #     "build_py": installers.ArkoudaInstall,
    # },
    cmdclass=versioneer.get_cmdclass(),
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    #
    # If using Python 2.6 or earlier, then these have to be included in
    # MANIFEST.in as well.
    # package_data={  # Optional
    #     'sample': ['package_data.dat'],
    # },
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],  # Optional
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    entry_points={  # Optional
        # 'console_scripts': [
        #     'sample=sample:main',
        # ],
    },
    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        "Bug Reports": "https://github.com/Bears-R-Us/arkouda/issues",
        "Source": "https://github.com/Bears-R-Us/arkouda",
        "Chapel": "https://chapel-lang.org",
    },
)
