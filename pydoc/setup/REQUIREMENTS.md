# Requirements

## Dependency List

The installation instructions for the dependencies listed here may vary depending on your preferred operating system. Refer to the [Installation Section](install_menu.rst) for more information.

- ``Chapel 1.30.0 or later``
- `cmake>=3.13.4`
- `zeromq>=4.2.5`
- `hdf5`
- `python>=3.8`
- `iconv`
- `idn2`
- `Arrow`

## Python Dependencies

The following python packages are required by the Arkouda client package.

- `python>=3.8`
- `numpy>=1.24.1`
- `pandas>=1.4.0,<2.2.0`
- `pyzmq>=20.0.0`
- `typeguard==2.10.0`
- `tabulate`
- `pyfiglet`
- `versioneer`
- `matplotlib>=3.3.2`
- `h5py>=3.7.0`
- `hdf5==1.12.2`
- `pip`
- `types-tabulate`
- `tables>=3.7.0`
- `pyarrow`

### Developer Specific

The dependencies listed here are only required if you will be doing development for Arkouda.

- `pexpect`
- `pytest>=6.0`
- `pytest-env`
- `Sphinx>=5.1.1`
- `sphinx-argparse`
- `sphinx-autoapi`
- `furo`
- `myst-parser`
- `linkify-it-py`
- `typed-ast`
- `mypy>=0.931,<0.990`
- `flake8`

### Installing/Updating Python Dependencies

Dependencies can be installed using `Anaconda` (Recommended) or `pip`. 

#### Using Anaconda

Arkouda provides 2 files for installing dependencies, one for users and one for developers. 

- Users Environment YAML: `arkouda-env.yml`
- Developer Environment YAML: `arkouda-env-dev.yml`

**When running the commands below, replace `<env_name>` with the name you want to give/have given your conda environment. Replace `<yaml_file>` with the file appropriate to your interaction with Arkouda.**

```commandline
# Creating a new environment with dependencies installed
conda env create -n <env_name> -f <yaml_file>

# Updating env using the yaml 
conda env update -n <env_name> -f <yaml_file> --prune 
# Only use the --prune option if you want to remove packages that are no longer requirements for arkouda.
```

#### Using Pip

When you `pip install Arkouda`, dependencies should be installed as well. However, dependencies may change during the life-cycle of Arkouda, so here we detail how to update dependencies when using `pip` for package management.

```commandline
# navigate to arkouda directory
cd <path_to_arkouda>/arkouda

# Update Dependencies
pip install --upgrade --upgrade-strategy eager -e .

# Updating Developer Dependencies
pip install --upgrade --upgrade-strategy eager -e .[dev]
```
