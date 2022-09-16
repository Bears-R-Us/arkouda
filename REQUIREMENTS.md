# Dependency List

The dependencies listed here have varying installation instructions depending upon your preferred operating system. Please use in the installation instructions provided in [INSTALL.md](INSTALL.md).

- `Chapel 1.27.0 or later`
- `cmake>=3.11.0`
- `zeromq>=4.2.5`
- `hdf5`
- `python>=3.8`

# Python Dependencies

The following python packages are required by the Arkouda client package.

- `python>=3.8`
- `numpy>=1.22.2`
- `pandas>=1.4.0`
- `pyzmq>=20.0.0`
- `typeguard==2.10.0`
- `tabulate`
- `pyfiglet`
- `versioneer`
- `matplotlib>=3.3.2`
- `h5py`
- `pip`
- `types-tabulate`
- `tables>=3.7.0`
- `pyarrow>=1.0.1`

## Developer Specific

The dependencies listed here are only required if you will be doing development for Arkouda.

- `pexpect`
- `pytest>=6.0`
- `pytest-env`
- `Sphinx`
- `sphinx-argparse`
- `sphinx-autoapi`
- `typed-ast`
- `mypy>=0.931`
- `flake8`

## Installing/Updating Python Dependencies

Dependencies can be installed using `Anaconda` (Recommended) or `pip`. 

### Using Anaconda
Arkouda provides 2 files for installing dependencies, one for users and one for developers. 

- Users Environment YAML: `arkouda-env.yml`
- Developer Environment YAML: `arkouda-env-dev.yml`

**When running the commands below, replace `<env_name>` with the name you want to give/have given your conda environment. Replace `<yaml_file>` with the file appropriate to your interaction with Arkouda.**
```commandline
# Creating a new environment with dependencies installed
conda env create -n <env_name> -f <yaml_file>

# Updating env using the yaml 
conda env update -n <env_name> -f <yaml_file> --prune 
#Only use the --prune option if you want to remove packages that are no longer requirements for arkouda.
```

### Using Pip
When you `pip install Arkouda`, dependencies should be installed as well. However, dependencies may change during the life-cycle of Arkouda, so here we detail how to update dependencies when using `pip` for package management.

```commandline
# navigate to arkouda directory
cd <path_to_arkouda>/arkouda

# Update Dependencies
pip install --upgrade --upgrade-strategy eager -e .

# Updating Developer Dependencies
pip install --upgrade --upgrade-strategy eager -e .[dev]
```
 
