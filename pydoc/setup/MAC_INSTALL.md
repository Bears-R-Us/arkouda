# MacOS

Prerequisites for Arkouda can be installed using `Homebrew` or manually. Both installation methods have variances to account for the chipset being run.

## Install Chapel

It is recommended to compile Chapel from source. Alternatively, it can be installed via `Homebrew`.

### Build from Source (Recommended)

For convenience, the steps to install Chapel from source are detailed here. If you need more information, please visit the [Chapel Quickstart Guide](https://chapel-lang.org/docs/usingchapel/QUICKSTART.html).

1) Download the current version of Chapel from [here](https://chapel-lang.org/download.html).

2) Unpack the release

```bash
tar xzf chapel-1.29.0.tar.gz
```

3) Access the directory created when the release was unpacked

```bash
cd chapel-1.29.0
```

4) Configure environment variables. *Please Note: This command assumes the use of `bash` or `zsh`. Please refer to the [Chapel Documentation](https://chapel-lang.org/docs/usingchapel/QUICKSTART.html#quickstart-with-other-shells) if you are using another shell.*

```bash
source util/quickstart/setchplenv.bash
```

5) Update some environment variables to the recommended settings. 
- *If you have not installed LLVM, set `CHPL_LLVM=bundled`. It is recommended to install LLVM using Homebrew, `brew install llvm`.* 
- *If you have not installed GMP, set `CHPL_GMP=system`. It is recommended to install GMP using Homebrew, `brew install gmp`.*

6) Add the following to your `rc` file.

```bash
export CHPL_LLVM=system
export CHPL_RE2=bundled
export CHPL_GMP=system # set to the same value as CHPL_LLVM
```

6) Use GNU make to build Chapel

```bash
make
```

7) Ensure that Chapel was built successfully

```bash
chpl examples/hello3-datapar.chpl
./hello3-datapar
```

### Homebrew

Chapel and all supporting dependencies will be installed.

```bash
brew install chapel
```

## Python Environment - Anaconda

Arkouda provides 2 `.yml` files for configuration, one for users and one for developers. The `.yml` files are configured with a default name for the environment, which is used for example interactions with conda. *Please note that you are able to provide a different name by using the `-n` or `--name` parameters when calling `conda env create`

```bash
#Works with all Chipsets (including Apple Silicon)
brew install miniforge
#Add /opt/homebrew/Caskroom/miniforge/base/bin as the first line in /etc/paths

#works with only x86 Architecture (excludes Apple Silicon)
brew install anaconda3
#Add /opt/homebrew/Caskroom/anaconda3/base/bin as the first line in /etc/paths

# User conda env
conda env create -f arkouda-env.yml
conda activate arkouda

# Developer conda env
conda env create -f arkouda-env-dev.yml
conda activate arkouda-dev

#These packages are not required, but nice to have (these are included with Anaconda3)
conda install jupyter

# Install the Arkouda Client Package
pip install -e . --no-deps
```

## Updating Environment

As Arkouda progresses through its life-cycle, dependencies may change. As a result, it is recommended that you keep your development environment in sync with the latest dependencies. The instructions vary depending upon you preferred environment management tool.

### Anaconda

*If you provided a different name when creating the environment, replace `arkouda-dev` or `arkouda` with the name of your Conda environment.*

```bash
# developer environment update
conda env update -n arkouda-dev -f arkouda-env-dev.yml

# user environment update
conda env update -n arkouda -f arkouda-env.yml
```

## Next Steps

Now that you have Arkouda and its dependencies installed on your machine, you will need to be sure to have the appropriate environment variables configured. A complete list can be found at [ENVIRONMENT.md](ENVIRONMENT.md).

Once your environment variables are configured, you are ready to build the server. More information on the build process can be found at [BUILD.md](BUILD.md)
