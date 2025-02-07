# MacOS

Prerequisites for Arkouda can be installed using `Homebrew` or manually.

## Clone Arkouda Repository

Download, clone, or fork the [arkouda repo](https://github.com/Bears-R-Us/arkouda).

We encourage developers to fork the repo if they expect to make any changes to arkouda.
They can then clone their fork and add the Bears-R-Us repo as a remote:
```bash
git clone https://github.com/YOUR_FORK/arkouda.git
cd arkouda
git remote add upstream https://github.com/Bears-R-Us/arkouda.git
```

For users who aren't intending to make any changes, cloning the arkouda repo should be enough
```bash
git clone https://github.com/Bears-R-Us/arkouda.git
```

Further instructions assume that the current directory is the top-level directory of the arkouda repo.

## Python Environment - Anaconda

Arkouda provides 2 `.yml` files for configuration, one for users and one for developers.
The `.yml` files are configured with a default name for the environment, which is used in the example interactions with conda below. 
To provide a different name for the environment, use the `-n` or `--name` parameters when calling `conda env create`.

```bash
# We recommend running the full Anaconda 
brew install --cask anaconda

# Note - the exact path may vary based on the most current release of Anaconda and your mac's chipset
# Run the script to install Anaconda.
/opt/homebrew/Caskroom/anaconda/2022.10/Anaconda3-2022.10-MacOSX-arm64.sh

# initialize conda
conda init

# User conda env
conda env create -f arkouda-env.yml
conda activate arkouda

# Developer conda env
conda env create -f arkouda-env-dev.yml
conda activate arkouda-dev

# Install the Arkouda Client Package and add it to your PYTHONPATH.
# For this to work properly you need to change directories to where arkouda lives
pip install -e . --no-deps
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

## Updating Environment

As Arkouda progresses through its life-cycle, dependencies may change.
As a result, it is recommended that you keep your development environment in sync with the latest dependencies.
The instructions vary depending upon you preferred environment management tool.

### Anaconda

*If you provided a different name when creating the environment, replace `arkouda-dev` or `arkouda` with the name of your Conda environment.*

```bash
# developer environment update
conda env update -n arkouda-dev -f arkouda-env-dev.yml

# user environment update
conda env update -n arkouda -f arkouda-env.yml
```

## Install Chapel

It is recommended to compile Chapel from source. Alternatively, it can be installed via `Homebrew`.

### Build from Source (Recommended)

For convenience, the steps to install Chapel from source are detailed here.
If you need more information, please visit the [Chapel Quickstart Guide](https://chapel-lang.org/docs/usingchapel/QUICKSTART.html).

**Step 1** 
> Navigate to the directory where you want Chapel to be installed.
> You should not install Chapel in the arkouda directory

**Step 2**
> Download the current version of Chapel from [here](https://chapel-lang.org/download.html).

**Step 3**
> Unpack the release
> ```bash
> tar xzf chapel-2.1.0.tar.gz
> ```

**Step 4**
> Access the directory created when the release was unpacked
> ```bash
> cd chapel-2.1.0
> ```

**Step 5**
>Configure environment variables. *Please Note: This command assumes the use of `bash` or `zsh`. Please refer to the [Chapel Documentation](https://chapel-lang.org/docs/usingchapel/QUICKSTART.html#quickstart-with-other-shells) if you are using another shell.*
> ```bash
> source util/quickstart/setchplenv.bash
> ```

**Step 6**
> Update environment variables to the recommended settings. 
> ```bash
> brew install llvm
> export CHPL_LLVM=system
>
> brew install gmp
> export CHPL_GMP=system
> 
> export CHPL_RE2=bundled
>
> unset CHPL_DEVELOPER
> ```

> If you choose to use the packages bundled with Chapel, use the following settings.
> ```bash
> export CHPL_GMP=bundled
> export CHPL_LLVM=bundled
> export CHPL_RE2=bundled
> ```

**Step 7**
> Add the following to your `rc` file.
> ```bash
> # update paths to reflect where chapel and arkouda live on your machine
> export CHPL_HOME=/Users/USER/PATH_TO_CHPL/chapel-2.1.0
> # your binary might differ especially if you have a different chipset
> export PATH=$PATH:$CHPL_HOME/bin/darwin-arm64
> source ${CHPL_HOME}/util/setchplenv.bash
> export CHPL_LLVM=system # set to the same value as in the previous step
> export CHPL_GMP=system # set to the same value as in the previous step
> export CHPL_RE2=bundled
> export CHPL_COMM=none
> export CHPL_TARGET_CPU=native
> export ARKOUDA_QUICK_COMPILE=true
> export PYTHONPATH="${PYTHONPATH}:/Users/USER/PATH_TO_ARK/arkouda"
> ```

**Step 8**
> Source your `rc` file to set any environment variables, you might need to reactivate your conda environment
> ```bash
> source ~/.zshrc # or ~/.bashrc depending on your shell
> conda activate arkouda-dev # or the name of your conda environment
> ```

**Step 9**
> Use `make` to build Chapel
> ```bash
> make -j 8  # you can bump this up 16 if you have enough memory
> ```

**Step 10**
> Ensure that Chapel was built successfully
> ```bash
> chpl examples/hello3-datapar.chpl
> ./hello3-datapar
> ```

### Homebrew

Alternatively, you can use homebrew to install Chapel and all it's supporting dependencies.

```bash
brew install chapel
```

## Next Steps

We've installed Arkouda and its dependencies and built chapel with reasonable default environment variables.

There are also several external tools we recommend to install for consistancy across developers. The instructions 
are at [EXTERNAL_TOOLS.md](EXTERNAL_TOOLS.md).

Now you are ready to build the server! Follow the build instructions at [BUILD.md](BUILD.md).

We've set up chapel and arkouda to run locally with no communication! If you want to simulate running on a distributed machine follow
the instructions at [GASNet Development](https://bears-r-us.github.io/arkouda/developer/GASNET.html).
