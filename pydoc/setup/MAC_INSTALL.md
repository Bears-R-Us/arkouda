# MacOS

Prerequisites for Arkouda can be installed using `Homebrew` or manually. Both installation methods have variances to account for the chipset being run.

## Python Environment - Anaconda

Arkouda provides 2 `.yml` files for configuration, one for users and one for developers. The `.yml` files are configured with a default name for the environment, which is used in the example interactions with conda below. 
To provide a different name for the environment, use the `-n` or `--name` parameters when calling `conda env create`.

```bash
# We recommend running the full Anaconda 
brew install anaconda3

# Note - the exact path may vary based on the release of Anaconda that is current and your mac's chipset
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

# These packages are not required, but nice to have (these are included with Anaconda3)
conda install jupyter

# Install the Arkouda Client Package. For this to work properly you need to change directories to where arkouda lives
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

## Install Chapel

It is recommended to compile Chapel from source. Alternatively, it can be installed via `Homebrew`.

### Build from Source (Recommended)

For convenience, the steps to install Chapel from source are detailed here. If you need more information, please visit the [Chapel Quickstart Guide](https://chapel-lang.org/docs/usingchapel/QUICKSTART.html).

**Step 1**
> Download the current version of Chapel from [here](https://chapel-lang.org/download.html).

**Step 2**
> Unpack the release
> ```bash
> tar xzf chapel-1.32.0.tar.gz
> ```

**Step 3**
> Access the directory created when the release was unpacked
> ```bash
> cd chapel-1.32.0
> ```

**Step 4**
>Configure environment variables. *Please Note: This command assumes the use of `bash` or `zsh`. Please refer to the [Chapel Documentation](https://chapel-lang.org/docs/usingchapel/QUICKSTART.html#quickstart-with-other-shells) if you are using another shell.*
> ```bash
> source util/quickstart/setchplenv.bash
> ```

**Step 5**
> Update environment variables to the recommended settings. 
> ```bash
> brew install llvm@15
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

**Step 6**
> Add the following to your `rc` file.
> ```bash
> export CHPL_LLVM=system # set to the same value as in Step 5
> export CHPL_RE2=bundled
> export CHPL_GMP=system # set to the same value as CHPL_LLVM
> ```

**Step 7**
> Use GNU make to build Chapel
> ```bash
> make -j 16
> ```

**Step 8**
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
We've set up chapel to run locally, to simulate running on a distributed machine follow
the instructions at [GASNet Development](https://bears-r-us.github.io/arkouda/developer/GASNET.html).

Now that you have Arkouda and its dependencies installed on your machine, you will need to have the appropriate environment variables configured. A complete list can be found at [ENVIRONMENT.md](ENVIRONMENT.md).

Once your environment variables are configured, you are ready to build the server. More information on the build process can be found at [BUILD.md](BUILD.md).
