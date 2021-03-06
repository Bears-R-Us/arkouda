# Chapel Patches

From time to time we may need to apply a patch to chapel.  This outlines the
patches, corresponding chapel version, and the reason it is needed along with any
instructions to apply it.

### chpldoc.693.patch
This patch is required for Chapel version 1.23.0 and corresponds to Arkouda
Github issue #693.  It syncs the version of Sphinx etc. needed to get the chpldoc
make target to compile correctly.  To apply (perfom commands in your CHPL_HOME:
```bash
# In the root of your chapel source code run
git apply --stat ${AK_HOME}/chapel-patches/chpldoc.693.patch  # To show what will be changed
git apply --check ${AK_HOME}/chapel-patches/chpldoc.693.patch  # To check
git apply ${AK_HOME}/chapel-patches/chpldoc.693.patch  # To apply patch and modify files
make chpldoc  # Previously that target should have failed, after the patch it should work correctly

# Note, if you haven't installed python packages sphinx-rtd-theme and sphinxcontrib-chapeldomain you should do that now
pip install sphinx-rtd-theme
pip install sphinxcontrib-chapeldomain
```
