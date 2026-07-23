# External Chapel `Parquet` Mason package integration.
#
# get_parquet_package.sh clones/builds the package and prints the chpl arguments
# (C++ prerequisite headers/objects, include paths, Arrow link flags) plus the
# module source path. We use this instead of `mason modules`, which does not
# work reliably with Chapel 2.4. Override PARQUET_INSTALL_DIR to relocate the
# clone; set ARKOUDA_PARQUET_SRC_DIR to reuse an existing checkout.
PARQUET_PACKAGE_SCRIPT := $(ARKOUDA_PROJECT_DIR)/scripts/get_parquet_package.sh
PARQUET_INSTALL_DIR ?= $(DEP_BUILD_DIR)/Parquet

# Lazily evaluated (recursive `=`): the script only runs when this variable is
# expanded inside a recipe (the server build and the Arrow dependency check),
# not at parse time, so targets like `clean` don't trigger a clone/build.
PARQUET_PKG_FLAGS = $(shell $(PARQUET_PACKAGE_SCRIPT) $(PARQUET_INSTALL_DIR))
