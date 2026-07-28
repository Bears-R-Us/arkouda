# External Chapel `Parquet` Mason package integration.
#
# get_parquet_package.sh clones/builds the package and prints the chpl arguments
# (C++ prerequisite headers/objects, include paths, Arrow link flags) plus the
# module source path. We use this instead of `mason modules`, which does not
# work reliably with Chapel 2.4. Override PARQUET_INSTALL_DIR to relocate the
# clone; set ARKOUDA_PARQUET_SRC_DIR to reuse an existing checkout.
PARQUET_PACKAGE_SCRIPT := $(ARKOUDA_PROJECT_DIR)/scripts/get_parquet_package.sh
PARQUET_INSTALL_DIR ?= $(DEP_BUILD_DIR)/Parquet

# Keep this as a shell command rather than Make's `$(shell ...)`: Make discards
# the command's exit status, which would let a failed package build continue
# into a misleading Chapel compile failure.
PARQUET_PACKAGE_FLAGS_CMD = env CHPL_HOME="$(ARKOUDA_CHPL_HOME)" \
	"$(PARQUET_PACKAGE_SCRIPT)" "$(PARQUET_INSTALL_DIR)"
