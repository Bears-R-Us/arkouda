# Flags for Python interop: this is required for ApplyMsg to call into the Python interpreter
#
# Adds the include path for the Python.h header file
CHPL_FLAGS += --ccflags -isystem$(shell python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
# Adds the library path for the Python shared library
CHPL_FLAGS += -L$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
CHPL_FLAGS += --ldflags -Wl,-rpath,$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
# Adds the Python shared library to the list of libraries to link against
CHPL_FLAGS += -lpython$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
# Ignore warnings from the Python headers. This is irrelevant for newer Python versions
CHPL_FLAGS += --ccflags -Wno-macro-redefined

# Allows the user to set a different instantiation limit
ifneq ($(instantiate_max),)
CHPL_USER_FLAGS += --instantiate-max=$(instantiate_max)
endif


PYTHON_VERSION := $(shell python3 -c "import sys; print(*sys.version_info[:2], sep='.')")
