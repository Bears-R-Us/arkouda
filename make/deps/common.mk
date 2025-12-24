DEP_DIR := dep
DEP_INSTALL_DIR := $(ARKOUDA_PROJECT_DIR)/$(DEP_DIR)
DEP_BUILD_DIR := $(ARKOUDA_PROJECT_DIR)/$(DEP_DIR)/build

.PHONY: \
	install-deps \
	deps-download-source

install-deps: install-zmq install-hdf5 install-arrow install-iconv install-idn2

deps-download-source: \
	zmq-download-source \
	hdf5-download-source \
	arrow-download-source \
	iconv-download-source \
	idn2-download-source


