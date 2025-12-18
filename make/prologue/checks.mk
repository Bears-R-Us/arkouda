.PHONY: \
	check-deps \
	check-chpl

ifndef ARKOUDA_SKIP_CHECK_DEPS
CHECK_DEPS = check-chpl check-zmq check-hdf5 check-re2 check-arrow check-iconv check-idn2
endif
check-deps: $(CHECK_DEPS)

CHPL_VERSION_OK := $(shell test $(CHPL_MAJOR) -ge 2 -o $(CHPL_MINOR) -ge 0  && echo yes)
# CHPL_VERSION_WARN := $(shell test $(CHPL_MAJOR) -eq 1 -a $(CHPL_MINOR) -le 33 && echo yes)

check-chpl:
ifneq ($(CHPL_VERSION_OK),yes)
	$(error Chapel 2.0 or newer is required, found $(CHPL_MAJOR).$(CHPL_MINOR))
endif
# ifeq ($(CHPL_VERSION_WARN),yes)
# 	$(warning Chapel 1.33.0 or newer is recommended, found $(CHPL_MAJOR).$(CHPL_MINOR))
# endif

ZMQ_CHECK = $(DEP_INSTALL_DIR)/checkZMQ.chpl
check-zmq: $(ZMQ_CHECK)
	@echo "Checking for ZMQ"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

HDF5_CHECK = $(DEP_INSTALL_DIR)/checkHDF5.chpl
check-hdf5: $(HDF5_CHECK)
	@echo "Checking for HDF5"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

RE2_CHECK = $(DEP_INSTALL_DIR)/checkRE2.chpl
check-re2: $(RE2_CHECK)
	@echo "Checking for RE2"
	@$(CHPL) $(CHPL_FLAGS) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

ARROW_CHECK = $(DEP_INSTALL_DIR)/checkArrow.chpl
check-arrow: $(ARROW_CHECK) $(ARROW_UTIL_O) $(ARROW_READ_O) $(ARROW_WRITE_O)
	@echo "Checking for Arrow"
	$(MAKE) compile-arrow-cpp
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) $< \
		$(ARROW_M) -M $(ARKOUDA_SOURCE_DIR) \
		-I $(ARKOUDA_SOURCE_DIR)/parquet \
		-o $(DEP_INSTALL_DIR)/$@ && \
		([ $$? -eq 0 ] && echo "Success compiling program") || \
		echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real


ICONV_CHECK = $(DEP_INSTALL_DIR)/checkIconv.chpl
check-iconv: $(ICONV_CHECK)
	@echo "Checking for iconv"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) -M $(ARKOUDA_SOURCE_DIR) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

IDN2_CHECK = $(DEP_INSTALL_DIR)/checkIdn2.chpl
check-idn2: $(IDN2_CHECK)
	@echo "Checking for idn2"
	@$(CHPL) $(CHPL_FLAGS) $(ARKOUDA_COMPAT_MODULES) -M $(ARKOUDA_SOURCE_DIR) $< -o $(DEP_INSTALL_DIR)/$@ && ([ $$? -eq 0 ] && echo "Success compiling program") || echo "\nERROR: Please ensure that dependencies have been installed correctly (see -> https://github.com/Bears-R-Us/arkouda/blob/main/pydoc/setup/BUILD.md)\n"
	$(DEP_INSTALL_DIR)/$@ -nl 1
	@rm -f $(DEP_INSTALL_DIR)/$@ $(DEP_INSTALL_DIR)/$@_real

