install-arrow-quick:
	./scripts/install_arrow_quick.sh $(DEP_BUILD_DIR)


ARROW_VER := 19.0.1
ARROW_NAME_VER := apache-arrow-$(ARROW_VER)
ARROW_FULL_NAME_VER := arrow-apache-arrow-$(ARROW_VER)
ARROW_BUILD_DIR := $(DEP_BUILD_DIR)/$(ARROW_FULL_NAME_VER)
ARROW_DEP_DIR :=  $(DEP_BUILD_DIR)/arrow_dependencies
ARROW_INSTALL_DIR := $(DEP_INSTALL_DIR)/arrow-install
ARROW_SOURCE_LINK := https://github.com/apache/arrow/archive/refs/tags/$(ARROW_NAME_VER).tar.gz

NUM_CORES := $(shell nproc --all)

ARROW_DEPENDENCY_SOURCE := BUNDLED

arrow-download-source:
	mkdir -p $(DEP_BUILD_DIR)

    #   If the tar.gz file does not exist, fetch it
    ifeq (,$(wildcard ${DEP_BUILD_DIR}/$(ARROW_NAME_VER).tar.gz))
		cd $(DEP_BUILD_DIR) && wget $(ARROW_SOURCE_LINK)
    endif

	cd $(DEP_BUILD_DIR) && tar -xvf $(ARROW_NAME_VER).tar.gz

    # if the arrow dependency directory is empty of tar.gz, download the dependencies
    ifeq (,$(wildcard $(ARROW_DEP_DIR)/*.tar.gz))
		rm -fr $(DEP_BUILD_DIR)/arrow_exports.sh
		mkdir -p $(ARROW_DEP_DIR)
		cd $(ARROW_BUILD_DIR)/cpp/thirdparty/ && ./download_dependencies.sh $(ARROW_DEP_DIR) > $(DEP_BUILD_DIR)/arrow_exports.sh
    endif

	rm -fr $(ARROW_BUILD_DIR)

install-arrow: arrow-download-source
	@echo "Installing Apache Arrow/Parquet"
	@echo "from build directory: ${DEP_BUILD_DIR}"
	rm -rf $(ARROW_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)

	cd $(DEP_BUILD_DIR) && tar -xvf $(ARROW_NAME_VER).tar.gz
	mkdir -p $(ARROW_BUILD_DIR)/cpp/build-release

	cd $(DEP_BUILD_DIR) && . ./arrow_exports.sh && cd $(ARROW_BUILD_DIR)/cpp/build-release && cmake -S $(ARROW_BUILD_DIR)/cpp .. -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_INSTALL_PREFIX=$(ARROW_INSTALL_DIR) -DCMAKE_BUILD_TYPE=Release -DARROW_PARQUET=ON -DARROW_WITH_SNAPPY=ON -DARROW_WITH_BROTLI=ON -DARROW_WITH_BZ2=ON -DARROW_WITH_LZ4=ON -DARROW_WITH_ZLIB=ON -DARROW_WITH_ZSTD=ON -DARROW_DEPENDENCY_SOURCE=$(ARROW_DEPENDENCY_SOURCE) $(ARROW_OPTIONS) && make -j$(NUM_CORES)

	cd $(ARROW_BUILD_DIR)/cpp/build-release && make install

	echo '$$(eval $$(call add-path,$(ARROW_INSTALL_DIR)))' >> Makefile.paths

arrow-clean:
	rm -rf $(DEP_BUILD_DIR)/apache-arrow*
	rm -rf $(DEP_BUILD_DIR)/arrow-apache-arrow*
	rm -rf $(ARROW_DEP_DIR)
	rm -fr $(DEP_BUILD_DIR)/arrow_exports.sh

