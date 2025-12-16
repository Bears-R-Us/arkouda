ZMQ_VER := 4.3.5
ZMQ_NAME_VER := zeromq-$(ZMQ_VER)
ZMQ_BUILD_DIR := $(DEP_BUILD_DIR)/$(ZMQ_NAME_VER)
ZMQ_INSTALL_DIR := $(DEP_INSTALL_DIR)/zeromq-install
ZMQ_LINK := https://github.com/zeromq/libzmq/releases/download/v$(ZMQ_VER)/$(ZMQ_NAME_VER).tar.gz

zmq-download-source:
	mkdir -p $(DEP_BUILD_DIR)
    #If the build directory does not exist,  create it
    ifeq (,$(wildcard ${ZMQ_BUILD_DIR}*/.*))
        #   If the tar.gz not found, download it
        ifeq (,$(wildcard ${DEP_BUILD_DIR}/${ZMQ_NAME_VER}*.tar.gz))
			cd $(DEP_BUILD_DIR) && curl -sL $(ZMQ_LINK) | tar xz
        #   Otherwise just unzip it
        else
			cd $(DEP_BUILD_DIR) && tar -xzf $(ZMQ_NAME_VER)*.tar.gz
        endif
    endif

install-zmq: zmq-download-source
	@echo "Installing ZeroMQ"
	rm -rf $(ZMQ_INSTALL_DIR)
	mkdir -p $(DEP_INSTALL_DIR) $(DEP_BUILD_DIR)

	cd $(ZMQ_BUILD_DIR) && ./configure --prefix=$(ZMQ_INSTALL_DIR) CFLAGS=-O3 CXXFLAGS=-O3 && make && make install
	echo '$$(eval $$(call add-path,$(ZMQ_INSTALL_DIR)))' >> Makefile.paths

zmq-clean:
	rm -r $(ZMQ_BUILD_DIR)

