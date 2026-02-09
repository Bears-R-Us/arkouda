# Core project layout
ARKOUDA_PROJECT_DIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
ARKOUDA_PROJECT_DIR := $(patsubst %/,%,$(ARKOUDA_PROJECT_DIR))

PROJECT_NAME := arkouda
ARKOUDA_SOURCE_DIR := $(ARKOUDA_PROJECT_DIR)/src
ARKOUDA_REGISTRY_DIR := $(ARKOUDA_SOURCE_DIR)/registry
ARKOUDA_MAIN_MODULE := arkouda_server
ARKOUDA_MAKEFILES := Makefile Makefile.paths $(wildcard make/*.mk)

DEFAULT_TARGET := $(ARKOUDA_MAIN_MODULE)
.PHONY: default
default: $(DEFAULT_TARGET)

VERBOSE ?= 0
