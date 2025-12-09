#################
#### Test.mk ####
#################

TEST_SOURCE_DIR := tests/server
TEST_SOURCES := $(wildcard $(TEST_SOURCE_DIR)/*.chpl)
TEST_MODULES := $(basename $(notdir $(TEST_SOURCES)))

TEST_BINARY_DIR := test-bin
TEST_BINARY_SIGIL := #t-
TEST_TARGETS := $(addprefix $(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL),$(TEST_MODULES))

ifeq ($(VERBOSE),1)
TEST_CHPL_FLAGS ?= $(CHPL_DEBUG_FLAGS) $(CHPL_FLAGS)
else
TEST_CHPL_FLAGS ?= $(CHPL_FLAGS)
endif

define TEST_HELP_TEXT
# test			Build all tests ($(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL)*); Can override TEST_CHPL_FLAGS
  test-help
  test-clean
 $(foreach t,$(sort $(TEST_TARGETS)), $(t)\n)
endef
$(eval $(call create_help_target,test-help,TEST_HELP_TEXT))

.PHONY: test
test: test-python

.PHONY: test-chapel
test-chapel:
	start_test $(TEST_SOURCE_DIR)

.PHONY: test-all
test-all: test-python test-chapel

mypy:
	python3 -m mypy arkouda

$(TEST_BINARY_DIR):
	mkdir -p $(TEST_BINARY_DIR)

.PHONY: $(TEST_TARGETS) # Force tests to always rebuild.
$(TEST_TARGETS): $(TEST_BINARY_DIR)/$(TEST_BINARY_SIGIL)%: $(TEST_SOURCE_DIR)/%.chpl | $(TEST_BINARY_DIR)
	$(CHPL) $(TEST_CHPL_FLAGS) -M $(ARKOUDA_SOURCE_DIR) $(ARKOUDA_COMPAT_MODULES) $< -o $@

print-%:
	$(info $($*)) @trues

size=100
skip_doctest="False"
test-python:
	python3 -m pytest -c pytest.ini --size=$(size) $(ARKOUDA_PYTEST_OPTIONS) --skip_doctest=$(skip_doctest) --html=.pytest/report.html --self-contained-html
	python3 -m pytest -c pytest.opts.ini --size=$(size) $(ARKOUDA_PYTEST_OPTIONS)

CLEAN_TARGETS += test-clean
.PHONY: test-clean
test-clean:
	$(RM) $(TEST_TARGETS) $(addsuffix _real,$(TEST_TARGETS))

size_bm = 10**8
DATE := $(shell date '+%Y_%m_%d_%H_%M_%S')
out=benchmark_v2/data/benchmark_stats_$(DATE).json
.PHONY: benchmark
benchmark:
	mkdir -p benchmark_v2/data
	python3 -m pytest -c benchmark.ini --benchmark-autosave --benchmark-storage=file://benchmark_v2/.benchmarks --size=$(size_bm) --benchmark-json=$(out)
	python3 benchmark_v2/reformat_benchmark_results.py --benchmark-data $(out)

version:
	@echo $(VERSION);


