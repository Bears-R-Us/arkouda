ARROW_PIC ?=

.PHONY: \
	compile-arrow-cpp \
	compile-arrow-write \
	compile-arrow-read \
	compile-arrow-util

SANITIZER = $(shell $(ARKOUDA_CHPL_HOME)/util/chplenv/chpl_sanitizers.py --exe 2>/dev/null)
ifneq ($(SANITIZER),none)
ARROW_SANITIZE = -fsanitize=$(SANITIZER)
endif

compile-arrow-cpp:
	$(MAKE) compile-arrow-write
	$(MAKE) compile-arrow-read
	$(MAKE) compile-arrow-util

compile-arrow-write:
	$(CHPL_CXX) -O3 -std=c++17 $(ARROW_PIC) -c $(ARROW_WRITE_CPP) -o $(ARROW_WRITE_O) \
		$(INCLUDE_FLAGS) $(ARROW_SANITIZE)

compile-arrow-read:
	$(CHPL_CXX) -O3 -std=c++17 $(ARROW_PIC) -c $(ARROW_READ_CPP) -o $(ARROW_READ_O) \
		$(INCLUDE_FLAGS) $(ARROW_SANITIZE)

compile-arrow-util:
	$(CHPL_CXX) -O3 -std=c++17 $(ARROW_PIC) -c $(ARROW_UTIL_CPP) -o $(ARROW_UTIL_O) \
		$(INCLUDE_FLAGS) $(ARROW_SANITIZE)

$(ARROW_UTIL_O): $(ARROW_UTIL_CPP) $(ARROW_UTIL_H)
	$(MAKE) compile-arrow-util

$(ARROW_READ_O): $(ARROW_READ_CPP) $(ARROW_READ_H)
	$(MAKE) compile-arrow-read

$(ARROW_WRITE_O): $(ARROW_WRITE_CPP) $(ARROW_WRITE_H)
	$(MAKE) compile-arrow-write

