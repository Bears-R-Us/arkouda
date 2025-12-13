#####################
#### Epilogue.mk ####
#####################

define CLEAN_HELP_TEXT
# clean
  clean-help
 $(foreach t,$(CLEAN_TARGETS), $(t)\n)
endef
$(eval $(call create_help_target,clean-help,CLEAN_HELP_TEXT))

.PHONY: clean cleanall
clean: $(CLEAN_TARGETS)
cleanall: clean $(CLEANALL_TARGETS)

.PHONY: help
help: $(HELP_TARGETS)


