# Split top-level Makefile for Arkouda

# Include build configuration, dependencies, flags, macros, etc.
include make/Prologue.mk

# Core Arkouda build rules
include make/Arkouda.mk

# Archive / release helpers
include make/Archive.mk

# Documentation build rules
include make/Doc.mk

# Linters and style formatters
include make/Dev.mk

# Testing and related helpers
include make/Test.mk

# Command registration helpers
include make/register.mk

# Clean/help epilogue
include make/Epilogue.mk
