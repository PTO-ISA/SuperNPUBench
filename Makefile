LIBNAME = tileop-api
VERSION = 1.0
HEADERS = $(wildcard include/*.h) $(wildcard include/*.hpp) include/jcore include/cpu_sim include/aarch64 include/common

# install to system include directory of Clang
CLANG_PREFIX ?= /remote/lms60/c00622284/janus/linx_blockisa_llvm_musl
INSTALL_DIR = $(shell $(CLANG_PREFIX)/bin/clang -print-resource-dir)/include/$(LIBNAME)

.PHONY: install uninstall copy

install:
	@echo "Installing $(LIBNAME) to Clang toolchain at $(INSTALL_DIR)"
	@mkdir -p $(INSTALL_DIR)
	@cp -r $(HEADERS) $(INSTALL_DIR)
	@echo "Installation complete. Now you can use #include <$(LIBNAME)/header.h>"

uninstall:
	@echo "Removing $(INSTALL_DIR)"
	@rm -rf $(INSTALL_DIR)
	@echo "Uninstall complete"

#copy tileop from pto lib
copy: PTO_INCLUDE_PATH = $(shell realpath ../PTOTileLib/include)
copy:
	@echo "Copying $(PTO_INCLUDE_PATH) to current include directory"
	@cp -r $(PTO_INCLUDE_PATH)/* ./include/
	@echo "Copy Complete"