LIBNAME = tileop-api
VERSION = 1.0
HEADERS = $(wildcard include/*.h) $(wildcard include/*.hpp) include/jcore include/cpu_sim include/aarch64 include/common


# install to system include directory of Clang
CLANG_PREFIX ?=
INSTALL_DIR = $(shell $(CLANG_PREFIX)/bin/clang -print-resource-dir)/include/$(LIBNAME)

.PHONY: install uninstall

install:
	@echo "Installing $(LIBNAME) to Clang toolchain at $(INSTALL_DIR)"
	@mkdir -p $(INSTALL_DIR)
	@cp -r $(HEADERS) $(INSTALL_DIR)
	@echo "Installation complete. Now you can use #include <$(LIBNAME)/header.h>"

uninstall:
	@echo "Removing $(INSTALL_DIR)"
	@rm -rf $(INSTALL_DIR)
	@echo "Uninstall complete"
