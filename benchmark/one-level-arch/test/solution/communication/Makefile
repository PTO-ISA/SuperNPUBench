TESTCASE ?= comm_payload
HOST_CXX ?= c++
HOST_FLAGS ?= -O2
HOST_OUT := ../../../output/solution/communication/host
KERNEL_HEADERS := $(wildcard ../../../kernels/communication/*.hpp)

ifneq ($(filter host-check host-sanitize,$(MAKECMDGOALS)),)
.PHONY: host-check host-sanitize
host-check: $(HOST_OUT)/host_mmap_test
	$(HOST_OUT)/host_mmap_test
	python3 -m unittest discover -s src -p 'test_*.py'

$(HOST_OUT)/host_mmap_test: src/host_mmap_test.cpp $(KERNEL_HEADERS)
	@mkdir -p $(HOST_OUT)
	$(HOST_CXX) -std=c++20 -Wall -Wextra -Werror -pthread $(HOST_FLAGS) -I../../../kernels $< -o $@

host-sanitize:
	@mkdir -p $(HOST_OUT)
	$(HOST_CXX) -std=c++20 -Wall -Wextra -Werror -pthread -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I../../../kernels src/host_mmap_test.cpp -o $(HOST_OUT)/host_mmap_test_sanitize
	$(HOST_OUT)/host_mmap_test_sanitize
else
ifneq ($(TESTCASE),comm_payload)
$(error Unsupported TESTCASE=$(TESTCASE); expected comm_payload)
endif
res_check ?= on
ifneq ($(res_check),on)
$(error comm_payload requires res_check=on for independent binary result checking)
endif
SRC_FILE += $(TEST_ROOT)/$(CASE_SRC_DIR)/comm_payload.cpp
TARGET = $(ELF_HEAD)_comm_payload.elf
include ../../common/Makefile.common
endif
