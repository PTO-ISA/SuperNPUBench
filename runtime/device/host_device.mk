# Consumer sets HOST_MAIN and DEVICE_KERNEL. Run make in its benchmark directory.
ifneq ($(MAKECMDGOALS),test-host)
ifndef COMPILER_DIR
$(error Set COMPILER_DIR to the Linx toolchain bin directory)
endif
ifndef TILEOP_API_ROOT
$(error Set TILEOP_API_ROOT to the Linx-TileOP-API checkout)
endif
ifndef MODEL_ROOT
$(error Set MODEL_ROOT to the SuperScalarModel checkout)
endif
endif

RUNTIME := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
REPO := $(abspath $(RUNTIME)/..)
OUTPUT ?= $(CURDIR)/output
OUTPUT := $(abspath $(OUTPUT))
HOST_CXX ?= c++
LINX_CXX := $(COMPILER_DIR)/clang++
LINX_LD := $(COMPILER_DIR)/ld.lld
DUMP := $(COMPILER_DIR)/llvm-objdump
WORKER_RUNTIME := $(MODEL_ROOT)/runtime/linx-runtime
TILEOP_INCLUDE := $(TILEOP_API_ROOT)/include
DEVICE_FLAGS := --target=linx64v5-unknown-linux-musl -std=c++20 -O2 -mlxbc \
    -fenable-matrix -mllvm -enable-all-vector-as-tilereg=true -DENABLE_TENSOR_INSTR \
    -ffreestanding -fno-exceptions -fno-rtti -fno-jump-tables -fno-stack-protector \
    -I$(TILEOP_INCLUDE) -I$(REPO)/benchmark/one-level-arch/kernels -I$(RUNTIME)
# Standalone LLVM builds can use an existing target sysroot for libc++ headers.
ifneq ($(strip $(LINX_SYSROOT)),)
DEVICE_FLAGS += --sysroot=$(LINX_SYSROOT) -isystem $(LINX_SYSROOT)/usr/include/c++/v1
endif

DEVICE_KERNEL_SOURCES ?= $(DEVICE_KERNEL)
DEVICE_KERNEL_OBJECTS := $(addprefix $(OUTPUT)/device/,$(DEVICE_KERNEL_SOURCES:.cpp=.o))

.PHONY: all run run-negative firmware test-host
all: $(OUTPUT)/host firmware

$(OUTPUT):
	mkdir -p "$@"

$(OUTPUT)/host: $(HOST_MAIN) $(HOST_HEADERS) $(RUNTIME)/host/device.hpp $(RUNTIME)/host/device.cpp | $(OUTPUT)
	$(HOST_CXX) -pthread -std=c++17 -O2 -Wall -Wextra -I$(RUNTIME) -I$(RUNTIME)/host \
	    $(HOST_MAIN) $(RUNTIME)/host/device.cpp -o "$@"

ifneq ($(strip $(HOST_NEGATIVE)),)
$(OUTPUT)/invalid_host: $(HOST_NEGATIVE) $(HOST_HEADERS) $(RUNTIME)/host/device.hpp $(RUNTIME)/common/kernel_abi.hpp $(RUNTIME)/host/device.cpp | $(OUTPUT)
	$(HOST_CXX) -pthread -std=c++17 -O2 -Wall -Wextra -I$(RUNTIME) -I$(RUNTIME)/host \
	    $(HOST_NEGATIVE) $(RUNTIME)/host/device.cpp -o "$@"
endif

test-host: $(OUTPUT)/device_test $(OUTPUT)/kernel_entry_test
	"$(OUTPUT)/device_test" "$(RUNTIME)/host/fake_gfrun.py"
	"$(OUTPUT)/kernel_entry_test"

$(OUTPUT)/device_test: $(RUNTIME)/host/device_test.cpp $(RUNTIME)/host/device.cpp $(RUNTIME)/host/device.hpp $(RUNTIME)/common/kernel_abi.hpp | $(OUTPUT)
	$(HOST_CXX) -pthread -std=c++17 -O2 -Wall -Wextra -I$(RUNTIME) $(filter %.cpp,$^) -o "$@"

$(OUTPUT)/kernel_entry_test: $(RUNTIME)/device/kernel_entry_test.cpp $(RUNTIME)/device/kernel_entry.hpp $(RUNTIME)/common/kernel_abi.hpp | $(OUTPUT)
	$(HOST_CXX) -std=c++17 -O2 -Wall -Wextra -I$(RUNTIME) "$<" -o "$@"

$(OUTPUT)/device/%.o: %.cpp $(RUNTIME)/device/kernel_entry.hpp $(RUNTIME)/common/kernel_abi.hpp | $(OUTPUT)
	@mkdir -p "$(@D)"
	$(LINX_CXX) $(DEVICE_FLAGS) -MMD -MP -c "$<" -o "$@"

firmware: $(DEVICE_KERNEL_OBJECTS)
	$(MAKE) -C "$(WORKER_RUNTIME)" msgb-image OUTPUT="$(OUTPUT)" \
	    LINX_CXX="$(LINX_CXX)" LINX_LD="$(LINX_LD)" TILEOP_INCLUDE="$(TILEOP_INCLUDE)" \
	    KERNEL_OBJECTS="$(DEVICE_KERNEL_OBJECTS)"
	$(DUMP) -t "$(OUTPUT)/workers.elf" > "$(OUTPUT)/workers.syms"
	awk '$$2 == "g" && $$3 == "F" {print $$1, $$NF}' "$(OUTPUT)/workers.syms" > "$(OUTPUT)/workers.elf.kernels"

run: all
	"$(OUTPUT)/host" "$(GFRUN)" "$(OUTPUT)/workers.elf"

run-negative: firmware $(if $(strip $(HOST_NEGATIVE)),$(OUTPUT)/invalid_host)
	@test -n "$(strip $(HOST_NEGATIVE))" || { echo "HOST_NEGATIVE is not set" >&2; exit 2; }
	"$(OUTPUT)/invalid_host" "$(GFRUN)" "$(OUTPUT)/workers.elf"

GFRUN ?= $(MODEL_ROOT)/bin/gfrun
-include $(DEVICE_KERNEL_OBJECTS:.o=.d)
