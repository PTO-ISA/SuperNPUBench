# Consumer sets HOST_MAIN and DEVICE_KERNEL. Run make in its benchmark directory.
ifndef COMPILER_DIR
$(error Set COMPILER_DIR to the Linx toolchain bin directory)
endif
ifndef TILEOP_API_ROOT
$(error Set TILEOP_API_ROOT to the Linx-TileOP-API checkout)
endif
ifndef MODEL_ROOT
$(error Set MODEL_ROOT to the SuperScalarModel checkout)
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
    -I$(TILEOP_INCLUDE) -I$(REPO)/benchmark/one-level-arch/kernels

.PHONY: all run firmware
all: $(OUTPUT)/host firmware

$(OUTPUT):
	mkdir -p "$@"

$(OUTPUT)/host: $(HOST_MAIN) $(HOST_HEADERS) $(RUNTIME)/host/device.hpp $(RUNTIME)/host/device.cpp | $(OUTPUT)
	$(HOST_CXX) -std=c++17 -O2 -Wall -Wextra -I$(RUNTIME)/host \
	    $(HOST_MAIN) $(RUNTIME)/host/device.cpp -o "$@"

$(OUTPUT)/kernel.o: $(DEVICE_KERNEL) | $(OUTPUT)
	$(LINX_CXX) $(DEVICE_FLAGS) -MMD -MP -c "$<" -o "$@"

firmware: $(OUTPUT)/kernel.o
	$(MAKE) -C "$(WORKER_RUNTIME)" msgb-image OUTPUT="$(OUTPUT)" \
	    LINX_CXX="$(LINX_CXX)" LINX_LD="$(LINX_LD)" TILEOP_INCLUDE="$(TILEOP_INCLUDE)" \
	    KERNEL_OBJECTS="$(OUTPUT)/kernel.o"
	$(DUMP) -t "$(OUTPUT)/workers.elf" > "$(OUTPUT)/workers.syms"
	awk '$$2 == "g" && $$3 == "F" {print $$1, $$NF}' "$(OUTPUT)/workers.syms" > "$(OUTPUT)/workers.elf.kernels"

run: all
	"$(OUTPUT)/host" "$(GFRUN)" "$(OUTPUT)/workers.elf"

GFRUN ?= $(MODEL_ROOT)/bin/gfrun
-include $(OUTPUT)/kernel.d
