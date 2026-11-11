// SSM-specific platform adapter for the compiler-rt common-entry runtime.
// This is NOT an ISA-defined note, register, syscall or completion protocol.
#include <stdint.h>
#include "ssm_hosted_startup.h"

struct pto_runtime_group_context;
extern void __linx_group_worker_start(void) __attribute__((noreturn));

// Strong writable slot populated by the SSM launcher before any PE executes.
volatile uintptr_t __ssm_pto_group_context;
__attribute__((used, section(".note.ssm.pto"), aligned(4)))
static const uint32_t ssm_hosted_note[] = {
    8, 16, 0x50544f47, 0x2e4d5353, 0x004f5450, // owner "SSM.PTO\0"
    1, 4, 32, 0 // version, participants, context bytes, reserved flags
};

const struct pto_runtime_group_context *__pto_runtime_group_context_get(void)
{
    return (const struct pto_runtime_group_context *)__ssm_pto_group_context;
}

__attribute__((noreturn)) void __pto_runtime_group_complete(
    const struct pto_runtime_group_context *context, int status)
{
    const uintptr_t service = 0x50544f01u;
    // Root manager request: normal ACRC permission/trap capture happens first.
    // These GPR arguments belong ONLY to the opt-in SSM.PTO v1 platform ABI.
    __asm__ volatile(
        "BSTART.SYS FALL\n"
        "ori %0, 0, ->a0\n"
        "ori %1, 0, ->a1\n"
        "ori %2, 0, ->x1\n"
        "acrc 0\n"
        "BSTOP\n"
        : : "r"(context), "r"((intptr_t)status), "r"(service)
        : "a0", "a1", "x1", "memory");
    __builtin_unreachable();
}

#ifndef SSM_HOSTED_USE_MUSL_CRT
// PE-private stack/TLS and the context slot are valid before this common
// entry. Only PE0 reads/writes process initialization state; workers perform
// their ordinary private worker bootstrap and wait at its generation-zero
// SYNC.CORE. No new barrier, host service or completion protocol is added.
extern void __ssm_pto_process_initialize(void) __attribute__((weak));
extern void __ssm_pto_invocation_initialize(
    const struct pto_runtime_group_context *) __attribute__((weak));
static unsigned ssm_process_initialized;

// Minimal freestanding common entry; link the actual compiler-rt group worker.
// Define SSM_HOSTED_USE_MUSL_CRT when an application's real CRT owns _start.
__attribute__((noreturn)) void _start(void)
{
    // The paired compiler must lower this to canonical SYS SSRGET 0x26.
    // Its generated envelope is part of the runtime acceptance tests.
    const unsigned pe = __builtin_linx_get_thread_idx();
    if (pe == 0) {
        if (!ssm_process_initialized) {
            if (__ssm_pto_process_initialize) __ssm_pto_process_initialize();
            ssm_process_initialized = 1;
            __asm__ volatile("" ::: "memory");
        }
        if (__ssm_pto_invocation_initialize)
            __ssm_pto_invocation_initialize(__pto_runtime_group_context_get());
        __asm__ volatile("" ::: "memory");
    }
    __linx_group_worker_start();
}
#endif
