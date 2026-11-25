#ifndef SSM_HOSTED_STARTUP_H
#define SSM_HOSTED_STARTUP_H

// Native Scalar GM defaults to intra scope. Declare the same contract at
// separately compiled hosted-runtime boundaries; it adds no synchronization.
#if defined(__linx) && __has_attribute(linx_scalar_shareability)
#define SSM_SCALAR_INTRA __attribute__((linx_scalar_shareability("intra")))
#else
#define SSM_SCALAR_INTRA
#endif

// Optional software hooks for the SSM.PTO v1 freestanding common entry.
// These are platform ABI names, not PTO instructions or compiler builtins.
struct pto_runtime_group_context;
#ifdef __cplusplus
extern "C" {
#endif
// Runs on PE0 once per loaded process, before invocation preparation. It must
// not retain an invocation context or require a four-PE collective.
SSM_SCALAR_INTRA void __ssm_pto_process_initialize(void);
// Runs on PE0 for each invocation, before the worker runtime's first SYNC.CORE.
// Workers may already be waiting there. Return means successful preparation;
// an unrecoverable fault follows the existing hosted Abort path, not release.
SSM_SCALAR_INTRA void __ssm_pto_invocation_initialize(const struct pto_runtime_group_context *);
#ifdef __cplusplus
}
#endif
#endif
