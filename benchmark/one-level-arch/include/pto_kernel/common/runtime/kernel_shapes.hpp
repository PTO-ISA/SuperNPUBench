#ifndef PTO_COMMON_RUNTIME_KERNEL_SHAPES_HPP
#define PTO_COMMON_RUNTIME_KERNEL_SHAPES_HPP

#include <pto_kernel/common/runtime/kernel_env.hpp>

#ifndef PTO_ATTENTION_SMOKE_SEQ
#define PTO_ATTENTION_SMOKE_SEQ 16
#endif

#ifndef PTO_ATTENTION_LARGE_SMOKE_SEQ
#define PTO_ATTENTION_LARGE_SMOKE_SEQ 16
#endif

#ifndef PTO_ATTENTION_SMOKE_QD
#define PTO_ATTENTION_SMOKE_QD 16
#endif

#ifndef PTO_ATTENTION_SMOKE_VD
#define PTO_ATTENTION_SMOKE_VD 16
#endif

#ifndef PTO_ATTENTION_SMALL_SMOKE_QD
#define PTO_ATTENTION_SMALL_SMOKE_QD 4
#endif

#ifndef PTO_ATTENTION_MASKED_SMOKE_SEQ
#define PTO_ATTENTION_MASKED_SMOKE_SEQ 18
#endif

#ifndef PTO_ATTENTION_MASKED_SMOKE_QD
#define PTO_ATTENTION_MASKED_SMOKE_QD 16
#endif

#ifndef PTO_ATTENTION_MASKED_SMOKE_VD
#define PTO_ATTENTION_MASKED_SMOKE_VD 16
#endif

namespace pto {
namespace kernels {
namespace shapes {

inline constexpr int kMemoryRows = env::select(32, 1024);
inline constexpr int kMemoryCols = env::select(32, 1024);

inline constexpr int kMatmulM = env::select(16, 256);
inline constexpr int kMatmulN = env::select(16, 256);
inline constexpr int kMatmulK = env::select(16, 256);
inline constexpr int kMatmulReuseExtent = env::select(16, 64);

inline constexpr int kAttentionSeq = env::select(PTO_ATTENTION_SMOKE_SEQ, 128);
inline constexpr int kAttentionLargeSeq =
    env::select(PTO_ATTENTION_LARGE_SMOKE_SEQ, 256);
inline constexpr int kAttentionQD = env::select(PTO_ATTENTION_SMOKE_QD, 16);
inline constexpr int kAttentionVD = env::select(PTO_ATTENTION_SMOKE_VD, 16);
inline constexpr int kAttentionSmallQD =
    env::select(PTO_ATTENTION_SMALL_SMOKE_QD, 4);
inline constexpr int kAttentionMaskedSeq =
    env::select(PTO_ATTENTION_MASKED_SMOKE_SEQ, 130);
inline constexpr int kAttentionMaskedQD =
    env::select(PTO_ATTENTION_MASKED_SMOKE_QD, 16);
inline constexpr int kAttentionMaskedVD =
    env::select(PTO_ATTENTION_MASKED_SMOKE_VD, 16);

inline constexpr int kMlaInputDim = 16;
inline constexpr int kMlaLatentDim = 4;
inline constexpr int kMlaOutputDim = 16;

inline constexpr int kNormTokens = env::select(16, 128);

inline constexpr int kSmallVector = env::select(64, 1024);
inline constexpr int kSmallTable = env::select(97, 2048);

} // namespace shapes
} // namespace kernels
} // namespace pto

#endif // PTO_COMMON_RUNTIME_KERNEL_SHAPES_HPP
