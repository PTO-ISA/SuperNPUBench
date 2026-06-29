# Kernels - Operator Implementations

This directory contains the core operator implementations of SuperNPUBench, organized in a header-only manner for easy reuse and integration.

## Operator List

### 1. Matmul (Matrix Multiplication)

**Directory**: `matmul/`

**Implementation Files**:
- `matmul.hpp` - General matrix multiplication
  - Supports FP32/FP16/FP8 data types
  - Provides mask, dynamic, vec variants
  - Supports A/B tile reuse optimization
  
- `matmul_mx.hpp` - MX quantized matrix multiplication
  - Supports FP4 x FP4 quantized computation
  - Supports BF16 x FP4 mixed precision
  - Implements microscaling factor handling

**Typical Scenarios**:
- Matrix multiplication in large model inference
- Mixed precision computation for quantized models
- Supports dynamic and static shapes

### 2. Flash Attention (FA)

**Directory**: `fa/`

**Implementation Files**:
- `fa_2d_unroll.hpp` - Basic 2D unroll version
  - Supports 2D unroll optimization for X/Y dimensions
  - Suitable for sequence lengths 256/512
  
- `fa_2d_unroll_pto.hpp` - PTO-style 2D unroll
  - Uses PTO tile programming paradigm
  - Optimized memory access pattern
  
- `fa_unalign_2d_unroll.hpp` - Unaligned boundary handling
  - Handles cases where sequence length is not a multiple of tile size
  
- `fa_hif4.hpp` - HIF4 quantized version
  - Supports FP4 quantized computation
  - Suitable for low-precision inference
  
- `fa_dcore.hpp` - DCore optimized version
  - Optimized for specific hardware
  
- `fa_utils.h` / `fa_fp4_utils.h` - Utility functions
  - Shared helper functions

**Typical Scenarios**:
- Attention mechanism in Transformer models
- Long sequence processing
- Quantized attention computation

### 3. Broadcast

**Directory**: `broadcast/`

**Implementation Files**:
- `broadcast.hpp` - Basic broadcast implementation
- `broadcast_07.hpp` - 2D broadcast (1334,1) → (1334,129)
- `broadcast_019.hpp` - 3D broadcast (1280,1,49) → (1280,8,49)
- `broadcast_039.hpp` - 4D broadcast (1,128,1,16) → (64,128,8,16)
- `broadcast_Hunyuan.hpp` - 5D broadcast (1,1,1,65,128) → (1,1,7,65,128)
- `broadcast_vec_*.hpp` - Vectorized versions
  - Optimized vectorized implementations for specific dimensions

**Typical Scenarios**:
- Tensor shape expansion
- Dimension alignment in batch processing
- Broadcast operations in models

### 4. Reduction

**Directory**: `reduction/`

**Implementation Files**:
- `reducemax_colvec.hpp` - Column-wise max reduction
- `reducemax_rowvec.hpp` - Row-wise max reduction
- `reducesum_colvec.hpp` - Column-wise sum reduction
- `reducesum_rowvec.hpp` - Row-wise sum reduction
- `*_single*.hpp` - Single-step optimized versions
- `*_unalign*.hpp` - Unaligned optimized versions
- `cumsum_*.hpp` - Cumulative sum implementations
- `reduceprod_*.hpp` - Product reduction

**Typical Scenarios**:
- Max/sum computation in Softmax
- Statistics computation in LayerNorm
- Reduction operations in attention mechanism

### 5. GELU (Activation Function)

**Directory**: `element_wise/`

**Implementation Files**:
- `gelu.hpp` - New polynomial fitting implementation
  - Optimized polynomial coefficients
  - Supports exact mode and tanh approximation
  
- `gelu_origin.hpp` - Original implementation
  - Exact computation based on erf function
  - Approximation based on tanh

**Typical Scenarios**:
- Activation function in Transformer models
- BERT/GPT and other models

### 6. Gather

**Directory**: `gather/`

**Implementation Files**:
- `gather.hpp` - General data gathering implementation
  - Supports large-scale input
  - Supports various indexing modes

**Typical Scenarios**:
- Embedding lookup
- Sequence sampling
- Data rearrangement

### 7. Concat

**Directory**: `concat/`

**Implementation Files**:
- `concat_gather.hpp` - Gather-based concatenation
- `concat_scatter.hpp` - Scatter-based concatenation

**Typical Scenarios**:
- Feature concatenation
- Multi-input merging
- KV Cache extension

### 8. Transpose

**Directory**: `transpose/`

**Implementation Files**:
- `transpose.hpp` - Basic transpose implementation
  - Supports 3D~6D tensors
  - Supports multiple data types
  
- `transpose_vector_007.hpp` - Vectorized version 007
- `transpose_vector_050.hpp` - Vectorized version 050

**Typical Scenarios**:
- QKV transpose in attention mechanism
- Tensor format conversion
- Memory layout optimization

### 9. Utils

**Directory**: `utils/`

**Implementation Files**:
- `layout_transform.hpp` - Layout transformation utilities
  - ND to ZZ offset calculation
  - ND to NN offset calculation

## Design Principles

1. **Header-only**: All implementations are header-only for easy integration and reuse
2. **PTO Paradigm**: Uses PTO tile programming paradigm with unified tile operation interface
3. **Templated**: Uses C++ templates for type and dimension parameterization
4. **Optimization-oriented**: Provides multiple optimized versions for different scenarios

## Usage

```cpp
// Include header
#include "matmul/matmul.hpp"

// Call operator
matmul_mask<float, M, N, K, tM, tN, tK>(dst, src0, src1);
```

## Performance Optimization Tips

1. **Tile Size Selection**: Choose appropriate tM/tN/tK based on hardware characteristics
2. **Data Type**: Choose appropriate data type based on precision requirements
3. **Reuse Strategy**: Use reuseA/reuseB optimization for repeated computations
4. **Vectorization**: Prefer vectorized versions (vec/vector) when available

## Related Documentation

- [Top-level README](../README.md) - Project overview
- [Test Suites](../test/kernel/README.md) - Test documentation
- [Compilation Report](../COMPILATION_REPORT.md) - Compilation statistics
