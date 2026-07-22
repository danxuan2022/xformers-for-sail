/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"

#if defined(__HGGCCC__)
#ifndef ENABLE_AIU
#define ENABLE_AIU 1
#endif
#else
#define ENABLE_AIU 0
#endif
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "accutlass.h"
#include "extension/fmha/kernel_backward.h"
#include "extension/fmha/kernel_backward_fa2.h"
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k192_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassB_bf16_aligned_64x64_k192_seqaligned_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k192_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassB_bf16_aligned_64x64_k192_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
