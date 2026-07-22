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
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
#include "accutlass.h"
#include "extension/fmha/kernel_forward.h"
#include "extension/fmha/kernel_forward_fa2.h"
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k32x128x64_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k32x128x64_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k48x128x32_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k48x128x32_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k64x128x64_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k64x128x64_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k80x128x64_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k80x128x64_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k96x128x64_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k96x128x64_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k128x128x64_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k128x128x64_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k160x256x128_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k160x256x128_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k256x256x32_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k256x256x32_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k32x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k32x128x64_dropout_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k48x128x32_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k48x128x32_dropout_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k64x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k64x128x64_dropout_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k80x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k80x128x64_dropout_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k96x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k96x128x64_dropout_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k128x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k128x128x64_dropout_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k160x256x128_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k160x256x128_dropout_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k256x256x32_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_k256x256x32_dropout_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_k64_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_64x64_k64_rf_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_k128_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_32x128_k128_rf_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_k65536_gmem_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == 890
  return;
#else
  printf(
        "FATAL: kernel `fmha_cutlassF_f16_aligned_32x128_k65536_gmem_sm80` is for sm80/sm89, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
