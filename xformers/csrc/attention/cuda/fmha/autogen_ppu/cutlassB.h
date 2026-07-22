/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#pragma once

#if defined(__HGGCCC__)
#ifndef ENABLE_AIU
#define ENABLE_AIU 1
#endif
#else
#define ENABLE_AIU 0
#endif
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "extension/fmha/kernel_backward.h"
#include "extension/fmha/kernel_backward_fa2.h"
// ======== bf16 / sm80 / fa2 ======== 
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k32_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k64_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k96_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k96_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k160_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k160_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k192_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k192_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k224_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k224_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k256_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k256_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k32_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k64_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k96_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k96_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k128_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k160_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k160_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k192_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k192_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k224_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k224_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k256_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x64_k256_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::Params p);

template <typename T> void dispatch_cutlassB_bf16_sm80_fa2(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>(), fmha_cutlassB_bf16_aligned_128x128_k32_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>(), fmha_cutlassB_bf16_aligned_128x128_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_128x128_k64_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_128x128_k64_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x128_k96_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x128_k96_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>(), fmha_cutlassB_bf16_aligned_64x128_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>(), fmha_cutlassB_bf16_aligned_64x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x64_k160_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x64_k160_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>(), fmha_cutlassB_bf16_aligned_64x64_k192_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>(), fmha_cutlassB_bf16_aligned_64x64_k192_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x64_k224_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x64_k224_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>(), fmha_cutlassB_bf16_aligned_64x64_k256_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>(), fmha_cutlassB_bf16_aligned_64x64_k256_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>(), fmha_cutlassB_bf16_aligned_128x128_k32_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>(), fmha_cutlassB_bf16_aligned_128x128_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_128x128_k64_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_128x128_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x128_k96_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x128_k96_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>(), fmha_cutlassB_bf16_aligned_64x128_k128_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>(), fmha_cutlassB_bf16_aligned_64x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x64_k160_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x64_k160_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>(), fmha_cutlassB_bf16_aligned_64x64_k192_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>(), fmha_cutlassB_bf16_aligned_64x64_k192_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x64_k224_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_bf16_aligned_64x64_k224_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>(), fmha_cutlassB_bf16_aligned_64x64_k256_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>(), fmha_cutlassB_bf16_aligned_64x64_k256_dropout_sm80);
}

// ======== f16 / sm80 / fa2 ======== 
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k32_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k64_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k96_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k96_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k160_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k160_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k192_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k192_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k224_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k224_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k256_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k256_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k32_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k64_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k96_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k96_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k128_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k160_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k160_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k192_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k192_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k224_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k224_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k256_dropout_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x64_k256_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>::Params p);

template <typename T> void dispatch_cutlassB_f16_sm80_fa2(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>(), fmha_cutlassB_f16_aligned_128x128_k32_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>(), fmha_cutlassB_f16_aligned_128x128_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_128x128_k64_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_128x128_k64_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x128_k96_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x128_k96_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>(), fmha_cutlassB_f16_aligned_64x128_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>(), fmha_cutlassB_f16_aligned_64x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x64_k160_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x64_k160_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>(), fmha_cutlassB_f16_aligned_64x64_k192_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>(), fmha_cutlassB_f16_aligned_64x64_k192_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x64_k224_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x64_k224_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>(), fmha_cutlassB_f16_aligned_64x64_k256_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>(), fmha_cutlassB_f16_aligned_64x64_k256_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32, true, true, true, 8, 4, 4, 4>(), fmha_cutlassB_f16_aligned_128x128_k32_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32, false, true, true, 8, 4, 4, 4>(), fmha_cutlassB_f16_aligned_128x128_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_128x128_k64_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_128x128_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 96, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x128_k96_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 96, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x128_k96_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 128, true, true, true, 8, 2, 4, 2>(), fmha_cutlassB_f16_aligned_64x128_k128_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 128, 128, false, true, true, 8, 2, 4, 2>(), fmha_cutlassB_f16_aligned_64x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 160, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x64_k160_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 160, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x64_k160_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 192, true, true, true, 8, 2, 2, 2>(), fmha_cutlassB_f16_aligned_64x64_k192_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 192, false, true, true, 8, 2, 2, 2>(), fmha_cutlassB_f16_aligned_64x64_k192_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 224, true, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x64_k224_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 224, false, true, true, 8, 2, 4, 4>(), fmha_cutlassB_f16_aligned_64x64_k224_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 256, true, true, true, 8, 2, 2, 2>(), fmha_cutlassB_f16_aligned_64x64_k256_dropout_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 64, 64, 256, false, true, true, 8, 2, 2, 2>(), fmha_cutlassB_f16_aligned_64x64_k256_dropout_sm80);
}

// ======== f32 / sm80 / fa1 ======== 
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 32, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 32, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 32, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 64, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 64, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 64, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 128, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 128, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 128, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 128, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 256, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 256, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k256_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 256, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 256, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 256, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k256_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 256, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 65536, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 65536, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 32, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 32, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 32, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 64, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 64, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 64, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 128, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 128, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 128, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 128, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 256, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 256, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k256_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 256, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 256, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 256, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k256_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 256, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 65536, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 65536, false>::Params p);

template <typename T> void dispatch_cutlassB_f32_sm80_fa1(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 32, false>(), fmha_cutlassB_f32_aligned_128x128_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 64, false>(), fmha_cutlassB_f32_aligned_128x128_k64_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 128, false>(), fmha_cutlassB_f32_aligned_128x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 128, false>(), fmha_cutlassB_f32_aligned_64x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 256, false>(), fmha_cutlassB_f32_aligned_128x128_k256_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 256, false>(), fmha_cutlassB_f32_aligned_64x128_k256_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 65536, false>(), fmha_cutlassB_f32_aligned_128x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 65536, false>(), fmha_cutlassB_f32_aligned_64x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 32, false>(), fmha_cutlassB_f32_aligned_128x128_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 64, false>(), fmha_cutlassB_f32_aligned_128x128_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 128, false>(), fmha_cutlassB_f32_aligned_128x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 128, false>(), fmha_cutlassB_f32_aligned_64x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 256, false>(), fmha_cutlassB_f32_aligned_128x128_k256_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 256, false>(), fmha_cutlassB_f32_aligned_64x128_k256_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 65536, false>(), fmha_cutlassB_f32_aligned_128x128_k65536_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 65536, false>(), fmha_cutlassB_f32_aligned_64x128_k65536_dropout_sm80);
}

// ======== bf16 / sm80 / fa1 ======== 
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 65536, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 65536, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 65536, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 65536, false>::Params p);

template <typename T> void dispatch_cutlassB_bf16_sm80_fa1(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 65536, false>(), fmha_cutlassB_bf16_aligned_128x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 65536, false>(), fmha_cutlassB_bf16_aligned_64x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 65536, false>(), fmha_cutlassB_bf16_aligned_128x128_k65536_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 65536, false>(), fmha_cutlassB_bf16_aligned_64x128_k65536_dropout_sm80);
}

// ======== f16 / sm80 / fa1 ======== 
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 65536, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 65536, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 65536, false>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 65536, false>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 65536, false>::Params p);

template <typename T> void dispatch_cutlassB_f16_sm80_fa1(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 65536, false>(), fmha_cutlassB_f16_aligned_128x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 65536, false>(), fmha_cutlassB_f16_aligned_64x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 65536, false>(), fmha_cutlassB_f16_aligned_128x128_k65536_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 65536, false>(), fmha_cutlassB_f16_aligned_64x128_k65536_dropout_sm80);
}


template <typename DT, typename T>
void dispatch_cutlassB_fa1(T cb, int cc = 0) {

        if (std::is_same<DT, float>::value && 80 <= cc && cc < 89) {
        dispatch_cutlassB_f32_sm80_fa1(cb, cc);
    }
        if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 89) {
        dispatch_cutlassB_bf16_sm80_fa1(cb, cc);
    }
        if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 89) {
        dispatch_cutlassB_f16_sm80_fa1(cb, cc);
    }
}

template <typename DT, typename T>
void dispatch_cutlassB_fa2(T cb, int cc = 0) {

        if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 89) {
        dispatch_cutlassB_bf16_sm80_fa2(cb, cc);
    }
        if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 89) {
        dispatch_cutlassB_f16_sm80_fa2(cb, cc);
    }
}

// kNumWarpsPerBlock = kQueriesPerBlock * kKeysPerBlock / (32 * 32);
template <int kNumWarpsPerBlock, int kKeysPerBlock, bool kSingleValueIteration>
struct MaxkTrait {
    static constexpr int kMaxK = 0;
};

template <>
struct MaxkTrait<4, 64, true> {
    static constexpr int kMaxK = 64;
};
template <>
struct MaxkTrait<4, 128, true> {
    static constexpr int kMaxK = 128;
};

template <>
struct MaxkTrait<4, 128, false> {
    static constexpr int kMaxK = 65536;
};
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
