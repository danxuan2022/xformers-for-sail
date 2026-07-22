/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#pragma once
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "extension/fmha/kernel_backward.h"
// ======== bf16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k32_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k64_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x64_k96_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k256_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k256_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k256_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k256_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_128x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_bf16_aligned_64x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 65536>::Params p);

template <typename T> void dispatch_cutlassB_bf16_sm80(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32, true>(), fmha_cutlassB_bf16_aligned_128x128_k32_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 32>(), fmha_cutlassB_bf16_aligned_128x128_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64, true>(), fmha_cutlassB_bf16_aligned_128x128_k64_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 64>(), fmha_cutlassB_bf16_aligned_128x128_k64_sm80);
    if (cc == 80 || cc == 89) cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 96>(), fmha_cutlassB_bf16_aligned_128x64_k96_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128, true>(), fmha_cutlassB_bf16_aligned_128x128_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 128, 128>(), fmha_cutlassB_bf16_aligned_128x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 128, true>(), fmha_cutlassB_bf16_aligned_64x128_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 128>(), fmha_cutlassB_bf16_aligned_64x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 256>(), fmha_cutlassB_bf16_aligned_128x128_k256_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 256>(), fmha_cutlassB_bf16_aligned_64x128_k256_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 128, 65536>(), fmha_cutlassB_bf16_aligned_128x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 64, 128, 65536>(), fmha_cutlassB_bf16_aligned_64x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 32>(), fmha_cutlassB_bf16_aligned_128x128_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 64>(), fmha_cutlassB_bf16_aligned_128x128_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 128, 128>(), fmha_cutlassB_bf16_aligned_128x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 128>(), fmha_cutlassB_bf16_aligned_64x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 256>(), fmha_cutlassB_bf16_aligned_128x128_k256_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 256>(), fmha_cutlassB_bf16_aligned_64x128_k256_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 128, 65536>(), fmha_cutlassB_bf16_aligned_128x128_k65536_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 64, 128, 65536>(), fmha_cutlassB_bf16_aligned_64x128_k65536_dropout_sm80);
}

// ======== f16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k32_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k64_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x64_k96_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 128, true>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 128, true>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k128_seqaligned_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 128, true>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k256_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k256_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k256_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k256_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_128x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f16_aligned_64x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f16_sm80(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32, true>(), fmha_cutlassB_f16_aligned_128x128_k32_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 32>(), fmha_cutlassB_f16_aligned_128x128_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64, true>(), fmha_cutlassB_f16_aligned_128x128_k64_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 64>(), fmha_cutlassB_f16_aligned_128x128_k64_sm80);
    if (cc == 80 || cc == 89) cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 96>(), fmha_cutlassB_f16_aligned_128x64_k96_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128, true>(), fmha_cutlassB_f16_aligned_128x128_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 128, 128>(), fmha_cutlassB_f16_aligned_128x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 128, true>(), fmha_cutlassB_f16_aligned_64x128_k128_seqaligned_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 128>(), fmha_cutlassB_f16_aligned_64x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 256>(), fmha_cutlassB_f16_aligned_128x128_k256_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 256>(), fmha_cutlassB_f16_aligned_64x128_k256_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 128, 65536>(), fmha_cutlassB_f16_aligned_128x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 64, 128, 65536>(), fmha_cutlassB_f16_aligned_64x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 32>(), fmha_cutlassB_f16_aligned_128x128_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 64>(), fmha_cutlassB_f16_aligned_128x128_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 128, 128>(), fmha_cutlassB_f16_aligned_128x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 128>(), fmha_cutlassB_f16_aligned_64x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 256>(), fmha_cutlassB_f16_aligned_128x128_k256_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 256>(), fmha_cutlassB_f16_aligned_64x128_k256_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 128, 65536>(), fmha_cutlassB_f16_aligned_128x128_k65536_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 64, 128, 65536>(), fmha_cutlassB_f16_aligned_64x128_k65536_dropout_sm80);
}

// ======== f32 / sm80 ========
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k32_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k64_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k128_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k256_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k256_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k65536_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 32>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 32>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k32_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 32>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 64>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 64>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k64_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 128>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k128_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k256_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 256>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 256>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k256_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 256>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_128x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 65536>::Params p);
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 65536>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 65536>::kMinBlocksPerSm)
fmha_cutlassB_f32_aligned_64x128_k65536_dropout_sm80(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 65536>::Params p);

template <typename T> void dispatch_cutlassB_f32_sm80(T cb, int cc) {
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 32>(), fmha_cutlassB_f32_aligned_128x128_k32_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 64>(), fmha_cutlassB_f32_aligned_128x128_k64_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 128>(), fmha_cutlassB_f32_aligned_128x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 128>(), fmha_cutlassB_f32_aligned_64x128_k128_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 256>(), fmha_cutlassB_f32_aligned_128x128_k256_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 256>(), fmha_cutlassB_f32_aligned_64x128_k256_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 128, 65536>(), fmha_cutlassB_f32_aligned_128x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 64, 128, 65536>(), fmha_cutlassB_f32_aligned_64x128_k65536_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 32>(), fmha_cutlassB_f32_aligned_128x128_k32_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 64>(), fmha_cutlassB_f32_aligned_128x128_k64_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 128>(), fmha_cutlassB_f32_aligned_128x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 128>(), fmha_cutlassB_f32_aligned_64x128_k128_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 256>(), fmha_cutlassB_f32_aligned_128x128_k256_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 256>(), fmha_cutlassB_f32_aligned_64x128_k256_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 128, 65536>(), fmha_cutlassB_f32_aligned_128x128_k65536_dropout_sm80);
    cb(AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 64, 128, 65536>(), fmha_cutlassB_f32_aligned_64x128_k65536_dropout_sm80);
}


template <typename DT, typename T>
void dispatch_cutlassB(T cb, int cc = 0) {

    if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassB_bf16_sm80(cb, cc);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassB_f16_sm80(cb, cc);
    }
    if (std::is_same<DT, float>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassB_f32_sm80(cb, cc);
    }
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
