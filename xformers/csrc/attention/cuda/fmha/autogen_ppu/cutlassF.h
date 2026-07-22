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
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
#include "extension/fmha/kernel_forward.h"
#include "extension/fmha/kernel_forward_fa2.h"
// ======== bf16 / sm80 / fa2 ======== 
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k32x128x64_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k48x128x32_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k64x128x64_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k80x128x64_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k96x128x64_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k128x128x64_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k160x256x128_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k256x256x32_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k32x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k48x128x32_dropout_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k64x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k80x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k96x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k128x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k160x256x128_dropout_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_k256x256x32_dropout_sm80(typename AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>::Params p);

template <typename T> void dispatch_cutlassF_bf16_sm80_fa2(T cb, int cc) {
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>(), fmha_cutlassF_bf16_aligned_k32x128x64_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>(), fmha_cutlassF_bf16_aligned_k48x128x32_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>(), fmha_cutlassF_bf16_aligned_k64x128x64_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>(), fmha_cutlassF_bf16_aligned_k80x128x64_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>(), fmha_cutlassF_bf16_aligned_k96x128x64_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>(), fmha_cutlassF_bf16_aligned_k128x128x64_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>(), fmha_cutlassF_bf16_aligned_k160x256x128_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>(), fmha_cutlassF_bf16_aligned_k256x256x32_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>(), fmha_cutlassF_bf16_aligned_k32x128x64_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>(), fmha_cutlassF_bf16_aligned_k48x128x32_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>(), fmha_cutlassF_bf16_aligned_k64x128x64_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>(), fmha_cutlassF_bf16_aligned_k80x128x64_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>(), fmha_cutlassF_bf16_aligned_k96x128x64_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>(), fmha_cutlassF_bf16_aligned_k128x128x64_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>(), fmha_cutlassF_bf16_aligned_k160x256x128_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>(), fmha_cutlassF_bf16_aligned_k256x256x32_dropout_sm80);
}

// ======== f16 / sm80 / fa2 ======== 
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k32x128x64_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k48x128x32_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k64x128x64_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k80x128x64_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k96x128x64_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k128x128x64_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k160x256x128_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k256x256x32_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k32x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k48x128x32_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k64x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k80x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k96x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k128x128x64_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k160x256x128_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>::kNumThreads,
    AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_k256x256x32_dropout_sm80(typename AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f16_sm80_fa2(T cb, int cc) {
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, false, true>(), fmha_cutlassF_f16_aligned_k32x128x64_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, false, true>(), fmha_cutlassF_f16_aligned_k48x128x32_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, false, true>(), fmha_cutlassF_f16_aligned_k64x128x64_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, false, true>(), fmha_cutlassF_f16_aligned_k80x128x64_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, false, true>(), fmha_cutlassF_f16_aligned_k96x128x64_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, false, true>(), fmha_cutlassF_f16_aligned_k128x128x64_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, false, true>(), fmha_cutlassF_f16_aligned_k160x256x128_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, false, true>(), fmha_cutlassF_f16_aligned_k256x256x32_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64, 32, 64, true, true>(), fmha_cutlassF_f16_aligned_k32x128x64_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 48, 128, 32, 32, 32, true, true>(), fmha_cutlassF_f16_aligned_k48x128x32_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64, 32, 64, true, true>(), fmha_cutlassF_f16_aligned_k64x128x64_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 80, 128, 64, 32, 64, true, true>(), fmha_cutlassF_f16_aligned_k80x128x64_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 96, 128, 64, 32, 32, true, true>(), fmha_cutlassF_f16_aligned_k96x128x64_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 64, 32, 64, true, true>(), fmha_cutlassF_f16_aligned_k128x128x64_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 160, 256, 128, 32, 64, true, true>(), fmha_cutlassF_f16_aligned_k160x256x128_dropout_sm80);
    cb(AttentionKernelFA2<cutlass::half_t, cutlass::arch::Sm80, true, 256, 256, 32, 32, 32, true, true>(), fmha_cutlassF_f16_aligned_k256x256x32_dropout_sm80);
}

// ======== bf16 / sm80 / fa1 ======== 
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x64_k64_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_32x128_k128_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_32x128_k65536_gmem_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::Params p);

template <typename T> void dispatch_cutlassF_bf16_sm80_fa1(T cb, int cc) {
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>(), fmha_cutlassF_bf16_aligned_64x64_k64_rf_sm80);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>(), fmha_cutlassF_bf16_aligned_32x128_k128_rf_sm80);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>(), fmha_cutlassF_bf16_aligned_32x128_k65536_gmem_sm80);
}

// ======== f16 / sm80 / fa1 ======== 
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_k64_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_k128_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_k65536_gmem_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::Params p);

template <typename T> void dispatch_cutlassF_f16_sm80_fa1(T cb, int cc) {
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>(), fmha_cutlassF_f16_aligned_64x64_k64_rf_sm80);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>(), fmha_cutlassF_f16_aligned_32x128_k128_rf_sm80);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>(), fmha_cutlassF_f16_aligned_32x128_k65536_gmem_sm80);
}

// ======== f32 / sm80 / fa1 ======== 
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_k64_rf_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_k128_rf_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_k65536_gmem_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>::Params p);

template <typename T> void dispatch_cutlassF_f32_sm80_fa1(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, true, true, true, 64>(), fmha_cutlassF_f32_aligned_64x64_k64_rf_sm80);
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, true, true, true, 128>(), fmha_cutlassF_f32_aligned_32x128_k128_rf_sm80);
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, false, true, true, 65536>(), fmha_cutlassF_f32_aligned_32x128_k65536_gmem_sm80);
}


template <typename DT, typename T>
void dispatch_cutlassF_fa1(T cb, int cc = 0) {

        if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 89) {
        dispatch_cutlassF_bf16_sm80_fa1(cb, cc);
    }
        if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 89) {
        dispatch_cutlassF_f16_sm80_fa1(cb, cc);
    }
        if (std::is_same<DT, float>::value && 80 <= cc && cc < 89) {
        dispatch_cutlassF_f32_sm80_fa1(cb, cc);
    }
}

template <typename DT, typename T>
void dispatch_cutlassF_fa2(T cb, int cc = 0) {

        if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 89) {
        dispatch_cutlassF_bf16_sm80_fa2(cb, cc);
    }
        if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 89) {
        dispatch_cutlassF_f16_sm80_fa2(cb, cc);
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
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
