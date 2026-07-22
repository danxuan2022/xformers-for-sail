/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
#pragma once
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
#include "extension/fmha/kernel_forward.h"

// ======== bf16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_64x64_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_32x128_rf_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kMinBlocksPerSm)
fmha_cutlassF_bf16_aligned_32x128_gmem_sm80(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::Params p);

template <typename T> void dispatch_cutlassF_bf16_sm80(T cb, int cc) {
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>(), fmha_cutlassF_bf16_aligned_64x64_rf_sm80);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>(), fmha_cutlassF_bf16_aligned_32x128_rf_sm80);
    cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>(), fmha_cutlassF_bf16_aligned_32x128_gmem_sm80);
}

// ======== f16 / sm80 ========
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_64x64_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_rf_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f16_aligned_32x128_gmem_sm80(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f16_sm80(T cb, int cc) {
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>(), fmha_cutlassF_f16_aligned_64x64_rf_sm80);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>(), fmha_cutlassF_f16_aligned_32x128_rf_sm80);
    cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>(), fmha_cutlassF_f16_aligned_32x128_gmem_sm80);
}

// ======== f32 / sm80 ========
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_64x64_rf_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, true, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_rf_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, true, true, true>::Params p);
__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kMinBlocksPerSm)
fmha_cutlassF_f32_aligned_32x128_gmem_sm80(typename AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, false, true, true>::Params p);

template <typename T> void dispatch_cutlassF_f32_sm80(T cb, int cc) {
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, true, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm80);
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, true, true, true>(), fmha_cutlassF_f32_aligned_32x128_rf_sm80);
    cb(AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, false, true, true>(), fmha_cutlassF_f32_aligned_32x128_gmem_sm80);
}


template <typename DT, typename T>
void dispatch_cutlassF(T cb, int cc = 0) {
    if (std::is_same<DT, cutlass::bfloat16_t>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassF_bf16_sm80(cb, cc);
    }
    if (std::is_same<DT, cutlass::half_t>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassF_f16_sm80(cb, cc);
    }
    if (std::is_same<DT, float>::value && 80 <= cc && cc < 100) {
        dispatch_cutlassF_f32_sm80(cb, cc);
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
