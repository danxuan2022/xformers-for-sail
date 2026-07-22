/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holdvr nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <cmath>
#include <type_traits>
#include <vector>

#include <cuda_fp16.h>
#include <curand_kernel.h>

#ifdef HAS_PYTORCH
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#endif

#include "cutlass2/cutlass2.h"
#include "cutlass2/epilogue/thread/linear_combination.h"
#include "cutlass2/epilogue/thread/scale_type.h"
#include "cutlass2/fast_math.h"
#include "cutlass2/functional.h"
#include "cutlass2/gemm/gemm.h"
#include "cutlass2/layout/matrix.h"
#include "cutlass2/layout/vector.h"
#include "cutlass2/numeric_conversion.h"
#include "cutlass2/numeric_types.h"
#include "cutlass2/tensor_ref.h"

#include "debug_utils.h"
#include "gemm_kernel_utils.h"

#include "cutlass2/epilogue/thread/linear_combination_relu.h"
#include "cutlass2/epilogue/threadblock/epilogue_smem_accumulator.h"
#include "cutlass2/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass2/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass2/gemm/device/default_gemm_configuration.h"
#include "cutlass2/gemm/kernel/default_gemm.h"
#include "cutlass2/gemm/threadblock/default_mma.h"
#include "cutlass2/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass2/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass2/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass2/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass2/integer_subbyte.h"
#include "cutlass2/matrix_shape.h"
#include "cutlass2/platform/platform.h"
#include "cutlass2/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass2/transform/threadblock/vector_iterator.h"
#include "epilogue_pipelined.h"
#include "iterators/epilogue_predicated_tile_iterator.h"

#include "gemm/custom_mma.h"
#include "find_default_mma.h"
#include "gemm/mma_accum_lambda_iterator.h"
#include "gemm/mma_from_smem.h"
#include "transform/tile_smem_loader.h"
#include "cutlass2/semaphore.h"

#include <inttypes.h>

using namespace gemm_kernel_utils;

namespace {

template <typename FragmentType, int32_t kNumThreads>
struct GmemTile {
  /*
    Helper functions to efficient store/load RF to gmem

    GEMM accumulators have a particular format on A100, and
    it takes some compute/shared-memory to rearrange them to
    a RowMajor or ColumnMajor format in global memory through
    an Epilogue. The same complexity goes for loading into RF.

    This class loads/stores RF as they are, and can be used for
    efficient accumulation across gemms for instance:

    ```
    GmemTile tile;
    for (int i = 0; i < N; ++i) {
      // ...

      Fragment accum;
      if (i == 0) {
        accum.clear();
      } else {
        tile.load(accum);
      }
      mma(accum, ...);
      if (i < N-1) {
        // Store for next GEMM
        tile.store(accum);
      } else {
        // Store in tensor (eg RowMajor)
        epilogue(accum);
      }

      // ...
    }
    ```
  */

  // 128bits per thread
  using AccessType = cutlass::Array<float, 1>;
  static constexpr int32_t kBytes = sizeof(AccessType);
  static constexpr int32_t kStride = kNumThreads * AccessType::kElements;
  static constexpr int32_t kNumIters =
      FragmentType::kElements / AccessType::kElements;
  static constexpr int32_t kElementsStored =
      kNumThreads * FragmentType::kElements;
  static_assert(
      FragmentType::kElements % AccessType::kElements == 0,
      "fragment not aligned on 128 bits");

  float* ptr;

  CUTLASS_DEVICE void load(FragmentType& fragment, int thread_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumIters; ++i) {
      AccessType* __restrict__ gmem_ptr = reinterpret_cast<AccessType*>(
          ptr + thread_id * AccessType::kElements + i * kStride);
      AccessType sub_fragment;
      cutlass::arch::global_load<AccessType, kBytes>(
          sub_fragment, gmem_ptr, true);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessType::kElements; ++j) {
        fragment[i * AccessType::kElements + j] = sub_fragment[j];
      }
    }
  }

  CUTLASS_DEVICE void store(FragmentType const& fragment, int thread_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumIters; ++i) {
      AccessType* __restrict__ gmem_ptr = reinterpret_cast<AccessType*>(
          ptr + thread_id * AccessType::kElements + i * kStride);
      AccessType sub_fragment;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessType::kElements; ++j) {
        sub_fragment[j] = fragment[i * AccessType::kElements + j];
      }
      cutlass::arch::global_store<AccessType, kBytes>(
          sub_fragment, gmem_ptr, true);
    }
  }

  CUTLASS_DEVICE void storeAtomicAdd(
      FragmentType const& fragment,
      int thread_id) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumIters; ++i) {
      float* gmem_ptr = ptr + thread_id * AccessType::kElements + i * kStride;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < AccessType::kElements; ++j) {
        float val = fragment[i * AccessType::kElements + j];
        float* ptr = gmem_ptr + j;

        atomicAdd(reinterpret_cast<float*>(__cvta_generic_to_global(ptr)), val);
      }
    }
  }
};

struct AtomicLock {
  CUTLASS_DEVICE static void acquire(
      int32_t* lock,
      int set_val,
      int thread_id) {
    if (thread_id == 0) {
      while (atomicCAS(lock, 0 /*cmp*/, set_val /*setval*/) != set_val) {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(40);
#endif
      }
    }
    __syncthreads();
  }
  CUTLASS_DEVICE static void release(int32_t* lock, int thread_id) {
    if (thread_id == 0) {
      int status = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      asm volatile("st.global.release.gpu.b32 [%0], %1;\n"
                   :
                   : "l"(lock), "r"(status));
#else
      asm volatile("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#endif
    }
  }
};

template <typename scalar_t, typename Arch>
constexpr int getWarpsPerSmBw() {
  // bool is_half = !cutlass::platform::is_same<scalar_t, float>::value;
  // if (Arch::kMinComputeCapability >= 80) {
  //   return is_half ? 12 : 8;
  // }
  return 8;
}

enum CustomMaskType {
  NoCustomMask = 0,
  CausalFromTopLeft = 1,
  CausalFromBottomRight = 2,
  NumCustomMaskTypes,
};

} //namespace

template <
  // which arch we target (eg `cutlass::arch::Sm80`)
  typename ArchTag_,
  // input/output type
  typename scalar_t_,
  // run optimized kernel because memory accesses will be aligned
  bool kIsAligned_,
  // use dropout if enabled
  bool kApplyDropout_,
  // when doing a GEMM, preload the next one (uses more shmem)
  bool kPreload_,
  // block dimensions
  int kBlockSizeI_,
  int kBlockSizeJ_,
  // upperbound on `max(value.shape[-1], query.shape[-1])`
  int kMaxK_ = (int)cutlass::platform::numeric_limits<int32_t>::max(),
  // assumes that `cu_seqlen` is None, and
  // (1) `num_queries % kBlockSizeI == 0`
  // (2) `num_keys % kBlockSizeJ == 0`
  bool kKeysQueriesAlignedToBlockSize_ = false,
  // Allows to parallelize across keys
  bool kEnableSplitKeys_ = true,
  // Use flath attention v2
  bool kEnableV2_ = false,
  // warp dimensions
  int kWarpNum = 8,
  int kWarpDp  = 4,
  int kWarpKv  = 4,
  int kWarpDq  = 4>
struct AttentionBackwardKernel {};