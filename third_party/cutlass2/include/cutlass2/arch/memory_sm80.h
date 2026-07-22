/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Architecture-specific operators on memory added for SM80
*/

#pragma once

#include "cutlass2/cutlass2.h"
#include "cutlass2/arch/memory_sm75.h"
#include "cutlass2/arch/cache_operation.h"
#include "cuda_pipeline.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) || ((defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && (defined(ACOMPUTE_VERSION) && ACOMPUTE_VERSION > 10000))
  #define CUDA_CP_ASYNC_ACTIVATED 1
#else
  #define CUDA_CP_ASYNC_ACTIVATED 0
#endif

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Initiates an asynchronous copy from global memory to shared memory.
///
/// LDGSTS
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async;

/// Initiates an asynchronous copy from global memory to shared memory. Rather than predicate
/// the entire transfer, zeros are written to SMEM if the guard predicate is false.
///
/// LDGSTS
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async_zfill;

/// Initiates an asynchronous copy from global memory to shared memory. Rather than predicate
/// the entire transfer, nans (0x7eff) are written to SMEM if the guard predicate is false.
///
/// LDGSTS
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always,
    bool is_elem_fp32 = false>
struct cp_async_nan;

static const uint32_t OOB_NAN_F16 = 0x7eff;
static const uint32_t OOB_NAN_F16x2 = ((OOB_NAN_F16 << 16) | OOB_NAN_F16);
static const uint32_t OOB_NAN_FP32 = 0x7fe00000;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Always> {

  /// Copy
  CUTLASS_DEVICE
  cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    #if CUDA_CP_ASYNC_ACTIVATED

      // Make sure the size is supported.
      static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
                "Size is not supported");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
          "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)pred_guard),
          "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));

    #elif defined (__HGGCCC__)
      if (pred_guard) {
        __pipeline_memcpy_async(smem_ptr, global_ptr, SizeInBytes, 0);
      }
    #else
      using AccessType  = Array<uint8_t, SizeInBytes>;

      if (pred_guard) {
        *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
      }
    #endif
  }
};

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async_zfill<SizeInBytes, CacheOperation::Always> {

  /// Copy with zero fill
  CUTLASS_DEVICE
  cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    #if CUDA_CP_ASYNC_ACTIVATED

      // Make sure the size is supported.
      static_assert((SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
                "Size is not supported");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
      int src_in_bytes = (pred_guard ? SizeInBytes : 0);

      asm volatile(
        // "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));
    #else
      using AccessType  = Array<uint8_t, SizeInBytes>;

      #if defined (__HGGCCC__)
      int zfill_bytes = (pred_guard ? 0 : SizeInBytes);
      __ppu_pipeline_memcpy_async_zfill(smem_ptr, global_ptr, SizeInBytes, zfill_bytes);
      // __ppu_prefetch_nonebulk_L2(const_cast<void*>(static_cast<const void*>(static_cast<const char*>(global_ptr) + 128)));
      #else
      if (pred_guard) {
        *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
      }
      else {
        AccessType zeros;
        zeros.clear();
        *static_cast<AccessType *>(smem_ptr) = zeros;
      }
      #endif

    #endif
  }
};

/// Partial specialization for loading 32b/16b and cached at all levels
template <bool is_elem_fp32>
struct cp_async_nan<16, CacheOperation::Always, is_elem_fp32> {
  static int const kSizeInBytes = 16;

  /// Copy with nan fill
  CUTLASS_DEVICE
  cp_async_nan(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    #if CUDA_CP_ASYNC_ACTIVATED || defined(__HGGC_ARCH__)
      static const uint32_t OOB_NAN_VALUE = is_elem_fp32 ? OOB_NAN_FP32 : OOB_NAN_F16x2;
      static __constant__ uint4 OOB_NAN = {OOB_NAN_VALUE,  OOB_NAN_VALUE,
                                           OOB_NAN_VALUE,  OOB_NAN_VALUE};

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
          "  @p cp.async.ca.shared.global.L2::128B [%1], [%2], %3;\n"
#else
          "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
#endif
          "  @!p st.shared.v4.u32 [%1], {%4, %5, %6, %7};\n"
          "}\n"
          :
          : "r"((int)pred_guard), "r"(smem_int_ptr), "l"(global_ptr),
            "n"(kSizeInBytes), "r"(OOB_NAN.x), "r"(OOB_NAN.y), "r"(OOB_NAN.z),
            "r"(OOB_NAN.w));

    #else

      CUTLASS_UNUSED(smem_ptr);
      CUTLASS_UNUSED(global_ptr);
      CUTLASS_UNUSED(pred_guard);
      CUTLASS_NOT_IMPLEMENTED();

    #endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Global> {

  /// Copy
  CUTLASS_DEVICE
  cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    #if CUDA_CP_ASYNC_ACTIVATED

      static_assert(SizeInBytes == 16,
        "cp.async only supports CacheOperation::Global when access size is 16B.");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
          "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)pred_guard),
          "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes));

    #elif defined (__HGGCCC__)
      if (pred_guard) {
        __pipeline_memcpy_async(smem_ptr, global_ptr, SizeInBytes, 0);
      }
    #else
      using AccessType  = Array<uint8_t, SizeInBytes>;

      if (pred_guard) {
        *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
      }
    #endif
  }
};

template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async_zfill<SizeInBytes, CacheOperation::Global> {

  /// Copy with zero fill
  CUTLASS_DEVICE
  cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    #if CUDA_CP_ASYNC_ACTIVATED

      static_assert(SizeInBytes == 16,
        "cp.async only supports CacheOperation::Global when access size is 16B.");

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
      int src_in_bytes = (pred_guard ? SizeInBytes : 0);

      asm volatile(
        // "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes));

    #else
      using AccessType  = Array<uint8_t, SizeInBytes>;

      #if defined (__HGGCCC__)
      int zfill_bytes = (pred_guard ? 0 : SizeInBytes);
      __ppu_pipeline_memcpy_async_zfill(smem_ptr, global_ptr, SizeInBytes, zfill_bytes);
      // __ppu_prefetch_nonebulk_L2(const_cast<void*>(static_cast<const void*>(static_cast<const char*>(global_ptr) + 128)));
      #else
      if (pred_guard) {
        *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
      }
      else {
        AccessType zeros;
        zeros.clear();
        *static_cast<AccessType *>(smem_ptr) = zeros;
      }
      #endif
    #endif
  }
};

/// Partial specialization for loading 32b/16b and cached at global level
template <bool is_elem_fp32>
struct cp_async_nan<16, CacheOperation::Global, is_elem_fp32> {
  static int const kSizeInBytes = 16;

  /// Copy with nan fill
  CUTLASS_DEVICE
  cp_async_nan(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    #if CUDA_CP_ASYNC_ACTIVATED || defined(__HGGC_ARCH__)
      static const uint32_t OOB_NAN_VALUE = is_elem_fp32 ? OOB_NAN_FP32 : OOB_NAN_F16x2;
      static __constant__ uint4 OOB_NAN = {OOB_NAN_VALUE,  OOB_NAN_VALUE,
                                           OOB_NAN_VALUE,  OOB_NAN_VALUE};

      unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);

      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
          "  @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
#else
          "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
#endif
          "  @!p st.shared.v4.u32 [%1], {%4, %5, %6, %7};\n"
          "}\n"
          :
          : "r"((int)pred_guard), "r"(smem_int_ptr), "l"(global_ptr),
            "n"(kSizeInBytes), "r"(OOB_NAN.x), "r"(OOB_NAN.y), "r"(OOB_NAN.z),
            "r"(OOB_NAN.w));

    #else

      CUTLASS_UNUSED(smem_ptr);
      CUTLASS_UNUSED(global_ptr);
      CUTLASS_UNUSED(pred_guard);
      CUTLASS_NOT_IMPLEMENTED();

    #endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Establishes an ordering w.r.t previously issued cp.async instructions. Does not block.
CUTLASS_DEVICE
void cp_async_fence() {
  #if CUDA_CP_ASYNC_ACTIVATED
  asm volatile("cp.async.commit_group;\n" ::);
  #elif defined (__HGGCCC__)
  __pipeline_commit();
  #endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Blocks until all but <N> previous cp.async.commit_group operations have committed.
template <int N>
CUTLASS_DEVICE void cp_async_wait() {
  #if CUDA_CP_ASYNC_ACTIVATED
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  #elif defined (__HGGCCC__)
  __pipeline_wait_prior(N);
  #endif
}

/// Blocks until all previous cp.async.commit_group operations have committed.
template <>
CUTLASS_DEVICE void cp_async_wait<0>() {
  #if CUDA_CP_ASYNC_ACTIVATED
  asm volatile("cp.async.wait_all;\n" ::);
  #elif defined (__HGGCCC__)
  __pipeline_wait_prior(0);
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace arch
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
