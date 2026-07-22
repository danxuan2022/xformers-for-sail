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
    \brief Architecture-specific operators on memory added for SM75
*/

#pragma once

#include "cutlass2/array.h"
#include "cutlass2/layout/matrix.h"

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Layout of destination matrix (column-major implies transpose)
  typename Layout,
  /// .x1, .x2, or .x4
  int MatrixCount
>
inline __device__ void ldsm(Array<unsigned, MatrixCount> & D, void const* ptr);

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
template <
  /// Layout of destination matrix (column-major implies transpose)
  typename Layout,
  /// .x1, .x2, or .x4
  int MatrixCount,
  /// Element data type
  typename Element
>
struct ppu_ldsm {
  public:
    CUTLASS_DEVICE
    ppu_ldsm() {}

    CUTLASS_DEVICE
    void operator() (Array<unsigned, MatrixCount> &_D, void const* _ptr) {}
};
#endif

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
template <
  /// Layout of destination matrix (column-major implies transpose)
  typename Layout,
  /// .x1, .x2, or .x4
  int MatrixCount
>
inline __device__ void tsm_ld_ncom(Array<unsigned, MatrixCount> & D, void const* ptr);

template <
  /// Layout of destination matrix (column-major implies transpose)
  typename Layout,
  /// .x1, .x2, or .x4
  int MatrixCount
>
inline __device__ void vmem_ld(Array<unsigned, MatrixCount> & D, void const* ptr);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Determine the appropriate way to target PTX's "ldmatrix" instruction.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2) || (__CUDACC_VER_MAJOR__ >= 11)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
#define CUDA_LDMATRIX_ACTIVATED 1
#endif

#define CUDA_LDMATRIX_SUPPORTED 1
#endif

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
// PPU ldmatrix
#define CUDA_LDMATRIX_ACTIVATED 1
#define CUDA_LDMATRIX_SUPPORTED 1
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
/*
#if ! defined(CUDA_NVVM_GET_SMEM_POINTER_SUPPORTED) && (__CUDACC_VER_MAJOR__ > 10)
  #define CUDA_NVVM_GET_SMEM_POINTER_SUPPORTED 1
#endif
#if ! defined(CUDA_NVVM_GET_SMEM_POINTER_SUPPORTED)
  #define CUDA_NVVM_GET_SMEM_POINTER_SUPPORTED ((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 1))
#endif

#if ! defined(CUDA_NVVM_GET_SMEM_POINTER_ENABLED)
  #define CUDA_NVVM_GET_SMEM_POINTER_ENABLED CUDA_NVVM_GET_SMEM_POINTER_SUPPORTED
#endif
*/

#if (! defined (__clang__) && __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)
  extern "C" {
  //
  // This NVVM intrinsic is subject to change in future versions of CUDA.
  // Clients should not call it directly. Rather, they should use the
  // cutlass::arch::ldsm<>() template.
  //
  __device__ uint32_t __nvvm_get_smem_pointer(void *);
  }
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned cutlass_get_smem_pointer(void *ptr) {

// We prefer to use the new CVTA intrinsics if they are available, otherwise we will fall back to
// the previous internal intrinsics if they are available.
#if (! defined (__clang__) && defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 11)
  //
  // This NVVM intrinsic converts an address in shared memory to a plain
  // unsigned integer. This is necessary to pass to shared memory instructions
  // in inline PTX.
  //
  // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only available in 10.2].
  //
  //__device__ CUsize __cvta_generic_to_shared(void* ptr);

  /// CUTLASS helper to get SMEM pointer
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));

#elif (! defined (__clang__) && defined(__CUDA_ARCH__) &&  __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)

  return __nvvm_get_smem_pointer(ptr);

#elif defined(__CUDA_ARCH__)

  uint32_t smem_ptr;

  asm(
  "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
    : "=r"(smem_ptr) : "l"(ptr));

  return smem_ptr;

#elif (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
  /// PPU use CUTLASS helper to get SMEM pointer
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
#else

    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();
    return 0;
#endif
}

/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned cutlass_get_smem_pointer(void const *ptr) {
  return cutlass_get_smem_pointer(const_cast<void *>(ptr));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void ldsm<layout::RowMajor, 1>(
    Array<unsigned, 1> & D,
    void const* ptr) {
  #if defined(CUDA_LDMATRIX_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x;
    asm volatile ("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];" : "=r"(x) : "r"(addr));
    reinterpret_cast<int &>(D) = x;

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void ldsm<layout::RowMajor, 2>(
    Array<unsigned, 2> & D,
    void const* ptr) {

  #if defined(CUDA_LDMATRIX_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x, y;
    asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(addr));
    reinterpret_cast<int2 &>(D) = make_int2(x, y);

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void ldsm<layout::RowMajor, 4>(
    Array<unsigned, 4> & D,
    void const* ptr) {

  #if defined(CUDA_LDMATRIX_ACTIVATED)
    #if SAIL_SIMULATE_CUTLASS_MMA

      int lane_idx = threadIdx.x % 32;
      unsigned mask = 0xffffffff;
      // 128b of cur row
      int4 temp_reg;
      half *reg_ptr = reinterpret_cast<half *>(&D);
      const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
      temp_reg = *smem_ptr;
      __syncthreads();
      for (int loop = 0; loop < 8; loop++) {
        // 8 threads per group to load one matrix(128B)
        int thread_idx = (loop / 2) * 8 + (lane_idx / 4);
        // each thread store 32bit elem, 4 thread in each row
        int reg_idx = (lane_idx % 4) * 2 + (loop % 2);

        // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
        // this will shuffle the element which reg_idx is calculated in the target thread
        // reg_ptr[loop] = __shfl_sync(mask, temp_reg_half[reg_idx], thread_idx);

        int4 shuffle_reg;
        // can't shuffle int4 directly
        int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
        int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
        for (int i = 0; i < 4; i++) {
          shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
        }
        half *shuffle_reg_half = reinterpret_cast<half *>(&shuffle_reg);
        reg_ptr[loop] = shuffle_reg_half[reg_idx];
      }

    #else

      unsigned addr = cutlass_get_smem_pointer(ptr);

      int x, y, z, w;
      asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
      reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
    #endif

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
template <typename Element>
struct ppu_ldsm<layout::RowMajor, 1, Element> {
  public:
    CUTLASS_DEVICE
    ppu_ldsm() {}

    CUTLASS_DEVICE
    void operator() (Array<unsigned, 1> &D, void const* ptr) {
      #if CUDA_LDMATRIX_ACTIVATED
        #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
          // PPU Hardware
          /* awmma::ldmatrix<awmma::no_trans, 1>(&D[0],
            reinterpret_cast<void *>(cutlass_get_smem_pointer(ptr))); */
          unsigned addr = cutlass_get_smem_pointer(ptr);

          int x;
          asm volatile ("ppu.ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(x) : "r"(addr));
          reinterpret_cast<int &>(D) = x;

        #else
          unsigned *smem_ptr = const_cast<unsigned *>(reinterpret_cast<const unsigned *>(ptr));
          unsigned *reg_ptr = reinterpret_cast<unsigned *>(&D);
          *reg_ptr = *smem_ptr;
        #endif
      #else
        assert(0);
      #endif
  }
};

template <typename Element>
struct ppu_ldsm<layout::RowMajor, 2, Element> {
  public:
    CUTLASS_DEVICE
    ppu_ldsm() {}

    CUTLASS_DEVICE
    void operator() (Array<unsigned, 2> &D, void const* ptr) {
      #if CUDA_LDMATRIX_ACTIVATED
      #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
        // PPU Hardware
        /* awmma::ldmatrix<awmma::no_trans, 2>(&D[0],
          reinterpret_cast<void *>(cutlass_get_smem_pointer(ptr))); */

        unsigned addr = cutlass_get_smem_pointer(ptr);

        int x, y;
        asm volatile ("ppu.ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(addr));
        reinterpret_cast<int2 &>(D) = make_int2(x, y);

      #else
        // PPU ldsm simulate
        int lane_idx = threadIdx.x % 32;
        unsigned mask = 0xffffffff;
        // 32b per thread.
        int element_per_access = 32 / sizeof_bits<Element>::value;
        // 128b of cur row
        int4 temp_reg;
        Element *reg_ptr = reinterpret_cast<Element *>(&D);
        const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
        temp_reg = *smem_ptr;
        __syncthreads();
        // thread per elements * matrix counts
        for (int loop = 0; loop < element_per_access * 4; loop++) {
          // 8 threads per group to load one matrix(128B)
          int thread_idx = (loop / element_per_access) * 8 + (lane_idx / 4);
          // each thread store 32b elems, 128b/32b 4 groups for one shared memory thread pointers.
          int reg_idx = (lane_idx % 4) * element_per_access + (loop % element_per_access);
          // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
          // this will shuffle the element which reg_idx is calculated in the target thread

          int4 shuffle_reg;
          // can't shuffle int4 directly
          int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
          int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
          // each shared memory thread pointers load one row 128b data, 32b per thread store group for mma.
          for (int i = 0; i < 4; i++) {
            shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
          }
          Element *shuffle_reg_element = reinterpret_cast<Element *>(&shuffle_reg);
          reg_ptr[loop] = shuffle_reg_element[reg_idx];
        }
      #endif
    #else
      assert(0);
    #endif
  }
};
template <typename Element>
struct ppu_ldsm<layout::RowMajor, 4, Element> {
  public:
    CUTLASS_DEVICE
    ppu_ldsm() {}

    CUTLASS_DEVICE
    void operator() (Array<unsigned, 4> &D, void const* ptr) {
      #if CUDA_LDMATRIX_ACTIVATED
      #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
        // PPU Hardware
        /* awmma::ldmatrix<awmma::no_trans, 4>(&D[0],
          reinterpret_cast<void *>(cutlass_get_smem_pointer(ptr))); */

        unsigned addr = cutlass_get_smem_pointer(ptr);

        int x, y, z, w;
        //asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
        //reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
        asm volatile ("ppu.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
        reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
      #else
        // PPU ldsm simulate
        int lane_idx = threadIdx.x % 32;
        unsigned mask = 0xffffffff;
        // 32b per thread.
        int element_per_access = 32 / sizeof_bits<Element>::value;
        // 128b of cur row
        int4 temp_reg;
        Element *reg_ptr = reinterpret_cast<Element *>(&D);
        const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
        temp_reg = *smem_ptr;
        __syncthreads();
        // thread per elements * matrix counts
        for (int loop = 0; loop < element_per_access * 4; loop++) {
          // 8 threads per group to load one matrix(128B)
          int thread_idx = (loop / element_per_access) * 8 + (lane_idx / 4);
          // each thread store 32b elems, 128b/32b 4 groups for one shared memory thread pointers.
          int reg_idx = (lane_idx % 4) * element_per_access + (loop % element_per_access);
          // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
          // this will shuffle the element which reg_idx is calculated in the target thread

          int4 shuffle_reg;
          // can't shuffle int4 directly
          int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
          int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
          // each shared memory thread pointers load one row 128b data, 32b per thread store group for mma.
          for (int i = 0; i < 4; i++) {
            shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
          }
          Element *shuffle_reg_element = reinterpret_cast<Element *>(&shuffle_reg);
          reg_ptr[loop] = shuffle_reg_element[reg_idx];
        }
      #endif
    #else
      assert(0);
    #endif
  }
};

template <typename Element>
struct ppu_ldsm<layout::ColumnMajor, 4, Element> {
  public:
    CUTLASS_DEVICE
    ppu_ldsm() {}

    CUTLASS_DEVICE
    void operator() (Array<unsigned, 4> &D, void const* ptr) {
    #if CUDA_LDMATRIX_ACTIVATED
      #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
        // PPU Hardware
        /* awmma::ldmatrix<awmma::trans_16x16b16, 4>(&D[0],
          reinterpret_cast<void *>(cutlass_get_smem_pointer(ptr))); */

        unsigned addr = cutlass_get_smem_pointer(ptr);

        int x, y, z, w;
        asm volatile ("ppu.ldmatrix.sync.aligned.m16n16.x1.trans.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
        reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
      #else
        // PPU ldsm simulate
        int lane_idx = threadIdx.x % 32;
        unsigned mask = 0xffffffff;
        // 128b of cur row
        int4 temp_reg;
        half *reg_ptr = reinterpret_cast<half *>(&D);
        const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
        temp_reg = *smem_ptr;
        __syncthreads();
        for (int loop = 0; loop < 8; loop++) {
          // in which 8x8 and which row
          int thread_idx = (loop / 2) * 8 + (lane_idx % 4) * 2 + (loop % 2);
          // T0/4/8/12/16/20/24/28 in one row
          int reg_idx = (lane_idx / 4);
          // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
          // this will shuffle the element which reg_idx is calculated in the target thread
          // reg_ptr[loop] = __shfl_sync(mask, temp_reg_half[reg_idx], thread_idx);

          int4 shuffle_reg;
          // can't shuffle int4 directly
          int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
          int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
          for (int i = 0; i < 4; i++) {
            shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
          }
          half *shuffle_reg_half = reinterpret_cast<half *>(&shuffle_reg);
          reg_ptr[loop] = shuffle_reg_half[reg_idx];
        }
      #endif
    #else
      assert(0);
    #endif
    }
};
#endif

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
// TSM_LD_NCOM_B32X4, similar to nvidia
template <>
inline __device__ void tsm_ld_ncom<layout::RowMajor, 4>(
    Array<unsigned, 4> & D,
    void const* ptr) {
  int lane_idx = threadIdx.x % 32;
  unsigned mask = 0xffffffff;
  // 128b of cur row
  int4 temp_reg;
  half *reg_ptr = reinterpret_cast<half *>(&D);
  const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
  temp_reg = *smem_ptr;
  __syncthreads();
  for (int loop = 0; loop < 8; loop++) {
    // in which 8x8 and which row
    int thread_idx = (loop / 2) * 8 + (lane_idx / 4);
    // each thread store 2 elem, 4 thread in each row
    int reg_idx = (lane_idx % 4) * 2 + (loop % 2);
    // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
    // this will shuffle the element which reg_idx is calculated in the target thread
    // reg_ptr[loop] = __shfl_sync(mask, temp_reg_half[reg_idx], thread_idx);

    int4 shuffle_reg;
    // can't shuffle int4 directly
    int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
    int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
    for (int i = 0; i < 4; i++) {
      shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
    }
    half *shuffle_reg_half = reinterpret_cast<half *>(&shuffle_reg);
    reg_ptr[loop] = shuffle_reg_half[reg_idx];
  }
}

// TSM_LD_NCOM_B32X4 with transpose
template <>
inline __device__ void tsm_ld_ncom<layout::ColumnMajor, 4>(
    Array<unsigned, 4> & D,
    void const* ptr) {
  int lane_idx = threadIdx.x % 32;
  unsigned mask = 0xffffffff;
  // 128b of cur row
  int4 temp_reg;
  half *reg_ptr = reinterpret_cast<half *>(&D);
  const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
  temp_reg = *smem_ptr;
  __syncthreads();
  for (int loop = 0; loop < 8; loop++) {
    // in which 8x8 and which row
    int thread_idx = (loop / 2) * 8 + (lane_idx % 4) * 2 + (loop % 2);
    // T0/4/8/12/16/20/24/28 in one row
    int reg_idx = (lane_idx / 4);
    // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
    // this will shuffle the element which reg_idx is calculated in the target thread
    // reg_ptr[loop] = __shfl_sync(mask, temp_reg_half[reg_idx], thread_idx);

    int4 shuffle_reg;
    // can't shuffle int4 directly
    int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
    int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
    for (int i = 0; i < 4; i++) {
      shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
    }
    half *shuffle_reg_half = reinterpret_cast<half *>(&shuffle_reg);
    reg_ptr[loop] = shuffle_reg_half[reg_idx];
  }
}

// VMEM_LD_B32X4
template <>
inline __device__ void vmem_ld<layout::RowMajor, 4>(
    Array<unsigned, 4> & D,
    void const* ptr) {
  int4 *reg_ptr = reinterpret_cast<int4 *>(&D);
  const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
  *reg_ptr = *smem_ptr;
}

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Transpose on 16b granularity
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void ldsm<layout::ColumnMajor, 1>(
    Array<unsigned, 1> & D,
    void const* ptr) {

  #if CUDA_LDMATRIX_ACTIVATED

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x;
    asm volatile ("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];" : "=r"(x) : "r"(addr));
    reinterpret_cast<int &>(D) = x;

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void ldsm<layout::ColumnMajor, 2>(
    Array<unsigned, 2> & D,
    void const* ptr) {

  #if defined(CUDA_LDMATRIX_ACTIVATED)

    unsigned addr = cutlass_get_smem_pointer(ptr);

    int x, y;
    asm volatile ("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(addr));
    reinterpret_cast<int2 &>(D) = make_int2(x, y);

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void ldsm<layout::ColumnMajor, 4>(
    Array<unsigned, 4> & D,
    void const* ptr) {

  #if defined(CUDA_LDMATRIX_ACTIVATED)
    // reg layout is
    // t0.0  t4.0  t8.0  t12.0...
    // t0.1  t4.1  t8.1  t12.1...
    // t1.0  t5.0  t9.0  t13.0...
    // t1.1  t5.1  t9.1  t13.1...
    // ......
    #if SAIL_SIMULATE_CUTLASS_MMA
      int lane_idx = threadIdx.x % 32;
      unsigned mask = 0xffffffff;
      // 128b of cur row
      int4 temp_reg;
      half *reg_ptr = reinterpret_cast<half *>(&D);
      const int4 *smem_ptr = reinterpret_cast<const int4 *>(ptr);
      temp_reg = *smem_ptr;
      __syncthreads();
      for (int loop = 0; loop < 8; loop++) {
        // in which 8x8 and which row
        int thread_idx = (loop / 2) * 8 + (lane_idx % 4) * 2 + (loop % 2);
        // each thread store 1 elem, 8 thread in each row
        int reg_idx = (lane_idx / 4);
        // must shuffle whole 128b reg, since shuffle can't use threadIdx as index, such as the annotated line below
        // this will shuffle the element which reg_idx is calculated in the target thread
        // reg_ptr[loop] = __shfl_sync(mask, temp_reg_half[reg_idx], thread_idx);

        int4 shuffle_reg;
        // can't shuffle int4 directly
        int *temp_reg_int = reinterpret_cast<int *>(&temp_reg);
        int *shuffle_reg_int = reinterpret_cast<int *>(&shuffle_reg);
        for (int i = 0; i < 4; i++) {
          shuffle_reg_int[i] = __shfl_sync(mask, temp_reg_int[i], thread_idx);
        }
        half *shuffle_reg_half = reinterpret_cast<half *>(&shuffle_reg);
        reg_ptr[loop] = shuffle_reg_half[reg_idx];
      }

    #else

      unsigned addr = cutlass_get_smem_pointer(ptr);

      int x, y, z, w;
      asm volatile ("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
      reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
    #endif

  #else

    CUTLASS_UNUSED(D);
    CUTLASS_UNUSED(ptr);
    CUTLASS_NOT_IMPLEMENTED();

  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType, int Bytes>
struct shared_load_op {
  CUTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    D = *reinterpret_cast<AccessType const *>(ptr);
  }
};

template <typename AccessType>
CUTLASS_DEVICE void shared_load(AccessType &D, void const *ptr) {
  shared_load_op<AccessType, int(sizeof(AccessType))>(D, ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct shared_load_op<AccessType, 16> {
  CUTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    unsigned addr = cutlass_get_smem_pointer(ptr);

    uint4 v;
    asm volatile ("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" :
      "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));

    D = reinterpret_cast<AccessType const &>(v);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct shared_load_op<AccessType, 8> {
  CUTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    unsigned addr = cutlass_get_smem_pointer(ptr);

    uint2 v;
    asm volatile ("ld.shared.v2.b32 {%0, %1}, [%2];" :
      "=r"(v.x), "=r"(v.y) : "r"(addr));

    D = reinterpret_cast<AccessType const &>(v);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass
