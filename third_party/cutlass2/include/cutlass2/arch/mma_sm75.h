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
    \brief Matrix multiply for SM75
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "cutlass2/arch/wmma.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
// CUDA Toolkit includes for nvcuda::wmma needed for binarized matrix multiply.
#include <mma.h>
#include "cutlass2/wmma_array.h"
#endif

// CUTLASS includes
#include "cutlass2/arch/mma.h"
#include "cutlass2/layout/matrix.h"
#include "cutlass2/numeric_types.h"

////////////////////////////////////////////////////////////////////////////////

#if ((__CUDACC_VER_MAJOR__ > 10) || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))

#define CUTLASS_ARCH_MMA_SM75_SUPPORTED 1

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
#define CUTLASS_ARCH_MMA_SM75_ENABLED
#endif
#endif

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
#define CUTLASS_ARCH_MMA_SM75_SUPPORTED 1
#define CUTLASS_ARCH_MMA_SM75_ENABLED
#endif


////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - FP16 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation - F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<16, 8, 8>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 2>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  unsigned const *C = reinterpret_cast<unsigned const *>(&c);
  unsigned *D = reinterpret_cast<unsigned *>(&d);

  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 8>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)
#if SAIL_SIMULATE_CUTLASS_MMA
  const uint16_t *frag_a = reinterpret_cast<const uint16_t *>(&a);
  const uint16_t *frag_b = reinterpret_cast<const uint16_t *>(&b);
  const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
  ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
  int lane_idx = threadIdx.x % 32;
  unsigned mask = 0xffffffff;
  // two elem in each 8x8
  for (int loop = 0; loop < 4; loop++) {
    ElementC acc = 0;
    for (int k = 0; k < Shape::kK; k++) {
      // threadIdx of elem a, will use two elem each time
      int thread_a = lane_idx - (lane_idx % 4) + (k / 2);
      // (loop / 2) * 2 corresponds to top part or down part, top part start from reg[0], down part start from reg[2]
      // k % 2 corresponds which elem in each part, inner loop will use two elem in each thread
      int reg_a = ((loop / 2) * 2) + (k % 2);
      uint16_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
      ElementA cur_a_half;
      memcpy(&cur_a_half, &cur_a, sizeof(uint16_t));
      // first two part is which col in matrix b, last part is which row
      int thread_b = (lane_idx % 4) * 8 + (loop % 2) * 4 + k / 2;
      int reg_b = k % 2;
      uint16_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
      ElementB cur_b_half;
      memcpy(&cur_b_half, &cur_b, sizeof(uint16_t));
      acc += ElementC(cur_a_half * cur_b_half);
    }
    frag_d[loop] = acc + frag_c[loop];
  }
#else
  unsigned const *A = reinterpret_cast<unsigned const *>(&a);
  unsigned const *B = reinterpret_cast<unsigned const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );
#endif
#else
    assert(0);
#endif
  }
};

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
// two ppu mma implementation, half->half, half->float
template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  CUTLASS_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {
    // only fake mma now
    // assert(0);
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
      uint32_t *D = reinterpret_cast<uint32_t *>(&d);

      asm volatile(
      "ppu.mma.sync.aligned.m16n16k16.row.col.f16.f16.f16.f16  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
      "{%12,%13,%14,%15};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
        "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
    #else
      assert(0);
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

      asm volatile(
      "ppu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
      "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));

    #else
      // TSM_LD_NCOM_B32X4, similar to nvidia
      const uint16_t *frag_a = reinterpret_cast<const uint16_t *>(&a);
      const uint16_t *frag_b = reinterpret_cast<const uint16_t *>(&b);
      const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
      ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
      int lane_idx = threadIdx.x % 32;
      unsigned mask = 0xffffffff;
      // two elem in each 8x8
      for (int loop = 0; loop < 8; loop++) {
        ElementC acc = 0;
        for (int k = 0; k < Shape::kK; k++) {
          // threadIdx of elem a, will use two elem each time, right 8 k's thread is identical to left 8 k
          int thread_a = lane_idx - (lane_idx % 4) + ((k % 8) / 2);
          // (loop / 4) * 4 corresponds to top or down part, ((k / 8) * 2) corresponds to left or right part
          // k % 2 corresponds to which elem in each part, inner loop will use two elem in each thread
          int reg_a = ((loop / 4) * 4) + ((k / 8) * 2) + (k % 2);
          uint16_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
          ElementA cur_a_half;
          memcpy(&cur_a_half, &cur_a, sizeof(uint16_t));
          // first two part is which col in matrix b, last part is which row
          // int thread_b = (lane_idx % 4) * 8 + (loop % 2) * 4 + (k % 8) / 2;
          int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + (k % 8) / 2;
          // first is hori part, second is vert part, third is index in cur part
          int reg_b = (((loop % 4) / 2) * 4) + (k / 8) * 2 + k % 2;
          uint16_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
          ElementB cur_b_half;
          memcpy(&cur_b_half, &cur_b, sizeof(uint16_t));
          acc += ElementC(cur_a_half * cur_b_half);
        }
        frag_d[loop] = acc + frag_c[loop];
      }

      // VMEM_LD_B32X4 version
      // const uint16_t *frag_a = reinterpret_cast<const uint16_t *>(&a);
      // const uint16_t *frag_b = reinterpret_cast<const uint16_t *>(&b);
      // const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
      // ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
      // int lane_idx = threadIdx.x % 32;
      // unsigned mask = 0xffffffff;
      // ElementC acc = 0;
      // // each thread get 8 outputs of half row
      // for (int loop = 0; loop < 8; loop++) {
      //   for (int k = 0; k < Shape::kK; k++) {
      //     int thread_a = lane_idx - (lane_idx % 2) + (k / 8);
      //     int reg_a = k % 8;
      //     // shfl_sync can't shuffle half
      //     uint16_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
      //     ElementA cur_a_half;
      //     memcpy(&cur_a_half, &cur_a, sizeof(uint16_t));
      //     // first two part is which col in matrix b, last part is which row
      //     int thread_b = (lane_idx % 2) * 16 + loop * 2 + k / 8;
      //     int reg_b = k % 8;
      //     uint16_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
      //     ElementB cur_b_half;
      //     memcpy(&cur_b_half, &cur_b, sizeof(uint16_t));
      //     acc += ElementC(cur_a_half * cur_b_half);
      //     // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 11) {
      //     //   printf("%d, %d, %d, %d, %d, %d, 111111111\n", loop, k, thread_a, reg_a, thread_b, reg_b);
      //     // }
      //   }
      //   frag_d[loop] = acc + frag_c[loop];
      // }
    #endif
    return;

  }
};
#endif

////////////////////////////////////////////////////////////////////////////////
//
// Integer matrix multiply .8816 (8b)
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.s8.u8 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));


#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<8, 8, 16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 8, 16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Integer matrix multiply  (8b) with SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.u8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Integer matrix multiply  (4b)
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  int4b_t,
  layout::RowMajor,
  int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int4b_t, 8>;

  using ElementB = int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  uint4b_t,
  layout::RowMajor,
  int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint4b_t, 8>;

  using ElementB = int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  int4b_t,
  layout::RowMajor,
  uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int4b_t, 8>;

  using ElementB = uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  uint4b_t,
  layout::RowMajor,
  uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint4b_t, 8>;

  using ElementB = uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Integer matrix multiply  (4b) - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  int4b_t,
  layout::RowMajor,
  int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int4b_t, 8>;

  using ElementB = int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  uint4b_t,
  layout::RowMajor,
  int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint4b_t, 8>;

  using ElementB = int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.s4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  int4b_t,
  layout::RowMajor,
  uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int4b_t, 8>;

  using ElementB = uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.u4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<8,8,32>,
  32,
  uint4b_t,
  layout::RowMajor,
  uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<8,8,32>;

  using ElementA = uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint4b_t, 8>;

  using ElementB = uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint4b_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

  unsigned const & A = reinterpret_cast<unsigned const &>(a);
  unsigned const & B = reinterpret_cast<unsigned const &>(b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile("mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A), "r"(B), "r"(C[0]), "r"(C[1]));

#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// b1 ^ b1 + s32 => s32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <>
struct Mma<
  gemm::GemmShape<8,8,128>,
  32,
  uint1b_t,
  layout::RowMajor,
  uint1b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpXorPopc> {

  using Shape = gemm::GemmShape<8,8,128>;

  using ElementA = uint1b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint1b_t, 32>;

  using ElementB = uint1b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint1b_t, 32>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 2>;

  using Operator = OpXorPopc;
  using ArchTag = arch::Sm75;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM75_ENABLED)

#if defined(CUTLASS_ARCH_WMMA_ENABLED) && !defined(__HGGC_ARCH__)
/* ppu donot support wmma::bmma_sync and uint1b_t datatype. */
  using WmmaFragmentA = nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_a,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          nvcuda::wmma::experimental::precision::b1,
          nvcuda::wmma::row_major>;

  using WmmaFragmentB = nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          nvcuda::wmma::experimental::precision::b1,
          nvcuda::wmma::col_major>;

  using WmmaFragmentC = nvcuda::wmma::fragment<
          nvcuda::wmma::accumulator,
          Shape::kM,
          Shape::kN,
          Shape::kK,
          int>;

  WmmaFragmentA const & A = reinterpret_cast<WmmaFragmentA const &>(a);
  WmmaFragmentB const & B = reinterpret_cast<WmmaFragmentB const &>(b);

  WmmaFragmentC const & C = reinterpret_cast<WmmaFragmentC const &>(c);
  WmmaFragmentC & D = reinterpret_cast<WmmaFragmentC &>(d);

  nvcuda::wmma::bmma_sync(D, A, B, C, nvcuda::wmma::experimental::bmmaBitOpXOR,
                                          nvcuda::wmma::experimental::bmmaAccumulateOpPOPC);
#else

  assert(0); // WMMA must be supported to issue binary matrix multiply-accumulate instructions.

#endif // defined(CUTLASS_ARCH_WMMA_ENABLED)

#else
    assert(0);
#endif

  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass
