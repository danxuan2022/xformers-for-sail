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
    \brief Matrix multiply
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "mma.h"
#include "cutlass2/layout/matrix.h"
#include "cutlass2/numeric_types.h"

////////////////////////////////////////////////////////////////////////////////

#if ((__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))

#define CUTLASS_ARCH_MMA_SM80_SUPPORTED 1

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && ! defined(__HGGCCC__))
#define CUTLASS_ARCH_MMA_SM80_ENABLED
#endif
#endif

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
#define CUTLASS_ARCH_MMA_SM80_SUPPORTED 1
#define CUTLASS_ARCH_MMA_SM80_ENABLED
#endif

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - Float BF16, FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation - F32 = bf16 * bf16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 8>,
  32,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 4>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1684 - Float TF32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = tf32 * tf32 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 4>,
  32,
  tfloat32_t,
  layout::RowMajor,
  tfloat32_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 4>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 2>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 1>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  float const *C = reinterpret_cast<float const *>(&c);
  float *D = reinterpret_cast<float *>(&d);

  asm volatile(
      "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
  );

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 1688 - Float TF32
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = tf32 * tf32 + F32
template <>
struct Mma<gemm::GemmShape<16, 8, 8>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 8, 8>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 4>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
template <>
struct Mma<gemm::GemmShape<16, 16, 8>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 8>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 4>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    #else
      const float *frag_a = reinterpret_cast<const float *>(&a);
      const float *frag_b = reinterpret_cast<const float *>(&b);
      const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
      ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
      int lane_idx = threadIdx.x % 32;
      unsigned mask = 0xffffffff;
      // two elem in each 8x8
      for (int loop = 0; loop < 8; loop++) {
        ElementC acc = 0;
        for (int k = 0; k < Shape::kK; k++) {
          // threadIdx of elem a, will use two elem each time, right 8 k's thread is identical to left 8 k
          int thread_a = lane_idx - (lane_idx % 4) + (k % 4);
          // (loop / 4) * 4 corresponds to top or down part, ((k / 8) * 2) corresponds to left or right part
          // k % 2 corresponds to which elem in each part, inner loop will use two elem in each thread
          int reg_a = ((loop / 4) * 2) + (k / 4);
          float cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
          ElementA cur_a_tf32;
          memcpy(&cur_a_tf32, &cur_a, sizeof(float));
          // first two part is which col in matrix b, last part is which row
          int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + (k % 4);
          // first is hori part, second is vert part
          int reg_b = ((loop % 4) / 2) * 2 + (k / 4);
          float cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
          ElementB cur_b_tf32;
          memcpy(&cur_b_tf32, &cur_b, sizeof(float));
          acc += ElementC(cur_a_tf32 * cur_b_tf32);
        }
        frag_d[loop] = acc + frag_c[loop];
      }
    #endif
  }
};

template <>
struct Mma<gemm::GemmShape<16, 16, 4>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 4>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 2>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m16n16k4.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    #endif
  }
};

template <>
struct Mma<gemm::GemmShape<8, 16, 8>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 8>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 2>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m8n16k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
    #endif
  }
};

template <>
struct Mma<gemm::GemmShape<8, 16, 4>, 32, tfloat32_t, layout::RowMajor,
           tfloat32_t, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 4>;

  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<tfloat32_t, 1>;

  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<tfloat32_t, 2>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m8n16k4.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4}, {%5,%6}, {%7,%8,%9,%10};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
    #endif
  }
};

// MMA 16168, F32 = F32 * F32 + F32
template <>
struct Mma<gemm::GemmShape<16, 16, 8>, 32, float, layout::RowMajor,
           float, layout::ColumnMajor, float, layout::RowMajor,
           OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 8>;
  using ElementA = float;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<float, 4>;
  using ElementB = float;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<float, 4>;
  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;
  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;
  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      float const *A = reinterpret_cast<float const *>(&a);
      float const *B = reinterpret_cast<float const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);
      asm volatile(
          "ppu.mma.sync.aligned.m16n16k8.row.col.f32.f32.f32.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23};\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
          : "f"(A[0]), "f"(A[1]), "f"(A[2]), "f"(A[3]),
            "f"(B[0]), "f"(B[1]), "f"(B[2]), "f"(B[3]),
            "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    #endif
  }
};


// MMA 16168, F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 8>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 8>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m16n16k8.row.col.f32.f16.f16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    #endif
  }
};

// MMA 81616, F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 16, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 16, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m8n16k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
    #endif
  }
};

// MMA 8168, F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<8, 16, 8>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8, 16, 8>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 2>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m8n16k8.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, {%4}, {%5,%6}, "
        "{%7,%8,%9,%10};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 8>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      float const *C = reinterpret_cast<float const *>(&c);
      float *D = reinterpret_cast<float *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32  {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    #endif
  }
};

#endif

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16816
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F16 = F16 * F16 + F16
template <>
struct Mma<
  gemm::GemmShape<16, 8, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<half_t, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
  uint32_t const *C = reinterpret_cast<uint32_t const *>(&c);
  uint32_t *D = reinterpret_cast<uint32_t *>(&d);

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(D[0]), "=r"(D[1])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1])
  );

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = bf16 * bf16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 16>,
  32,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 16>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 8>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F32 = F16 * F16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 16>,
  32,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    float const *C = reinterpret_cast<float const *>(&c);
    float *D = reinterpret_cast<float *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%10,%11,%12,%13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 884 - F64
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: F64 = F64 * F64 + F64
template <>
struct Mma<
  gemm::GemmShape<8,8,4>,
  32,
  double,
  layout::RowMajor,
  double,
  layout::ColumnMajor,
  double,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<8,8,4>;

  using ElementA = double;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<double, 1>;

  using ElementB = double;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<double, 1>;

  using ElementC = double;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<double, 2>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::Sm80;

  CUTLASS_HOST_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

  double const & A = reinterpret_cast<double const &>(a);
  double const & B = reinterpret_cast<double const &>(b);

  double const *C = reinterpret_cast<double const *>(&c);
  double *D = reinterpret_cast<double *>(&d);

  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=d"(D[0]), "=d"(D[1])
      : "d"(A), "d"(B), "d"(C[0]), "d"(C[1]));

#else

    CUTLASS_UNUSED(d);
    CUTLASS_UNUSED(a);
    CUTLASS_UNUSED(b);
    CUTLASS_UNUSED(c);
    CUTLASS_NOT_IMPLEMENTED();

#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16816 - S8 input, S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;

  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
        "{%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));


#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16816 - S8 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const &B = reinterpret_cast<uint32_t const &>(b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32.satfinite {%0,%1,%2,%3}, {%4,%5}, "
        "{%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(B), "r"(C[0]), "r"(C[1]), "r"(C[2]),
          "r"(C[3]));

#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16832 - S8 input, S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {
#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
#if SAIL_SIMULATE_CUTLASS_MMA
  const int8_t *frag_a = reinterpret_cast<const int8_t *>(&a);
  const int8_t *frag_b = reinterpret_cast<const int8_t *>(&b);
  const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
  ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
  int lane_idx = threadIdx.x % 32;
  unsigned mask = 0xffffffff;

  for (int loop = 0; loop < 4; loop++) {
    ElementC acc = 0;
    for (int k = 0; k < Shape::kK; k++) {
      int thread_a = lane_idx - (lane_idx % 4) + ((k - (k / 16) * 16) / 4);
      int reg_a = ((loop / 2) * 4) + (k / 16) * 8 + (k % 4);
      int8_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
      ElementA cur_a_int;
      memcpy(&cur_a_int, &cur_a, sizeof(int8_t));
      int thread_b = (lane_idx % 4) * 8 + (loop % 2) * 4 + ((k - (k / 16) * 16) / 4);
      int reg_b = (k / 16) * 4 + (k % 4);
      int8_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
      ElementB cur_b_int;
      memcpy(&cur_b_int, &cur_b, sizeof(int8_t));
      acc += ElementC(cur_a_int * cur_b_int);
    }
    frag_d[loop] = acc + frag_c[loop];
  }
#else
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#endif
#else
    assert(0);
#endif
  }
};

#if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
/// Simulate 161632 Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m16n16k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
    #else
    const int8_t *frag_a = reinterpret_cast<const int8_t *>(&a);
    const int8_t *frag_b = reinterpret_cast<const int8_t *>(&b);
    const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
    ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
    int lane_idx = threadIdx.x % 32;
    unsigned mask = 0xffffffff;

    for (int loop = 0; loop < 8; loop++) {
      ElementC acc = 0;
      for (int k = 0; k < Shape::kK; k++) {
        // lane idx A row base[0,4,8,12,16,20,24,28] + loop k for one k dim to find shfl thread who store continue
        int thread_a = lane_idx - (lane_idx % 4) + ((k % 16) / 4);
        // loop id to find base k dim(first 4 loop in same row) + k to find k dim across 16 elements.
        int reg_a = ((loop / 4) * 8) + (k / 16) * 4 + (k % 4);
        int8_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
        ElementA cur_a_int;
        memcpy(&cur_a_int, &cur_a, sizeof(int8_t));
        // lane_idx B matrix n dim base + loop for one thread reg output is acrosss 128 bits(elements) + k to loop one colom thread(group 4)
        int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + ((k % 16) / 4);
        // loop to calcate row(k dim) base + k dim reg.
        int reg_b = (((loop % 4) / 2) * 8) + (k / 16) * 4 + (k % 4);
        int8_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
        ElementB cur_b_int;
        memcpy(&cur_b_int, &cur_b, sizeof(int8_t));
        acc += ElementC(cur_a_int * cur_b_int);
      }
      frag_d[loop] = acc + frag_c[loop];
    }
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {
  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m16n16k32.row.col.satfinite.s32.s8.s8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
    #else
    const int8_t *frag_a = reinterpret_cast<const int8_t *>(&a);
    const int8_t *frag_b = reinterpret_cast<const int8_t *>(&b);
    const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
    ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
    int lane_idx = threadIdx.x % 32;
    unsigned mask = 0xffffffff;
    const ElementC c_max = (1u << (cutlass::sizeof_bits<ElementC>::value-1)) - 1;
    const ElementC c_min = -(1u << (cutlass::sizeof_bits<ElementC>::value-1));

    for (int loop = 0; loop < 8; loop++) {
      ElementC acc = 0;
      for (int k = 0; k < Shape::kK; k++) {
        // lane idx A row base[0,4,8,12,16,20,24,28] + loop k for one k dim to find shfl thread who store continue
        int thread_a = lane_idx - (lane_idx % 4) + ((k % 16) / 4);
        // loop id to find base k dim(first 4 loop in same row) + k to find k dim across 16 elements.
        int reg_a = ((loop / 4) * 8) + (k / 16) * 4 + (k % 4);
        int8_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
        ElementA cur_a_int;
        memcpy(&cur_a_int, &cur_a, sizeof(int8_t));
        // lane_idx B matrix n dim base + loop for one thread reg output is acrosss 128 bits(elements) + k to loop one colom thread(group 4)
        int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + ((k % 16) / 4);
        // loop to calcate row(k dim) base + k dim reg.
        int reg_b = (((loop % 4) / 2) * 8) + (k / 16) * 4 + (k % 4);
        int8_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
        ElementB cur_b_int;
        memcpy(&cur_b_int, &cur_b, sizeof(int8_t));
        int64_t temp_sum = int64_t(acc) + int64_t(ElementC(cur_a_int * cur_b_int));
        if (temp_sum > c_max) {
          temp_sum = c_max;
        } else if (temp_sum < c_min) {
          temp_sum = c_min;
        }
        acc = ElementC(temp_sum);
      }

      int64_t temp_sum = int64_t(acc) + int64_t(frag_c[loop]);
      if (temp_sum > c_max) {
        temp_sum = c_max;
      } else if (temp_sum < c_min) {
        temp_sum = c_min;
      }
      frag_d[loop] = ElementC(temp_sum);
    }
    #endif
  }
};

/// Simulate 161632 Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m16n16k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
    #else
    const uint8_t *frag_a = reinterpret_cast<const uint8_t *>(&a);
    const uint8_t *frag_b = reinterpret_cast<const uint8_t *>(&b);
    const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
    ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
    int lane_idx = threadIdx.x % 32;
    unsigned mask = 0xffffffff;

    for (int loop = 0; loop < 8; loop++) {
      ElementC acc = 0;
      for (int k = 0; k < Shape::kK; k++) {
        // lane idx A row base[0,4,8,12,16,20,24,28] + loop k for one k dim to find shfl thread who store continue
        int thread_a = lane_idx - (lane_idx % 4) + ((k % 16) / 4);
        // loop id to find base k dim(first 4 loop in same row) + k to find k dim across 16 elements.
        int reg_a = ((loop / 4) * 8) + (k / 16) * 4 + (k % 4);
        uint8_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
        ElementA cur_a_int;
        memcpy(&cur_a_int, &cur_a, sizeof(uint8_t));
        // lane_idx B matrix n dim base + loop for one thread reg output is acrosss 128 bits(elements) + k to loop one colom thread(group 4)
        int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + ((k % 16) / 4);
        // loop to calcate row(k dim) base + k dim regs.
        int reg_b = (((loop % 4) / 2) * 8) + (k / 16) * 4 + (k % 4);
        uint8_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
        ElementB cur_b_int;
        memcpy(&cur_b_int, &cur_b, sizeof(uint8_t));
        acc += ElementC(cur_a_int * cur_b_int);
      }
      frag_d[loop] = acc + frag_c[loop];
    }
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {
  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {

    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__)) && !SAIL_SIMULATE_CUTLASS_MMA
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m16n16k32.row.col.satfinite.s32.u8.u8.s32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
    #else
    const uint8_t *frag_a = reinterpret_cast<const uint8_t *>(&a);
    const uint8_t *frag_b = reinterpret_cast<const uint8_t *>(&b);
    const ElementC *frag_c = reinterpret_cast<const ElementC *>(&c);
    ElementC *frag_d = reinterpret_cast<ElementC *>(&d);
    int lane_idx = threadIdx.x % 32;
    unsigned mask = 0xffffffff;
    const ElementC c_max = (1u << (cutlass::sizeof_bits<ElementC>::value-1)) - 1;
    const ElementC c_min = -(1u << (cutlass::sizeof_bits<ElementC>::value-1));

    for (int loop = 0; loop < 8; loop++) {
      ElementC acc = 0;
      for (int k = 0; k < Shape::kK; k++) {
        // lane idx A row base[0,4,8,12,16,20,24,28] + loop k for one k dim to find shfl thread who store continue
        int thread_a = lane_idx - (lane_idx % 4) + ((k % 16) / 4);
        // loop id to find base k dim(first 4 loop in same row) + k to find k dim across 16 elements.
        int reg_a = ((loop / 4) * 8) + (k / 16) * 4 + (k % 4);
        uint8_t cur_a = __shfl_sync(mask, frag_a[reg_a], thread_a);
        ElementA cur_a_int;
        memcpy(&cur_a_int, &cur_a, sizeof(uint8_t));
        // lane_idx B matrix n dim base + loop for one thread reg output is acrosss 128 bits(elements) + k to loop one colom thread(group 4)
        int thread_b = (lane_idx % 4) * 4 + (loop % 2) * 16 + ((k % 16) / 4);
        // loop to calcate row(k dim) base + k dim regs.
        int reg_b = (((loop % 4) / 2) * 8) + (k / 16) * 4 + (k % 4);
        uint8_t cur_b = __shfl_sync(mask, frag_b[reg_b], thread_b);
        ElementB cur_b_int;
        memcpy(&cur_b_int, &cur_b, sizeof(uint8_t));
        int64_t temp_sum = int64_t(acc) + int64_t(ElementC(cur_a_int * cur_b_int));
        if (temp_sum > c_max) {
          temp_sum = c_max;
        } else if (temp_sum < c_min) {
          temp_sum = c_min;
        }
        acc = ElementC(temp_sum);
      }
      int64_t temp_sum = int64_t(acc) + int64_t(frag_c[loop]);
      if (temp_sum > c_max) {
        temp_sum = c_max;
      } else if (temp_sum < c_min) {
        temp_sum = c_min;
      }
      frag_d[loop] = ElementC(temp_sum);
    }
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m16n16k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 8>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m16n16k16.row.col.s32.u8.u8.s32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3]), "=r"(D[4]), "=r"(D[5]), "=r"(D[6]), "=r"(D[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]), "r"(C[4]), "r"(C[5]), "r"(C[6]), "r"(C[7]));
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<8, 16, 32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m8n16k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<8, 16, 32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m8n16k32.row.col.s32.u8.u8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7,%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]), "r"(B[2]), "r"(B[3]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<8, 16, 16>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 16>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m8n16k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4}, {%5,%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
    #endif
  }
};

template <>
struct Mma<
  gemm::GemmShape<8, 16, 16>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {
  using Shape = gemm::GemmShape<8, 16, 16>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  CUTLASS_DEVICE
  void operator()(FragmentC &d, FragmentA const &a, FragmentB const &b,
                  FragmentC const &c) const {
    #if (defined(__HGGCCC__) || defined(__HGGCCC_RTC__))
      uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
      int const *C = reinterpret_cast<int const *>(&c);
      int *D = reinterpret_cast<int *>(&d);

      asm volatile(
        "ppu.mma.sync.aligned.m8n16k16.row.col.s32.u8.u8.s32 "
        "{%0,%1,%2,%3}, {%4}, {%5,%6}, {%7,%8,%9,%10};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
    #endif
  }
};
#endif

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16832 - S8 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

  uint32_t const * A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const * B = reinterpret_cast<uint32_t const *>(&b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * S8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  int8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U8 * U8 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,32>,
  32,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16,8,32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16864 - S4 input, S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};


////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16864 - S4 input, S32 accumulation - SATURATE
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

  uint32_t const * A = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const * B = reinterpret_cast<uint32_t const *>(&b);

  int const *C = reinterpret_cast<int const *>(&c);
  int *D = reinterpret_cast<int *>(&d);

  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
        "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * S4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::int4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::int4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::int4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = S4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::int4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::int4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::int4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = U4 * U4 + S32
template <>
struct Mma<
  gemm::GemmShape<16, 8, 64>,
  32,
  cutlass::uint4b_t,
  layout::RowMajor,
  cutlass::uint4b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAddSaturate> {

  using Shape = gemm::GemmShape<16, 8, 64>;

  using ElementA = cutlass::uint4b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint4b_t, 32>;

  using ElementB = cutlass::uint4b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint4b_t, 16>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAddSaturate;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32.satfinite {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

/// Matrix multiply-add operation: S32 = B1 & B1 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,256>,
  32,
  cutlass::uint1b_t,
  layout::RowMajor,
  cutlass::uint1b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,8,256>;

  using ElementA = cutlass::uint1b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint1b_t, 128>;

  using ElementB = cutlass::uint1b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint1b_t, 64>;

  using ElementC = int32_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int32_t, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 168256 - B1 input, S32 accumulation - XOR,POPC
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = B1 & B1 + S32
template <>
struct Mma<
  gemm::GemmShape<16,8,256>,
  32,
  cutlass::uint1b_t,
  layout::RowMajor,
  cutlass::uint1b_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpXorPopc> {

  using Shape = gemm::GemmShape<16,8,256>;

  using ElementA = cutlass::uint1b_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<cutlass::uint1b_t, 128>;

  using ElementB = cutlass::uint1b_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<cutlass::uint1b_t, 64>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<int, 4>;

  using Operator = OpXorPopc;
  using ArchTag = arch::Sm80;

  /// Computes multiply-add
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {

#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)

    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);

    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

