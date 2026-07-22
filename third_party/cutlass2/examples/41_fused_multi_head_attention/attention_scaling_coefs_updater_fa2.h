/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass2/functional.h"
#include "cutlass2/gemm/warp/mma_simt_tile_iterator.h"
#include "cutlass2/gemm/warp/mma_tensor_op_tile_iterator_sm70.h"
#include "cutlass2/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"
#include "cutlass2/matrix_shape.h"
#include "gemm_kernel_utils.h"

namespace {

static CUTLASS_DEVICE float atomicMaxFloat(float* addr, float value) {
  // source: https://stackoverflow.com/a/51549250
  return (value >= 0)
      ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
      : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}
} // namespace

/* Iterates on the accumulator and corresponding position on result matrix

(1) Update `mi[r]` to the max value of the row `r`
(2) In a second iteration do the following:
    (a) accum   <- exp(accum - mi)
    (b) m_prime <- exp(m_prime - mi)
    (c) s_prime <- s_prime * m_prime + sum(accum)

All of this is done on registers, before we store all of this
on shared memory for the next matmul with Value.

We have multiple implementations, because each configuration has a different way
of iterating in the accumulators.
*/


template <typename BASE, typename T, typename BASE2, typename T2, typename accum_t, int kWarpSize>
struct RegisterOpsFA2 {
  template <
      int kQueriesPerBlock,
      bool kFullColumns,
      bool kIsFirst,
      bool kKeepOutputInRF,
      int WarpNCount,
      bool kSupportsBias,
      int kNumWarpsPerBlock>
  CUTLASS_DEVICE static void update(
      typename T2::Fragment& frag_o, // output so far
      typename T::Fragment& frag,
      cutlass::Array<accum_t, kQueriesPerBlock>& mi,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      cutlass::Array<accum_t, kQueriesPerBlock * WarpNCount>& s_prime,
      int8_t lane_id,
      int16_t thread_id,
      int8_t warp_id,
      int8_t warp_idx_n,
      int max_col,
      typename T::TensorCoord const& tile_offset,
      float scaling) {
    // Convert to `accum_t` (rather than double)
    constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
    if (!kIsFirst) {
      if (thread_id < kQueriesPerBlock) {
        m_prime[thread_id] = mi[thread_id];
      }
      __syncthreads();
    }

    // handle warp partition in kv side.
    static_assert(kQueriesPerBlock % kNumWarpsPerBlock == 0, "");
    static constexpr int kLinesPerWarp = kQueriesPerBlock / kNumWarpsPerBlock;

    frag = cutlass::multiplies<typename T::Fragment>()(scaling * kLog2e, frag);

    auto lane_offset = BASE::get_lane_offset(lane_id, warp_id, tile_offset);
    auto lane_offset_2 = BASE2::get_lane_offset(lane_id, warp_id, tile_offset);

    // First update `mi` to the max per-row
    {
      accum_t max;
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) {
            max = -cutlass::platform::numeric_limits<accum_t>::infinity();
          },
          [&](int accum_m, int accum_n, int idx) {
            if (kFullColumns || accum_n < max_col) {
              max = cutlass::fast_max(max, frag[idx]);
            }
          },
          [&](int accum_m) {
            // Having 4x atomicMax seems faster than reduce within warp
            // first...
            atomicMaxFloat(&mi[accum_m], max);
          });
    }

    // Make sure we all share the update values for `mi`
    __syncthreads();

    // Doing this `exp` is quite expensive. Let's
    // split it across the warps
    bool restore_mi_to_minus_inf = false;
    if (lane_id < kLinesPerWarp) {
      int id = warp_id * kLinesPerWarp + lane_id;
      auto m_prime_id = m_prime[id];
      auto mi_id = mi[id];
      bool changed = m_prime_id < mi_id; // `false` if both are -inf
      if (changed) {
        auto m_prime_exp = exp2f(m_prime_id - mi_id);
        m_prime[id] = m_prime_exp;
        s_prime[id] *= m_prime_exp;
      } else {
        // Only when bias is enabled, it's possible that all the first values
        // of attention are masked to `-inf`. In that case we want to avoid
        // `nan = exp2f(-inf - (-inf))` so we temporarily set `mi` to 0
        if (kSupportsBias &&
          mi_id == -cutlass::platform::numeric_limits<accum_t>::infinity()) {
          restore_mi_to_minus_inf = true;
          mi[id] = 0.0f;
        }
        // out_rescale[id] = 1.0f;
        m_prime[id] = 1.0f;
      }

    }
    __syncthreads(); // Update output fragments

    if (kKeepOutputInRF && !kIsFirst) {
      accum_t mp;
      BASE2::iterateRows(
          lane_offset_2,
          [&](int accum_m) { mp = m_prime[accum_m]; },
          [&](int accum_m, int accum_n, int idx) { frag_o[idx] *= mp; },
          [&](int accum_m) {});
      __syncthreads();
    }
    // Update accum_m, accum_n, ...
    {
      accum_t mi_row, total_row;
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { mi_row = mi[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag[idx] = (kFullColumns || accum_n < max_col)
                ? exp2f(frag[idx] - mi_row)
                : accum_t(0.0);
          },
          [&](int accum_m) {});
      BASE::iterateRows(
          lane_offset,
          [&](int accum_m) { total_row = 0.0; },
          [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
          [&](int accum_m) {
            if (BASE::reduceSameRow(
                    lane_id, total_row, [](accum_t a, accum_t b) {
                      return a + b;
                    })) {
              // save result to shared
              s_prime[accum_m + warp_idx_n * kQueriesPerBlock] += total_row;
            }
          });

      __syncthreads();
      // read partial sum and reduce in warp0
      accum_t row_sum(0.0);
      if (lane_id < kLinesPerWarp) {
        int id = warp_id * kLinesPerWarp + lane_id;
        // accum_t total_row = s_prime[id];

        if (restore_mi_to_minus_inf) {
          // Restore `mi`, see above when we set `restore_mi_to_minus_inf=true`
          mi[id] = -cutlass::platform::numeric_limits<accum_t>::infinity();
        } else {
          m_prime[id] = mi[id];
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < WarpNCount; i++) {
          row_sum += s_prime[id + i * kQueriesPerBlock];
        }
        s_prime[id] = row_sum;

        //reset other warp's shared data to 0
        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < WarpNCount; i++) {
          s_prime[id + i * kQueriesPerBlock] = accum_t(0);
        }
      }

    }
  }
};

template <typename T, typename T2, typename accum_t, int kWarpSize>
struct AttentionScalingCoefsUpdaterSm80FA2
    : RegisterOpsFA2<
          AttentionScalingCoefsUpdaterSm80FA2<T, T2, accum_t, kWarpSize>,
          T,
          AttentionScalingCoefsUpdaterSm80FA2<T2, T, accum_t, kWarpSize>,
          T2,
          accum_t,
          kWarpSize> {
  static_assert(
      cutlass::platform::
          is_same<typename T::Layout, cutlass::layout::RowMajor>::value,
      "only RowMajor is supported");

  using Policy = typename T::Policy;
  using InstructionShape = typename T::InstructionShape;
  using OpDelta = typename T::OpDelta;
  using Shape = typename T::Shape;
  static int const kElementsPerAccess = InstructionShape::kN / 4;
  static int const kRowsPerTile = 8;
  static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

  static cutlass::MatrixCoord CUTLASS_DEVICE get_lane_offset(
      int8_t lane_id,
      int8_t warp_id,
      typename T::TensorCoord const& tile_offset) {
    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);
#if SAIL_PPU_MMA
    return cutlass::MatrixCoord(
        quad + tile_offset.row() * Shape::kRow,
        lane_in_quad +
            tile_offset.column() * Shape::kColumn);
#else
    return cutlass::MatrixCoord(
        quad + tile_offset.row() * Shape::kRow,
        lane_in_quad * kElementsPerAccess +
            tile_offset.column() * Shape::kColumn);
#endif
  }

  template <typename FA, typename FB, typename FC>
  CUTLASS_DEVICE static void iterateRows(
      cutlass::MatrixCoord& lane_offset,
      FA beginRow,
      FB op,
      FC endRow) {
    // See cutlass/gemm/warp/mma_tensor_op_tile_iterator.h
    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int row = 0; row < kAccumulatorRows; ++row) {
        int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
            row * kRowsPerTile + lane_offset.row();
        beginRow(accum_m);

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
          int mma_accum_start = kAccumulatorRows * kElementsPerAccess *
              (mma_n * Policy::MmaIterations::kRow + mma_m);
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
#if SAIL_PPU_MMA
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn +
                col * 4 + lane_offset.column();
#else
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn +
                col + lane_offset.column();
#endif
            int idx = mma_accum_start + row * kElementsPerAccess + col;
            op(accum_m, accum_n, idx);
          }
        }

        endRow(accum_m);
      }
    }
  }

  template <typename DT, typename F>
  CUTLASS_DEVICE static bool reduceSameRow(int lane_id, DT& myValue, F fn) {
    // In each warp, 4 threads will work on the same row
    // - the ones with the same `quad`
    auto otherV = __shfl_xor_sync(0xffffffff, myValue, 1);
    myValue = fn(myValue, otherV);
    otherV = __shfl_xor_sync(0xffffffff, myValue, 2);
    myValue = fn(myValue, otherV);
    int lane_in_quad = (lane_id & 3);
    return lane_in_quad == 0;
  }
};

template <typename T, typename T2, typename accum_t, int kWarpSize>
struct DefaultAttentionScalingCoefsUpdaterFA2 {
    using Updater =
      AttentionScalingCoefsUpdaterSm80FA2<T, T2, accum_t, kWarpSize>;
};

