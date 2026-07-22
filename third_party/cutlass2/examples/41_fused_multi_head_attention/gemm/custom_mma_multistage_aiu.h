/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass2/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass2/aligned_buffer.h"
#include "cutlass2/utils.h"

#include "cutlass2/gemm/warp/default_mma_tensor_op.h"
#include "cutlass2/gemm/threadblock/default_mma.h"

#include "cutlass2/transform/pitch_linear_thread_map.h"
#include "aiu/gemm/tool/cutlass_type_convert.h"
#include "cutlass2/tfloat32.h"

#include "custom_mma_base_aiu.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace aiu {
namespace gemm {
namespace threadblock {

template <
    // Iterator type for A matrix operand
    typename IteratorA_,
    // Iterator type for B matrix operand
    typename IteratorB_,
    // Iterator type for C matrix operand
    typename PrefetchIterator_,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator_,
    /// Wmma Fragment type for A and B (for FP32 tensorcore on PPU)
    typename WmmaFragABType>
struct CustomMmaMultistageAiu {
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using PrefetchIterator = PrefetchIterator_;

  using Operator = Operator_;
  using Shape = ThreadblockShape_;

  using WarpCount = cutlass::gemm::GemmShape<Shape::kM / WarpShape_::kM,
                                             Shape::kN / WarpShape_::kN,
                                             Shape::kK / WarpShape_::kK>;

  /// Fragment of accumulator tile
  using FragmentC = typename Operator_::FragmentC;

  using AccFragType = typename FromCutlassType<ElementAccumulator_>::type;

  using ElementA_ = typename FromCutlassType<typename IteratorA::Element>::type;

  using ElementB_ = typename FromCutlassType<typename IteratorB::Element>::type;

  using Base = typename cutlass::gemm::threadblock::CustomMmaBaseAiu<ThreadblockShape_,
                                                                 Operator_, Stages, WarpShape_>;
                                            
  using SharedStorageA = typename Base::SharedStorageA;

  using SharedStorageB = typename Base::SharedStorageB;

  ElementA_* tsm_astart_;

  ElementB_* tsm_bstart_;
  
  static int const kStages = Stages;

  /// Compatibility: whether entire K fits in SMEM (conservative: always false for AIU)
  static bool const kSmemContainsEntireMat = false;

  /// Compatibility: no-op for AIU path (prologue is handled externally)
  CUTLASS_DEVICE
  bool set_prologue_done(bool value) {
    return true;
  }

  static constexpr int CHUNK_LENGTH_K = ThreadblockShape_::kK;

  static constexpr int A_TSM_OFFSET = (ThreadblockShape_::kM * CHUNK_LENGTH_K);

  static constexpr int B_TSM_OFFSET = (ThreadblockShape_::kN * CHUNK_LENGTH_K);

  CUTLASS_DEVICE
  CustomMmaMultistageAiu(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorageA& shared_storageA,
      SharedStorageB& shared_storageB,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ) {
      tsm_astart_ = reinterpret_cast<ElementA_*>(shared_storageA.buffer.data());
      tsm_bstart_ = reinterpret_cast<ElementB_*>(shared_storageB.buffer.data());
      
  }

  CUTLASS_DEVICE
  bool set_zero_outside_bounds(int value) {
    return true;
  }

  template <bool kLoadA = true, bool kLoadB = true>
  CUTLASS_DEVICE static void prologue(
      SharedStorageA& shared_storageA,
      SharedStorageB& shared_storageB,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      int thread_idx,
      int problem_size_k) {
    
    Base::TensorRefA ref = shared_storageA.ref();

    ElementA_* tsm_astart_ = reinterpret_cast<ElementA_*>(shared_storageA.buffer.data());
    ElementB_* tsm_bstart_ = reinterpret_cast<ElementB_*>(shared_storageB.buffer.data());

    // prefetch for the Stages iteration
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Stages; ++stage) {
      if (kLoadA) {
        ElementA_* tsm_astart_staged = tsm_astart_ + stage * A_TSM_OFFSET;
        iterator_A.LoadToTsm(tsm_astart_staged);
      }

      if (kLoadB) {
        ElementB_* tsm_bstart_staged = tsm_bstart_ + stage * B_TSM_OFFSET;
        iterator_B.LoadToTsm(tsm_bstart_staged);
      }

      __pipeline_commit();
    }
  }

  /// 5-arg overload for backward kernel compatibility (delegates with entir_load=false)
  CUTLASS_DEVICE
  void operator()(
      int gemm_k_iterations,
      FragmentC &accum,
      IteratorA& iterator_A,
      IteratorB& iterator_B,
      FragmentC const &src_accum) {
    operator()(gemm_k_iterations, accum, iterator_A, iterator_B, src_accum, false);
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA& iterator_A,
      ///< iterator over B operand in global memory
      IteratorB& iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum,
      bool entir_load) {
    constexpr int WARP_X_ITER = (WarpShape_::kM / InstructionShape_::kM);
    constexpr int WARP_Y_ITER = (WarpShape_::kN / InstructionShape_::kN);
    constexpr int CHUNK_K = WarpShape_::kK / InstructionShape_::kK;

    using accType = awmma::fragment<awmma::accumulator, InstructionShape_::kM, InstructionShape_::kN,
                    InstructionShape_::kK, AccFragType>;
    auto & fragAcc = *reinterpret_cast<accType (*)[WARP_Y_ITER][WARP_X_ITER]>(accum.data());

    // Wait for any external prologue pipeline commits to complete
    __pipeline_wait_prior(0);
    #if SAIL_PPU_MMA
      __ppu_barrier_sync_nocnt(0, 1);
    #else
      __syncthreads();
    #endif

    // compute fragAcc = fragA * fragB
    {
      using WmmaLayoutA = typename cutlass::platform::conditional<
          cutlass::platform::is_same<typename IteratorA::Layout, cutlass::layout::RowMajor>::value,
          awmma::row_major, awmma::col_major>::type;
      using WmmaLayoutB = typename cutlass::platform::conditional<
          cutlass::platform::is_same<typename IteratorB::Layout, cutlass::layout::RowMajor>::value,
          awmma::row_major, awmma::col_major>::type;

      using FragAType =
          awmma::fragment<awmma::matrix_a, InstructionShape_::kM, InstructionShape_::kN,
                          InstructionShape_::kK, WmmaFragABType, WmmaLayoutA>;
      using FragBType =
          awmma::fragment<awmma::matrix_b, InstructionShape_::kM, InstructionShape_::kN,
                          InstructionShape_::kK, WmmaFragABType, WmmaLayoutB>;
      FragAType fragA[WARP_X_ITER][2];
      FragBType fragB[WARP_Y_ITER][2];

      const int gemm_k_size = gemm_k_iterations * ThreadblockShape_::kK;

      using CutlassElementABType = typename ToCutlassTypeAIU<WmmaFragABType>::type;
      using ArchOperator = typename cutlass::arch::Mma<
        cutlass::gemm::GemmShape<InstructionShape_::kM, InstructionShape_::kN, InstructionShape_::kK>,
        32,
        CutlassElementABType, cutlass::layout::RowMajor,
        CutlassElementABType, cutlass::layout::ColumnMajor,
        ElementAccumulator_, cutlass::layout::RowMajor,
        cutlass::arch::OpMultiplyAdd>;
      using CutlassFragA = typename ArchOperator::FragmentA;
      using CutlassFragB = typename ArchOperator::FragmentB;
      using CutlassFragC = typename ArchOperator::FragmentC;

      // Internal prologue: load first chunks from global to smem
      // (overrides external prologue data, advances iterators for subsequent loads)
      iterator_A.LoadToTsm(tsm_astart_);
      iterator_B.LoadToTsm(tsm_bstart_);
      __pipeline_commit();

      if (gemm_k_size > CHUNK_LENGTH_K) {
        iterator_A.LoadToTsm(tsm_astart_ + A_TSM_OFFSET);
        iterator_B.LoadToTsm(tsm_bstart_ + B_TSM_OFFSET);
        __pipeline_commit();
        __pipeline_wait_prior(1); // chunk 0 ready, chunk 1 may still be loading
      } else {
        __pipeline_wait_prior(0);
      }

      #if SAIL_PPU_MMA
        __ppu_barrier_sync_nocnt(0, 1);
      #else
        __syncthreads();
      #endif

      // Pre-load first K-slice fragments from smem stage 0
      iterator_A.LoadToVreg(tsm_astart_, fragA, 0);
      iterator_B.LoadToVreg(tsm_bstart_, fragB, 0);

      // Full kChunk loop with pipeline streaming (aligned with Stage=2 AIU MMA)
      for (int kChunk = 0; kChunk < gemm_k_size; kChunk += CHUNK_LENGTH_K) {

        #pragma unroll
        for (int kIter = 0; kIter < CHUNK_K; kIter++) {

          // Prefetch NEXT kIter's fragments from smem while computing current
          {
            int kIterNext = (kIter + 1) % CHUNK_K;
            int nextGlobalK = kChunk + (kIter + 1) * (int)InstructionShape_::kK;
            int nextStage = (nextGlobalK / CHUNK_LENGTH_K) % Stages;
            ElementA_* pSrcA = tsm_astart_ + nextStage * A_TSM_OFFSET;
            ElementB_* pSrcB = tsm_bstart_ + nextStage * B_TSM_OFFSET;
            iterator_A.LoadToVreg(pSrcA, fragA, kIterNext);
            iterator_B.LoadToVreg(pSrcB, fragB, kIterNext);
          }

          // convert from fp32 to tf32
          if (cutlass::platform::is_same<WmmaFragABType, awmma::precision::tf32>::value) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < WARP_X_ITER; i++) {
              frag_float_to_tf32(fragA[i][kIter & 0x1]);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < WARP_Y_ITER; i++) {
              frag_float_to_tf32(fragB[i][kIter & 0x1]);
            }
          }

          ArchOperator mmaOperator;
          #pragma unroll
          for (uint yIter = 0; yIter < WARP_Y_ITER; yIter++) {
            #pragma unroll
            for (uint xIter = 0; xIter < WARP_X_ITER; xIter++) {
              CutlassFragA a0 = HGGC_MMA_FRAGMENT_VALUE(fragA[xIter][kIter & 0x1], CutlassFragA);
              CutlassFragB b0 = HGGC_MMA_FRAGMENT_VALUE(fragB[yIter][kIter & 0x1], CutlassFragB);
              CutlassFragC c0 = HGGC_MMA_FRAGMENT_VALUE(fragAcc[yIter][xIter], CutlassFragC);
              mmaOperator(c0, a0, b0, c0);
              HGGC_MMA_FRAGMENT_VALUE(fragAcc[yIter][xIter], CutlassFragC) = c0;
            }
          }

          // Pipeline management: at kIter==CHUNK_K-2, sync and load next chunk from global
          if (kIter == CHUNK_K - 2 && kChunk < gemm_k_size - CHUNK_LENGTH_K) {
            __pipeline_wait_prior(0);
            #if SAIL_PPU_MMA
              __ppu_barrier_sync_nocnt(0, 1);
            #else
              __syncthreads();
            #endif
            if (kChunk < gemm_k_size - 2 * CHUNK_LENGTH_K) {
              // Load next-next chunk into current stage (about to finish using)
              int cur_stage = (kChunk / CHUNK_LENGTH_K) % Stages;
              iterator_A.LoadToTsm(tsm_astart_ + cur_stage * A_TSM_OFFSET);
              iterator_B.LoadToTsm(tsm_bstart_ + cur_stage * B_TSM_OFFSET);
              __pipeline_commit();
            }
          }
        } // end of kIter loop
      } // end of kChunk loop
    } // end of fragAcc = fragA * fragB

    __pipeline_wait_prior(0);
    #if SAIL_PPU_MMA
      __ppu_barrier_sync_nocnt(0, 1);
    #else
      __syncthreads();
    #endif
  }

};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
