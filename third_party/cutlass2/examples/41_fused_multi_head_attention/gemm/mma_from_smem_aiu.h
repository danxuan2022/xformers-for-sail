/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
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
    \brief Tools and utils to store a GEMM output in shmem, and to use that
   output as operandA for another GEMM back-to-back
*/

#pragma once

#include "aiu/gemm/tool/cutlass_type_convert.h"
#include "cutlass2/tfloat32.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

#define CONVERT_MMA_FRAGMENT_VALUE(cutlassfragType, wmmafrag)  (*(reinterpret_cast<wmmafrag *>(&(cutlassfragType))))
// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.

template <
    /// Size of the block problem - concept: gemm::GemmShape<>
    typename Shape_,
    // /// Size of the warp problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    // Maximum K dimension - also the dimension of the shared-memory
    // holding `OperandA`
    int kMaxK_,
    /// Policy describing tuning details (concept: MmaPolicyAiu)
    typename Operator_,
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Layout in shared-memory of operand A
    typename SmemLayoutA,
    /// B in TSM need transpose.
    bool kTransposeB = false>
class MmaBaseFromSharedMemoryAiu {
 public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  using WarpGemm = WarpShape_;

  static constexpr int kMaxK = kMaxK_;

  /// Warp-level Mma
  using Operator = Operator_;

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<
      Shape::kM / WarpGemm::kM,
      Shape::kN / WarpGemm::kN,
      Shape::kK / WarpGemm::kK>;
  using WarpCount1 = WarpCount;

  /// Number of warp-level GEMM oeprations
  static int const kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::MmaShape::kK);
  static int const kWarpGemmIterations1 = kWarpGemmIterations;

  /// Number of stages
  static int const kStages = Stages;

  /// If this is true, we fill the entire shmem buffer at start
  /// and don't need to iterate through it in a circular fashion
  static bool const kSmemContainsEntireB = kMaxK <= Shape::kK * kStages;

  /// Tensor reference to the A operand
  using TensorRefA = TensorRef<typename Policy_::Operator::ElementA, SmemLayoutA>;

  using LayoutB = typename Operator::LayoutB;

  /// Tensor reference to B operand.
  using TensorRefB =
      TensorRef<typename Operator::ElementB, LayoutB>;
  //
  // Nested structs
  //

  /// Shared storage object needed by threadblock-scoped GEMM
  class SharedStorage {
   public:
    //
    // Type definitions
    //

    /// Shape of the B matrix operand in shared memory
    using ShapeB = MatrixShape<
        Shape::kK * kStages, Shape::kN>;

   public:
    //
    // Data members
    //

    /// Buffer for B operand
    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

   public:
    //
    // Methods
    //

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static LayoutB LayoutB() {
      return LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_ref() {
      return TensorRefB{operand_B.data(), LayoutB()};
    }
  };

 public:
  /// Construct from tensor references
  CUTLASS_DEVICE
  MmaBaseFromSharedMemoryAiu(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      TensorRefB& b_tile,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx) {}
};

template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Size of the WarpGemm problem - concept: gemm::GemmShape<>
    typename WarpShape_,
    /// Iterates over the intermediate accumulator tile in shared memory
    typename WarpIteratorA_,
    /// whether or not to perform elementwise multiplication of A
    //  by another matrix (A_scale) that is also kept in shared memory prior
    //  to matmul A @ B
    bool ScaleOperandA_,
    /// Aiu Iterates over tiles of B operand in global memory
    //  (concept: AiuLoaderB)
    typename IteratorB_,
    /// Policy describing tuning details (concept: MmaPolicyAiu)
    typename Operator_,
    /// Policy describing tuning details (concept: MmaPolicyCutlass)
    typename Policy_,
    /// Number of stages,
    int Stages_,
    int kMaxK_,
    /// B in TSM need transpose.
    bool kTransposeB = false>
class MmaMultistageFromSharedMemoryAiu : public MmaBaseFromSharedMemoryAiu<
                                          Shape_,
                                          WarpShape_,
                                          kMaxK_,
                                          Operator_,
                                          Policy_,
                                          Stages_,
                                          typename WarpIteratorA_::Layout,
                                          kTransposeB> {
 public:
  ///< Base class
  using Base = MmaBaseFromSharedMemoryAiu<
      Shape_,
      WarpShape_,
      kMaxK_,
      Operator_,
      Policy_,
      Stages_,
      typename WarpIteratorA_::Layout,
      kTransposeB>;

  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;

  using WarpIteratorA = WarpIteratorA_; ///< Iterates over the intermediate
                                          ///< accumulator tile in shared memory
  static constexpr bool ScaleOperandA = ScaleOperandA_;

  ///< warp level iterator over A_scale matrix tile kept in shared memory.
  ///< if elementwise A scaling is disabled then everything this does is no-op.
  using WarpIteratorAScale = typename cutlass::platform::conditional<
      ScaleOperandA,
      WarpIteratorA,
      NoOpWarpIteratorScale<typename WarpIteratorA::TensorRef>>::type;

  static constexpr bool kSmemContainsEntireB = Base::kSmemContainsEntireB;

  //
  // Dependent types
  //

  /// Warp-level Mma
  using Operator_Aiu = Operator_;

    /// Warp-level Mma
  using Operator = typename Policy_::Operator;

  /// Fragment of accumulator tile
  using FragmentC = typename Operator::FragmentC;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  static constexpr int kNumStagesConcurrentLoad =
      kSmemContainsEntireB ? Base::kStages : Base::kStages - 1;

  using ElementB_ = typename FromCutlassType<typename IteratorB::Element>::type;

  static constexpr int B_TSM_OFFSET = (Shape::kN * Shape::kK);

  ElementB_* tsm_bstart_;

 private:
  using WarpLoadedFragmentA = typename Operator::FragmentA;
  /// fragment of OperandA scale matrix. if operand A scaling is disabled this
  /// is (almost) empty.
  using WarpLoadedFragmentAScale = typename WarpIteratorAScale::Fragment;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

  /// applies elementwise scaling to fragment of A. if operand A scaling is
  /// disabled this is a no-op.
  using FragmentAScaler = FragmentElementwiseScaler<
      WarpLoadedFragmentA,
      WarpLoadedFragmentAScale,
      ScaleOperandA>;

 private:
  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A1 operand from intermediate
  /// accumulator tile
  WarpIteratorA warp_tile_iterator_A_;

  /// Iterator to load a warp-scoped tile of A1_scale operand from shared memory
  /// if operand A scaling is disabled everything this does is a no-op.
  WarpIteratorAScale warp_tile_iterator_A_scale_;

 public:
  /// constructor for MMA with operand A scaling enabled.
  CUTLASS_DEVICE
  MmaMultistageFromSharedMemoryAiu(
      typename Base::TensorRefA a,
      typename Base::TensorRefA a_scale,
      typename Base::TensorRefB b_tile,
      int thread_idx,
      int warp_idx,
      int lane_idx)
      : Base(b_tile, thread_idx, warp_idx, lane_idx),
        warp_tile_iterator_A_(a, lane_idx),
        warp_tile_iterator_A_scale_(a_scale, lane_idx) {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension
    int warp_idx_mn_1 =
        warp_idx % (Base::WarpCount1::kM * Base::WarpCount1::kN);
    int warp_idx_k_1 = warp_idx / (Base::WarpCount1::kM * Base::WarpCount1::kN);
    int warp_idx_m_1 = warp_idx_mn_1 % Base::WarpCount1::kM;
    int warp_idx_n_1 = warp_idx_mn_1 / Base::WarpCount1::kM;

    // Add per-warp offsets in units of warp-level tiles
    warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m_1, Base::kWarpGemmIterations1 * warp_idx_k_1});
    warp_tile_iterator_A_scale_.add_tile_offset(
        {warp_idx_m_1, Base::kWarpGemmIterations1 * warp_idx_k_1});
    tsm_bstart_ = reinterpret_cast<ElementB_*>(b_tile.data());
  }

  /// Construct from tensor references
  CUTLASS_DEVICE
  MmaMultistageFromSharedMemoryAiu(
      typename Base::TensorRefA a,
      typename Base::TensorRefB b_tile,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx)
      : Base(b_tile, thread_idx, warp_idx, lane_idx),
        warp_tile_iterator_A_(a, lane_idx),
        warp_tile_iterator_A_scale_(a, lane_idx) {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension
    int warp_idx_mn_1 =
        warp_idx % (Base::WarpCount1::kM * Base::WarpCount1::kN);
    int warp_idx_k_1 = warp_idx / (Base::WarpCount1::kM * Base::WarpCount1::kN);

    int warp_idx_m_1 = warp_idx_mn_1 % Base::WarpCount1::kM;
    int warp_idx_n_1 = warp_idx_mn_1 / Base::WarpCount1::kM;

    // Add per-warp offsets in units of warp-level tiles
    warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m_1, Base::kWarpGemmIterations1 * warp_idx_k_1});
    tsm_bstart_ = reinterpret_cast<ElementB_*>(b_tile.data());
  }

  CUTLASS_DEVICE
  static void prologue(
      typename Base::SharedStorage& shared_storage,
      IteratorB iterator_B,
      int thread_idx,
      int problem_size_0_n) {
    ElementB_* tsm_bstart_ = reinterpret_cast<ElementB_*>(shared_storage.operand_B.data());

    // prefetch for the Stages iteration
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < kSmemContainsEntireB; ++stage) {
      ElementB_* tsm_bstart_staged = tsm_bstart_ + stage * B_TSM_OFFSET;
      iterator_B.LoadToTsm(tsm_bstart_staged);

      __pipeline_commit();
    }
  }

  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations_1_,
      ///< destination accumulator tile
      FragmentC& accum,
      ///< iterator over B1 operand from shared memory
      IteratorB &warp_tile_iterator_B_,
      ///< initial value of accumulator
      FragmentC const& src_accum,
      bool entir_load) {
    // 2nd Aiu Gemm

    // Perform accumulation in the 'd' output operand
    accum = src_accum;
    
    // Waits until kStages-2 stages have committed.
    __pipeline_wait_prior(kNumStagesConcurrentLoad - 1);
    #if SAIL_PPU_MMA
      __ppu_barrier_sync_nocnt(0, 1);
    #else
      __syncthreads();
    #endif

    WarpLoadedFragmentA warp_loaded_frag_A;
    WarpLoadedFragmentAScale warp_loaded_frag_A_scale;
    WarpLoadedFragmentB warp_loaded_frag_B;
    WarpTransformedFragmentA warp_transformed_frag_A;
    WarpTransformedFragmentB warp_transformed_frag_B;

    Operator warp_mma;

    // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
    // accumulator and this temporary accumulator is added to the final
    // accumulator once in every mainloop iteration.
    plus<FragmentC> plus_accum;

    FragmentC tmp_accum;

    if (platform::is_same<
            typename Operator::MathOperator,
            arch::OpMultiplyAddFastF32>::value ||
        platform::is_same<
            typename Operator::MathOperator,
            arch::OpMultiplyAddComplexFastF32>::value) {
      tmp_accum.clear();
    }

    gemm_k_iterations_1_ -= kNumStagesConcurrentLoad;

    // FIXME: greater vreg usage, llvm is checking.
    // CUTLASS_PRAGMA_UNROLL
    // for (; gemm_k_iterations_1_ > (-kNumStagesConcurrentLoad);) {
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations1;
           ++warp_mma_k) {
        warp_tile_iterator_A_.load(warp_loaded_frag_A);
        warp_tile_iterator_A_scale_.load(warp_loaded_frag_A_scale);
        if (kTransposeB) {
          warp_tile_iterator_B_.LoadToVregWithTsmTrans(tsm_bstart_,
            CONVERT_MMA_FRAGMENT_VALUE(warp_loaded_frag_B, typename IteratorB::TransFragType), warp_mma_k);
        } else {
          warp_tile_iterator_B_.LoadToVreg(tsm_bstart_,
            CONVERT_MMA_FRAGMENT_VALUE(warp_loaded_frag_B, typename IteratorB::FragType), warp_mma_k);
        }

        ++warp_tile_iterator_A_;
        ++warp_tile_iterator_A_scale_;

        warp_mma.transform(
          warp_transformed_frag_A,
          warp_transformed_frag_B,
          FragmentAScaler::apply(
              warp_loaded_frag_A,
              warp_loaded_frag_A_scale),
          warp_loaded_frag_B);

        if (platform::is_same<
              typename Operator::MathOperator,
              arch::OpMultiplyAddFastF32>::value ||
          platform::is_same<
              typename Operator::MathOperator,
              arch::OpMultiplyAddComplexFastF32>::value) {
          warp_mma(
              tmp_accum,
              warp_transformed_frag_A,
              warp_transformed_frag_B,
              tmp_accum);

          if (warp_mma_k == 0) {
            accum = plus_accum(accum, tmp_accum);
            tmp_accum.clear();
          }
        } else {

          warp_mma(
            accum,
            warp_transformed_frag_A,
            warp_transformed_frag_B,
            accum);
        }
      }
      --gemm_k_iterations_1_;
    // }
    
    warp_tile_iterator_B_.add_tile_offset({- Base::kWarpGemmIterations1, 0});
  }
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass