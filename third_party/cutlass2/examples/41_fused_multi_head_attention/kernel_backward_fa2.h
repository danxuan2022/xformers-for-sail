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
#include "kernel_backward_base.h"
#include "fmha_utils.h"

template <
    // which arch we target (eg `cutlass::arch::Sm80`)
    typename ArchTag_,
    // input/output type
    typename scalar_t_,
    // run optimized kernel because memory accesses will be aligned
    bool kIsAligned_,
    // use dropout if enabled
    bool kApplyDropout_,
    // block dimensions
    int kBlockSizeI_,
    int kBlockSizeJ_,
    // upperbound on `max(value.shape[-1], query.shape[-1])`
    int kMaxK_,
    // assumes that `cu_seqlen` is None, and
    // (1) `num_queries % kBlockSizeI == 0`
    // (2) `num_keys % kBlockSizeJ == 0`
    bool kKeysQueriesAlignedToBlockSize_,
    // warp dimensions
    int kWarpNum,
    int kWarpDp,
    int kWarpKv,
    int kWarpDq>
struct AttentionBackwardKernel<
  ArchTag_, scalar_t_, kIsAligned_, kApplyDropout_, true,
  kBlockSizeI_, kBlockSizeJ_, kMaxK_,
  kKeysQueriesAlignedToBlockSize_, true, true,
  kWarpNum, kWarpDp, kWarpKv, kWarpDq> {

  using scalar_t = scalar_t_;
  using output_t = scalar_t;
  using output_accum_t = float;
  using lse_scalar_t = float;
  using accum_t = float;
  using ArchTag = ArchTag_;
  static constexpr bool kIsAligned = kIsAligned_;
  static constexpr bool kApplyDropout = kApplyDropout_;
  static constexpr int kBlockSizeI = kBlockSizeI_;
  static constexpr int kBlockSizeJ = kBlockSizeJ_;
  static constexpr int kMaxK = kMaxK_;
  static constexpr bool kNeedsAccumGradV = false;
  static constexpr bool kKeysQueriesAlignedToBlockSize =
      kKeysQueriesAlignedToBlockSize_;

  static constexpr int64_t kWarpSize = 32;
  static constexpr int64_t kNumWarpsPerBlock = kWarpNum;

  // Launch bounds
  static constexpr int64_t kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int64_t kMinBlocksPerSm =
      getWarpsPerSmBw<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  static constexpr bool kKernelComputesDelta = true;
  static constexpr bool kNeedsAccumGradK = false;
  static constexpr bool kEnableSplitKeys = true;

  using GemmType = DefaultGemmType<ArchTag, scalar_t>;
  using DefaultConfig =
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          typename GemmType::OpClass,
          ArchTag,
          scalar_t,
          scalar_t,
          scalar_t, // ElementC
          accum_t // ElementAccumulator
          >;
  static constexpr auto kOptimalAlignement = cutlass::platform::max(
      DefaultConfig::kAlignmentA,
      DefaultConfig::kAlignmentB);
  static constexpr auto kMinimumAlignment = GemmType::kMinimumAlignment;

  struct MatmulQK {
    /*
    attn_T = q_i @ k_j.transpose(-2, -1) # matmul
    attn_T = (attn_T - logsumexp[i_start:i_end].unsqueeze(1).transpose(-2,
    -1)).exp() # epilogue

    with attn_T.shape = (kBlockSizeI, kBlockSizeJ)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, kMaxK>;
    using WarpShape = cutlass::gemm::GemmShape<
          // kBlockSizeJ / kWarpDp, kBlockSizeI / (kWarpNum / kWarpDp), kMaxK>;
          kBlockSizeI / kWarpDp, kBlockSizeJ / (kWarpNum / kWarpDp), kMaxK>;

    using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        accum_t, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        DefaultConfig::kStages,
        typename GemmType::Operator
    >::DefaultMma;
    using MmaCore = typename DefaultMma::MmaCore;
    using Mma =
        typename MakeCustomMma<typename DefaultMma::ThreadblockMma, kMaxK>::Mma;

    // used for efficient load of bias tile (Bij) from global memory to shared
    // memory
     using BiasLoader = TileSmemLoader<
        scalar_t,
        // Bij is applied to transposed attn matrix tile (Pij.T). Bij is loaded
        // row-major but needs to have transposed shape so we get the same
        // elements.
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kN>,
        MmaCore::kThreads,
        // input restriction: kv_len has to be a multiple of this value
        128 / cutlass::sizeof_bits<scalar_t>::value>;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape,
        true>;
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Iterator;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradV {
    /*
    grad_v[j_start:j_end] += attn_T @ do_i # matmul

    Dimensions: (kBlockSizeJ * kNumWarpsPerBlock, kBlockSizeI, K)
    (we might need to iterate multiple times on K)
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kMaxK, kBlockSizeI>;
    using WarpShape = cutlass::gemm::GemmShape<
        kBlockSizeJ / kWarpKv,
        kMaxK / (kWarpNum / kWarpKv),
        kBlockSizeI>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
      scalar_t, // ElementA,
      cutlass::layout::RowMajor, // LayoutA,
      DefaultConfig::kAlignmentA,
      scalar_t, // ElementB,
      cutlass::layout::RowMajor, // LayoutB,
      kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
      accum_t,
      cutlass::layout::RowMajor, // LayoutC,
      typename GemmType::OpClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      typename GemmType::InstructionShape,
      DefaultConfig::kStages,
      typename GemmType::Operator
    >::DefaultMma;

    // if dropout:
    //   for computing dVj += (Pij.T * Zij) @ dOi
    //   Pij_dropped.T = Pij.T * Zij is computed on the fly as fragments of
    //   Pij.T are loaded in. The reason we do it this way is because Pij.T and
    //   Zij are reused in later steps, while Pij_dropped.T is only needed in
    //   this step. computing Pij_dropped.T on the fly allows us to avoid
    //   keeping all 3 of Pij_dropped.T, Pij.T, and Zij in shared memory at the
    //   same time.
    // if no dropout:
    //   for computing dVj += Pij.T @ dOi
    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape, // WarpShape
            typename DefaultGemm::Mma::Operator::
                InstructionShape, // InstructionShape
            typename DefaultGemm::Mma::Operator::
                IteratorA, // RegularWarpIterator
            typename DefaultGemm::Mma::Policy, // Policy
            bool
            >::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultMma::ThreadblockMma,
            // typename DefaultGemm::Mma,
            // MatmulQK::AccumulatorSharedStorage::Shape::kN,
            MatmulQK::AccumulatorSharedStorage::Shape::kM,
            WarpIteratorA,
            typename DefaultGemm::Mma::Policy, // Policy
            kApplyDropout,
            true>; // kScaleOperandA

    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;
  };

  struct MatmulDOIVJ {
    /*
    doi_t_vj = do_i @ v_j.transpose(-2, -1) # matmul
    tmp = (doi_t_vj - Di.unsqueeze(1)) * attn # inplace / epilogue?
    */
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kBlockSizeJ, kMaxK>;
    using WarpShape = cutlass::gemm::GemmShape<
        kBlockSizeI / kWarpDp, kBlockSizeJ / (kWarpNum / kWarpDp), kMaxK>;

    using ElementC = output_t;
    using ElementAccum = accum_t;

    using DefaultGemm = typename cutlass::gemm::threadblock::FindDefaultMma<
        scalar_t, // ElementA
        cutlass::layout::RowMajor, // LayoutA
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment,
        scalar_t, // ElementB
        cutlass::layout::ColumnMajor, // LayoutB
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        ElementAccum, // ElementC
        cutlass::layout::RowMajor, // LayoutC
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        DefaultConfig::kStages,
        typename GemmType::Operator
    >::DefaultMma;
    using Mma = typename MakeCustomMma<typename DefaultGemm::ThreadblockMma, kMaxK>::Mma;
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        ElementAccum,
        kWarpSize>::Iterator;

    // no-op output op - epilogue just stores result to global memory
    using BiasGradEpilogueOutputOp =
        typename cutlass::epilogue::thread::LinearCombination<
            ElementC,
            DefaultConfig::EpilogueOutputOp::kCount,
            typename DefaultConfig::EpilogueOutputOp::ElementAccumulator,
            typename DefaultConfig::EpilogueOutputOp::ElementCompute,
            cutlass::epilogue::thread::ScaleType::Nothing>;

    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using BiasGradEpilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape, typename Mma::Operator, DefaultEpilogue::kPartitionsK, BiasGradEpilogueOutputOp,
        BiasGradEpilogueOutputOp::kCount>::Epilogue;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MatmulGradQ {
    // grad_q <- tmp @ k_j
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeI, kMaxK, kBlockSizeJ>;
    using WarpShape = cutlass::gemm::GemmShape<
        kBlockSizeI / kWarpDq,
        kMaxK / (kWarpNum / kWarpDq),
        kBlockSizeJ>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        // cutlass::layout::RowMajor, // LayoutB,
        cutlass::layout::ColumnMajor,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        DefaultConfig::kStages,
        typename GemmType::Operator
    >::DefaultMma;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape,
            typename DefaultGemm::Mma::Operator::InstructionShape,
            typename DefaultGemm::Mma::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy,
            bool>::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultMma::ThreadblockMma,
            // typename DefaultGemm::Mma,
            MatmulDOIVJ::AccumulatorSharedStorage::Shape::kN,
            WarpIteratorA,
            typename DefaultGemm::Mma::Policy, // Policy
            false, // kScaleOperandA
            false, // kTransposeA
            true>; // kIsTransposeB
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;
  };

  struct MatmulGradK {
    // grad_k <- tmp.transpose(-2, -1) @ q_i
    using ThreadblockShape =
        cutlass::gemm::GemmShape<kBlockSizeJ, kMaxK, kBlockSizeI>;
    using WarpShape = cutlass::gemm::GemmShape<
        kBlockSizeJ / kWarpKv,
        kMaxK / (kWarpNum / kWarpKv),
        kBlockSizeI>;
    using InstructionShape = typename GemmType::InstructionShape;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        DefaultConfig::kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::RowMajor, // LayoutB,
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
        output_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        typename GemmType::OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
      scalar_t, // ElementA,
      cutlass::layout::RowMajor, // LayoutA,
      DefaultConfig::kAlignmentA,
      scalar_t, // ElementB,
      cutlass::layout::RowMajor, // LayoutB,
      kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment,
      accum_t,
      cutlass::layout::RowMajor, // LayoutC,
      typename GemmType::OpClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      typename GemmType::InstructionShape,
      DefaultConfig::kStages,
      typename GemmType::Operator
    >::DefaultMma;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Operator::Shape,
            typename DefaultGemm::Mma::Operator::InstructionShape,
            typename DefaultGemm::Mma::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy,
            bool>::WarpIterator;
    using DefaultMmaFromSmemN =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultMma::ThreadblockMma,
            // typename DefaultGemm::Mma,
            // MatmulQK::AccumulatorSharedStorage::Shape::kN, // kMaxK
            MatmulQK::AccumulatorSharedStorage::Shape::kM, // kMaxK
            WarpIteratorA,
            typename DefaultGemm::Mma::Policy, // Policy
            false>; // kScaleOperandA
    using DefaultMmaFromSmemT =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultMma::ThreadblockMma,
            // typename DefaultGemm::Mma,
            MatmulDOIVJ::AccumulatorSharedStorage::Shape::kM, // kMaxK
            WarpIteratorA,
            typename DefaultGemm::Mma::Policy, // Policy
            false, // kScaleOperandA
            true>; // kTransposeA
    using DefaultMmaFromSmem = typename cutlass::platform::conditional<
        DefaultMmaFromSmemT::kIsTransposedA,
        DefaultMmaFromSmemT,
        DefaultMmaFromSmemN>::type;
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;

    // Epilogue
    using DefaultOutputOp = typename DefaultConfig::EpilogueOutputOp;
    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::MakePrefetchableIterator<
            typename DefaultEpilogue::OutputTileIterator>::Iterator;
    using AccumTileGmem = GmemTile<typename Mma::FragmentC, (int)kNumThreads>;
  };

  struct GradQTempStorage {
    // int32_t lock;
    // int32_t counter;
    // int32_t pad[2]; // pad to 128bits
    output_accum_t buffer[MatmulGradQ::AccumTileGmem::kElementsStored];
  };

  struct Params {
    // Input tensors
    scalar_t* query_ptr = nullptr; // [Mq, nH, K]
    scalar_t* key_ptr = nullptr; // [Mk, nH, K]
    scalar_t* value_ptr = nullptr; // [Mk, nH, Kv]
    scalar_t* bias_ptr = nullptr;
    lse_scalar_t* logsumexp_ptr = nullptr; // [nH, Mq]
    scalar_t* output_ptr = nullptr; // [Mq, nH, Kv]
    scalar_t* grad_output_ptr = nullptr; // [Mq, nH, Kv]
    accum_t* delta_ptr = nullptr; // [nH, Mq]
    int32_t* cu_seqlens_q_ptr = nullptr;
    int32_t* cu_seqlens_k_ptr = nullptr;

    // Output tensors
    output_t* grad_query_ptr = nullptr; //  [Mq, nH, K]
    output_t* grad_key_ptr = nullptr; //    [Mk, nH, K]
    output_t* grad_value_ptr = nullptr; //  [Mk, nH, Kv]
    output_t* grad_bias_ptr = nullptr;

    // Accumulators
    output_accum_t* workspace = nullptr; // [Mq, Kq] + [Mkv, Kq] + [Mkv, Kv]
    GradQTempStorage* workspace_gq =
        nullptr; // (will be calculated by the kernel)

    // Scale
    accum_t scale = 1.0f;

    // Dimensions/strides
    int32_t head_dim = -1;
    int32_t head_dim_value = -1;
    int32_t num_queries = -1;
    int32_t num_keys = -1;
    int32_t num_heads = -1;
    uint8_t custom_mask_type = NoCustomMask;

    int32_t q_strideM = -1;
    int32_t k_strideM = -1;
    int32_t v_strideM = -1;
    int32_t bias_strideM = 0;
    int32_t gO_strideM = -1;
    int32_t gB_strideM = -1;
    int8_t gQKV_strideM_multiplier = 1; // 3 for packed, 1 otherwise

    int32_t gQ_strideM_ = -1;
    int32_t gK_strideM_ = -1;
    int32_t gV_strideM_ = -1;

#ifdef HAS_PYTORCH
    // dropout
    at::PhiloxCudaState rng_engine_inputs = {0, 0};
    uint64_t * rng_state = nullptr;
#endif
    // RNG sequence offset based on batch_id and head_id
    unsigned long long dropout_batch_head_rng_offset = 0;
    float dropout_prob = 0.0f;

    CUTLASS_HOST_DEVICE int32_t o_strideM() const {
      return head_dim_value * num_heads;
    }
    CUTLASS_HOST_DEVICE int32_t gQ_strideM() const {
      if (gQ_strideM_ > 0) {
        return gQ_strideM_;
      }
      return gQKV_strideM_multiplier * num_heads * head_dim;
    }
    CUTLASS_HOST_DEVICE int32_t gK_strideM() const {
      if (gK_strideM_ > 0) {
        return gK_strideM_;
      }
      return gQKV_strideM_multiplier * num_heads * head_dim;
    }
    CUTLASS_HOST_DEVICE int32_t gV_strideM() const {
      if (gV_strideM_ > 0) {
        return gV_strideM_;
      }
      return gQKV_strideM_multiplier * num_heads * head_dim_value;
    }

    // Everything below is only used in `advance_to_block`
    // and shouldn't use registers
    int64_t o_strideH = -1;
    int32_t q_strideH = -1;
    int32_t k_strideH = -1;
    int32_t v_strideH = -1;
    int64_t bias_strideH = 0;
    int64_t o_strideB = -1;
    int64_t q_strideB = -1;
    int64_t k_strideB = -1;
    int64_t v_strideB = -1;
    int64_t bias_strideB = 0;
    int64_t lse_strideB = -1;
    int64_t lse_strideH = -1;
    int64_t delta_strideB = -1;
    int64_t delta_strideH = -1;
    int32_t num_batches = -1;
    int16_t num_splits_key = 1; // We use `gridDim.x` inside kernel

    int64_t gO_strideB = 0;
    int64_t gQ_strideB = 0;
    int64_t gK_strideB = 0;
    int64_t gV_strideB = 0;
    int64_t gB_strideB = 0;
    int64_t gO_strideH = 0;
    int64_t gQ_strideH = 0;
    int64_t gK_strideH = 0;
    int64_t gV_strideH = 0;
    int64_t gB_strideH = 0;

    CUTLASS_DEVICE int16_t num_splits_key_device() const {
      return gridDim.x;
    }
    CUTLASS_DEVICE int16_t split_key_device() const {
      return blockIdx.x;
    }

    CUTLASS_DEVICE bool advance_to_block() {
      int64_t batch_id = blockIdx.y;
      int32_t head_id = blockIdx.z;

      assert(workspace_size() == 0 || workspace != nullptr);

      workspace_gq =
          (GradQTempStorage*)(workspace + (batch_id * num_heads + head_id) * workspace_elements_gq());
      workspace_gq = warp_uniform(workspace_gq);

      // Advance pointers that depend on the total concatenated
      // number of queries, as `num_queries` is modified in the block
      // below
      dropout_batch_head_rng_offset = batch_id * num_heads + head_id;
      logsumexp_ptr += batch_id * lse_strideB + head_id * lse_strideH;

      if (cu_seqlens_q_ptr != nullptr) {
        assert(cu_seqlens_k_ptr != nullptr);
        cu_seqlens_q_ptr += batch_id;
        cu_seqlens_k_ptr += batch_id;
        int32_t q_start = cu_seqlens_q_ptr[0];
        int32_t k_start = cu_seqlens_k_ptr[0];
        int64_t q_next_start = cu_seqlens_q_ptr[1];
        int64_t k_next_start = cu_seqlens_k_ptr[1];
        assert(q_next_start - q_start <= num_queries);
        assert(k_next_start - k_start <= num_keys);
        num_queries = q_next_start - q_start;
        num_keys = k_next_start - k_start;

        // Jump manually
        batch_id = 0;

        query_ptr += q_start * q_strideM;
        key_ptr += k_start * k_strideM;
        value_ptr += k_start * v_strideM;
        assert(bias_ptr == nullptr);
        assert(grad_bias_ptr == nullptr);
        output_ptr += q_start * o_strideM();
        grad_output_ptr += q_start * gO_strideM;
        delta_ptr += q_start;

        grad_query_ptr += q_start * gQ_strideM();
        grad_key_ptr += k_start * gK_strideM();
        grad_value_ptr += k_start * gV_strideM();
      }

      query_ptr += batch_id * q_strideB + head_id * q_strideH;
      key_ptr += batch_id * k_strideB + head_id * k_strideH;
      value_ptr += batch_id * v_strideB + head_id * v_strideH;
      if (bias_ptr != nullptr) {
        bias_ptr += batch_id * bias_strideB + head_id * bias_strideH;
      }

      output_ptr += batch_id * o_strideB + head_id * o_strideH;
      grad_output_ptr += batch_id * gO_strideB + head_id * gO_strideH;
      delta_ptr += batch_id * delta_strideB + head_id * delta_strideH;

      grad_query_ptr += batch_id * gQ_strideB + head_id * gQ_strideH;
      grad_key_ptr += batch_id * gK_strideB + head_id * gK_strideH;
      grad_value_ptr += batch_id * gV_strideB + head_id * gV_strideH;
      if (grad_bias_ptr != nullptr) {
        grad_bias_ptr += batch_id * gB_strideB + head_id * gB_strideH;
      }

      // Some values are modified above
      // Signal to the compiler that they are the same in all threads
      // and can be stored in warp-uniform registers (Sm75+)
      num_queries = warp_uniform(num_queries);
      num_keys = warp_uniform(num_keys);
      custom_mask_type = warp_uniform(custom_mask_type);

      query_ptr = warp_uniform(query_ptr);
      key_ptr = warp_uniform(key_ptr);
      value_ptr = warp_uniform(value_ptr);
      bias_ptr = warp_uniform(bias_ptr);
      logsumexp_ptr = warp_uniform(logsumexp_ptr);
      output_ptr = warp_uniform(output_ptr);
      grad_output_ptr = warp_uniform(grad_output_ptr);
      delta_ptr = warp_uniform(delta_ptr);

      grad_query_ptr = warp_uniform(grad_query_ptr);
      grad_key_ptr = warp_uniform(grad_key_ptr);
      grad_value_ptr = warp_uniform(grad_value_ptr);
      grad_bias_ptr = warp_uniform(grad_bias_ptr);

// #if 0
//       PRINT_T0("[b:%d h:%d] dp[0]:%f Q:%f K:%f V:%f LSE:%f",
//         int(blockIdx.z), int(blockIdx.y),
//         float(delta_ptr[0]),
//         float(query_ptr[0]), float(key_ptr[0]), float(value_ptr[0]),
//         float(logsumexp_ptr[0])
//       )
// #endif
      return true;
    }

    __host__ dim3 getBlocksGridDq() const {
      return dim3(ceil_div(num_queries, kBlockSizeI), num_batches, num_heads);
    }

    __host__ dim3 getBlocksGrid() const {
      return dim3(num_splits_key, num_batches, num_heads);
    }

    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize * kNumWarpsPerBlock, 1, 1);
    }

    template <typename MatmulT>
    __host__ void printMmInfo(const char* s_name) {
      printf("  %-15s: block[%d, %d, %d], warp[%d, %d, %d], stage:%d\n",
        s_name,
        MatmulT::ThreadblockShape::kM, MatmulT::ThreadblockShape::kN, MatmulT::ThreadblockShape::kK,
        MatmulT::WarpShape::kM, MatmulT::WarpShape::kN, MatmulT::WarpShape::kK,
        MatmulT::Mma::kStages);
    }

    __host__ void printKernelInfo(const void *func, bool debug_smem = false) {
      cudaFuncAttributes attr;
      cudaFuncGetAttributes(&attr, func);
      int max_blocks_per_cu;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_cu, func, getThreadsGrid().x, sizeof(SharedStorage));

      printf("=====fmha backward configuration=====\n");
      printf("  Input: batch:%d, head size:%d, query:%d, key:%d, head dim:%d, dropout:%f, custom_mask:%d\n",
        num_batches, num_heads, num_queries, num_keys, head_dim, dropout_prob, custom_mask_type);
      printf("  BlockSizeI:%d, BlockSizeJ:%d, MaxK:%d, WarpNum:%d, warpDp:%d, warpKv:%d, warpDq:%d, split_key:%d, alignBlockSize:%d\n",
        kBlockSizeI, kBlockSizeJ, kMaxK, kWarpNum, kWarpDp, kWarpKv, kWarpDq, num_splits_key, kKeysQueriesAlignedToBlockSize);
      printf("  Grid[%d, %d, %d], block[%d, %d, %d], sharedMem:%d(k), workspace:%d(k), vreg:%d, stack:%d, occupancy:%d\n",
        getBlocksGrid().x, getBlocksGrid().y, getBlocksGrid().z,
        getThreadsGrid().x, getThreadsGrid().y, this->getThreadsGrid().z,
        int(sizeof(SharedStorage) / 1024), int(workspace_size() / 1024),
        int(attr.numRegs), int(attr.localSizeBytes), max_blocks_per_cu);
      printMmInfo<MatmulQK>("MatmulQK");
      printMmInfo<MatmulGradV>("MatmulGradV");
      printMmInfo<MatmulDOIVJ>("MatmulDOIVJ");
      printMmInfo<MatmulGradQ>("MatmulGradQ");
      printMmInfo<MatmulGradK>("MatmulGradK");
      if (debug_smem)
        SharedStorage::print_size();
      printf("\n");
    }

    CUTLASS_HOST_DEVICE int64_t workspace_elements_gq() const {
      int num_blocks = ceil_div(num_queries, kBlockSizeI);
      return  num_blocks * workspace_elements_cols();
    }

    CUTLASS_HOST_DEVICE int64_t workspace_elements_cols() const {
      int num_cols = ceil_div(head_dim, MatmulGradQ::ThreadblockShape::kN);

      return num_cols * sizeof(GradQTempStorage) /
          sizeof(output_accum_t);
    }

    CUTLASS_HOST_DEVICE int64_t workspace_size_gq() const {
      // Returns size of buffer we need to run this kernel
      return num_batches * num_heads * workspace_elements_gq() * sizeof(float);
    }

    CUTLASS_HOST_DEVICE int64_t workspace_size() const {
      // Returns size of buffer we need to run this kernel
      return num_batches * num_heads * workspace_elements_gq() * sizeof(float);
    }

    CUTLASS_HOST_DEVICE int should_zero_workspace() const {
      return false;
    }
  };

  // shared storage for keeping Zij matrix. not needed if we aren't using
  // dropout, in which case we use an empty array to save shared memory
  using ZijSharedStorage = typename cutlass::platform::conditional<
      kApplyDropout,
      typename MatmulQK::AccumulatorSharedStorage,
      // dummy shared storage object that takes up no space.
      typename cutlass::gemm::threadblock::AccumulatorSharedStorage<
          typename cutlass::gemm::GemmShape<0, 0, 0>,
          typename MatmulQK::AccumulatorSharedStorage::Element,
          typename MatmulQK::AccumulatorSharedStorage::Layout,
          typename cutlass::MatrixShape<0, 0>>>::type;


  struct SharedStoragePrologue {
    union {
      struct {
        struct {
          cutlass::Array<accum_t, kBlockSizeI> di; // (do_i * o_i).sum(-1)
          // typename MatmulQK::Mma::SharedStorageA mm_qk_k;
          typename MatmulQK::Mma::SharedStorageB mm_qk_k;
          typename MatmulDOIVJ::Mma::SharedStorageB mm_doivj_v;
          // typename MatmulQK::Mma::SharedStorageA mm_qk_q;
        } persistent;
        union {
          struct {
            // part1 - after Q.K / dV / dO.V
            union {
              // 1. efficient load of bias tile Bij, which is then applied to Pij
              typename MatmulQK::BiasLoader::SmemTile bias;
              // 4. store Pij. it is needed:
              // - in dVj += (Pij.T * Zij) @ dOi
              // - in dSij = Pij * (dPij - Di)
              // 6. dVj += (Pij.T * Zij) @ dOi
              // 10. write to fragment
              typename MatmulQK::AccumulatorSharedStorage attn_shared_storage;
            };
            // 5. store Zij. it is needed in dVj += (Pij.T * Zij) @ dOi
            ZijSharedStorage zij;
            typename MatmulDOIVJ::Mma::SharedStorageA mm_doivj_do;
            } part1;

          struct {
            // part2 - dQ
            union {
              typename MatmulQK::AccumulatorSharedStorage
                  tmpT_shared_storage; // (from part1)
              typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
            };
            typename MatmulGradK::Mma::SharedStorage mm_gradK; // (preload)
            typename MatmulDOIVJ::BiasGradEpilogue::SharedStorage gradB_epilogue;
          } part2;

          struct {
            // part3 - after last iteration on dQ's epilogue / dK
            union {
              typename MatmulQK::AccumulatorSharedStorage
                  tmpT_shared_storage; // (from part1)
              typename MatmulDOIVJ::AccumulatorSharedStorage tmp_shared_storage;
            };
            
            // typename MatmulGradQ::Mma::SharedStorage mm_gradQ; // (preload)
          } part3;

          struct {
            // part4 - after last iteration on dK's epilogue / preload next K.Q_t
            typename MatmulQK::Mma::SharedStorageA mm_qk_q;
          } part4;
        };
      } mma;

      struct {
        // If we reach end of current key, dump RF->gmem with "final" epilogues
        typename MatmulGradK::DefaultEpilogue::SharedStorage
            gradK_epilogue_final;
        typename MatmulGradV::DefaultEpilogue::SharedStorage
            gradV_epilogue_final;
      } epilogue;
    };

    static void print_size() {
      // Field size
#define FSZ(f) int((sizeof(((SharedStoragePrologue*)0)->f)))

      printf("Total smem: %d bytes\n", int(sizeof(SharedStoragePrologue)));
      printf("  mma: %db\n", FSZ(mma));
      printf("    persistent: %db\n", FSZ(mma.persistent));
      printf("      mm_qk_k: %db\n", FSZ(mma.persistent.mm_qk_k));
      printf("      mm_doivj_v: %db\n", FSZ(mma.persistent.mm_doivj_v));
      // printf("      mm_qk_q: %db\n", FSZ(mma.persistent.mm_qk_q));
      printf("    part1: %db\n", FSZ(mma.part1));
      printf("      bias: %db\n", FSZ(mma.part1.bias));
      printf("      attn_shared_storage: %db\n", FSZ(mma.part1.attn_shared_storage));
      printf("      zij: %db\n", FSZ(mma.part1.zij));
      printf("      mm_doivj_do: %db\n", FSZ(mma.part1.mm_doivj_do));
      printf("    part2: %db\n", FSZ(mma.part2));
      printf("      tmpT_shared_storage: %db\n", FSZ(mma.part2.tmpT_shared_storage));
      printf("      tmp_shared_storage: %db\n", FSZ(mma.part2.tmp_shared_storage));
      printf("      mm_gradK: %db\n", FSZ(mma.part2.mm_gradK));
      printf("      gradB_epilogue: %db\n", FSZ(mma.part2.gradB_epilogue));
      printf("    part3: %db\n", FSZ(mma.part3));
      printf("      tmpT_shared_storage: %db\n", FSZ(mma.part3.tmpT_shared_storage));
      printf("      tmp_shared_storage: %db\n", FSZ(mma.part3.tmp_shared_storage));
      // printf("      mm_gradQ: %db\n", FSZ(mma.part3.mm_gradQ));
      printf("    part4: %db\n", FSZ(mma.part4));
      printf("     mm_qk_q: %db\n", FSZ(mma.part4.mm_qk_q));
      printf("  epilogue: %db\n", FSZ(epilogue));
      printf(
          "       gradK_epilogue_final: %db\n", FSZ(epilogue.gradK_epilogue_final));
      printf(
          "       gradV_epilogue_final: %db\n", FSZ(epilogue.gradV_epilogue_final));
    }



// ===========================================
#define FIELD(INSIDE_STRUCT, FIELDNAME) \
  CUTLASS_DEVICE decltype(INSIDE_STRUCT.FIELDNAME)& FIELDNAME() {    \
    return INSIDE_STRUCT.FIELDNAME;     \
  }

    FIELD(mma.persistent, di)
    FIELD(mma.persistent, mm_qk_k)
    FIELD(mma.persistent, mm_doivj_v)
    // FIELD(mma.persistent, mm_qk_q)
    // FIELD(mma.persistent, mm_gradQ)
    FIELD(mma.part1, bias)
    FIELD(mma.part1, attn_shared_storage)
    FIELD(mma.part1, zij)
    FIELD(mma.part1, mm_doivj_do)
    FIELD(mma.part2, mm_gradK)
    FIELD(mma.part2, tmp_shared_storage)
    FIELD(mma.part2, tmpT_shared_storage)
    FIELD(mma.part2, gradB_epilogue)
    // FIELD(mma.part3, mm_gradQ)
    FIELD(mma.part4, mm_qk_q)
    FIELD(epilogue, gradK_epilogue_final)
    FIELD(epilogue, gradV_epilogue_final)
  };

  using SharedStorage = SharedStoragePrologue;

  struct OutputFragments {
    typename MatmulGradV::Mma::FragmentC gradV;
    typename MatmulGradK::Mma::FragmentC gradK;

    CUTLASS_DEVICE void clear() {
      gradV.clear();
      gradK.clear();
    }
  };

  static bool __host__ check_supported(Params const& p) {
    CHECK_ALIGNED_PTR(p.query_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.key_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.value_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.output_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.grad_output_ptr, kMinimumAlignment);
    CHECK_ALIGNED_PTR(p.bias_ptr, kMinimumAlignment);
    XFORMERS_CHECK(p.lse_strideH % 8 == 0, "LSE is not correctly aligned");
    XFORMERS_CHECK(p.lse_strideB % 8 == 0, "LSE is not correctly aligned");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.q_strideH % kMinimumAlignment == 0,
        "query is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.k_strideH % kMinimumAlignment == 0,
        "key is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.v_strideH % kMinimumAlignment == 0,
        "value is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_batches <= 1 || p.q_strideB % kMinimumAlignment == 0,
        "query is not correctly aligned (strideB)");
    XFORMERS_CHECK(
        p.num_batches <= 1 || p.k_strideB % kMinimumAlignment == 0,
        "key is not correctly aligned (strideB)");
    XFORMERS_CHECK(
        p.num_batches <= 1 || p.v_strideB % kMinimumAlignment == 0,
        "value is not correctly aligned (strideB)");
    XFORMERS_CHECK(
        p.q_strideM % kMinimumAlignment == 0,
        "query is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.k_strideM % kMinimumAlignment == 0,
        "key is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.v_strideM % kMinimumAlignment == 0,
        "value is not correctly aligned (strideM)");
    if (p.bias_ptr) {
      XFORMERS_CHECK(
          p.num_batches <= 1 || p.bias_strideB % kMinimumAlignment == 0,
          "attn_bias is not correctly aligned (strideB)");
      XFORMERS_CHECK(
          p.num_heads <= 1 || p.bias_strideH % kMinimumAlignment == 0,
          "attn_bias is not correctly aligned (strideH)");
      XFORMERS_CHECK(
          p.bias_strideM % kMinimumAlignment == 0,
          "attn_bias is not correctly aligned (strideM)");
    }
    if (p.grad_bias_ptr) {
      XFORMERS_CHECK(
          p.num_batches <= 1 || p.gB_strideB % kMinimumAlignment == 0,
          "attn_bias.grad is not correctly aligned (strideB)");
      XFORMERS_CHECK(
          p.num_heads <= 1 || p.gB_strideH % kMinimumAlignment == 0,
          "attn_bias.grad is not correctly aligned (strideH)");
      XFORMERS_CHECK(
          p.gB_strideM % kMinimumAlignment == 0,
          "attn_bias.grad is not correctly aligned (strideM)");
    }
    XFORMERS_CHECK(
        !(p.cu_seqlens_q_ptr && p.bias_ptr),
        "CuSeqlen + bias not implemented yet");
    XFORMERS_CHECK(
        p.custom_mask_type < NumCustomMaskTypes,
        "Invalid value for `custom_mask_type`");
    XFORMERS_CHECK(
        p.dropout_prob <= 1.0f && p.dropout_prob >= 0.0f,
        "Invalid value for `dropout_prob`");
    XFORMERS_CHECK(
        kApplyDropout || p.dropout_prob == 0.0f,
        "Set `kApplyDropout`=True to support `dropout_prob > 0`");
    XFORMERS_CHECK(p.head_dim % kOptimalAlignement == 0, "query is not correctly aligned (kOptimalAlignement)");
    XFORMERS_CHECK(p.head_dim > 0, "Invalid value for `head_dim`");
    XFORMERS_CHECK(p.head_dim_value > 0, "Invalid value for `head_dim_value`");
    XFORMERS_CHECK(p.num_queries > 0, "Invalid value for `num_queries`");
    XFORMERS_CHECK(p.num_keys > 0, "Invalid value for `num_keys`");
    XFORMERS_CHECK(p.num_heads > 0, "Invalid value for `num_heads`");
    XFORMERS_CHECK(p.num_batches > 0, "Invalid value for `num_batches`");
    XFORMERS_CHECK(p.head_dim <= kMaxK, "kMaxK: Expected `head_dim < kMaxK`");
    XFORMERS_CHECK(
        p.head_dim_value <= kMaxK, "kMaxK: Expected `head_dim_value < kMaxK`");
    if (kKeysQueriesAlignedToBlockSize) {
      XFORMERS_CHECK(
          p.cu_seqlens_k_ptr == nullptr,
          "This kernel does not support cu_seqlen");
      XFORMERS_CHECK(
          p.cu_seqlens_q_ptr == nullptr,
          "This kernel does not support cu_seqlen");
      XFORMERS_CHECK(
          p.num_queries % kBlockSizeI == 0,
          "kKeysQueriesAlignedToBlockSize condition not respected");
      XFORMERS_CHECK(
          p.num_keys % kBlockSizeJ == 0,
          "kKeysQueriesAlignedToBlockSize condition not respected");
    }
    XFORMERS_CHECK(
        p.num_splits_key > 0, "Invalid `num_splits_key` (expected >0)");
    XFORMERS_CHECK(
        p.num_splits_key <= cutlass::ceil_div(p.num_keys, kBlockSizeJ),
        "Invalid `num_splits_key` (too large)");
    return true;
  }

  static CUTLASS_DEVICE void attention_kernel(Params p) {
#if ENABLE_AIU
    extern __shared__ __align__(128) char smem_buffer[];
#else
    extern __shared__ char smem_buffer[];
#endif
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);

    uint16_t thread_id = threadIdx.x;
    uint8_t warp_id = warp_uniform(thread_id / 32);
    uint8_t lane_id = thread_id % 32;

    int32_t key_start = p.split_key_device() * kBlockSizeJ;

    if (key_start >= p.num_keys) {
      return;
    }

    int32_t query_start = getQueryStart(p, key_start);
    // PrologueQK
    prologueQkNextIteration(
        shared_storage, p, query_start, key_start, warp_id, lane_id);

    prologueDovNextIteration<false, true>(
        shared_storage, p, query_start, key_start, warp_id, lane_id);

    OutputFragments output_frags;

    curandStatePhilox4_32_10_t rng_state_init;


      output_frags.clear();

      CUTLASS_PRAGMA_UNROLL
      for (int32_t query_start_shifted = getQueryStart(p, key_start);
           query_start_shifted < getQueryStartShift(p) + getQueryEnd(p);
           query_start_shifted += kBlockSizeI) {

        warp_id = warp_uniform(warp_id);
        int32_t query_start = query_start_shifted;
        if (query_start >= p.num_queries) {
          query_start = query_start % getQueryEnd(p);
        }

        processBlockIJ<kKeysQueriesAlignedToBlockSize>(
            shared_storage,
            output_frags,
            p,
            query_start,
            key_start,
            rng_state_init,
            warp_id,
            lane_id);
      }
      writeFragsToGmem<kKeysQueriesAlignedToBlockSize>(
          shared_storage, output_frags, p, key_start, warp_id, lane_id);
      __syncthreads();
  }

  template <bool skipBoundsChecks>
  static CUTLASS_DEVICE void processBlockIJ(
      SharedStorage& shared_storage,
      OutputFragments& output_frags,
      Params& p,
      int32_t query_start,
      int32_t key_start,
      const curandStatePhilox4_32_10_t& curand_state_init,
      uint8_t warp_id,
      uint8_t lane_id) {

    const float dropout_scale =
        kApplyDropout ? 1.0f / (1.0f - p.dropout_prob) : 1.0f;

    cutlass::MatrixCoord no_offset{0, 0};
    accum_t scale = p.scale;
    int16_t thread_id = 32 * warp_id + lane_id;

    auto rematerializeThreadIds = [&]() {
      // Prevents `nvcc` from keeping values deduced from
      // `thread_id`, `warp_id`, ... in RF - to reduce register pressure
      warp_id = warp_uniform(thread_id / 32);
      lane_id = thread_id % 32;
      thread_id = 32 * warp_id + lane_id;
    };

    accum_t di_rf = accum_t(0);
    if (thread_id < kBlockSizeI) {
      if (query_start + thread_id < p.num_queries) {
        di_rf = p.delta_ptr[query_start + thread_id];
      }
      shared_storage.di()[thread_id] = di_rf;
    }

    int32_t num_queries_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kM
        : warp_uniform(cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kM, p.num_queries - query_start));

    int32_t num_keys_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kN
        : warp_uniform(cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kN, p.num_keys - key_start));

    auto prologueGradK = [&](int col) {
      typename MatmulGradK::Mma::IteratorB iterator_Q(
          {int32_t(p.q_strideM)},
          p.query_ptr + query_start * p.q_strideM + col,
          {num_queries_in_block, p.head_dim - col},
          thread_id,
          no_offset);
      MatmulGradK::Mma::prologue(
          shared_storage.mm_gradK(),
          iterator_Q,
          thread_id,
          num_queries_in_block);
    };

    typename MatmulQK::Mma::FragmentC accum_qk;
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulQK
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulQK::Mma;

      cutlass::gemm::GemmCoord problem_size(
          num_queries_in_block,
          num_keys_in_block,
          p.head_dim // k
      );

      // q_j
      typename Mma::IteratorA iterator_A(
          {int32_t(p.q_strideM)},
          p.query_ptr + query_start * p.q_strideM,
          {problem_size.m(), problem_size.k()},
          thread_id,
          no_offset);

      // k_j.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.k_strideM)},
          p.key_ptr + key_start * p.k_strideM,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      Mma mma(
          shared_storage.mm_qk_q(),
          shared_storage.mm_qk_k(),
          thread_id,
          warp_id,
          lane_id);

      // typename Mma::FragmentC accum;

      accum_qk.clear();

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma.set_zero_outside_bounds(!skipBoundsChecks);
      mma(gemm_k_iterations, accum_qk, iterator_A, iterator_B, accum_qk, true);
      accum_qk = cutlass::multiplies<typename Mma::FragmentC>()(scale, accum_qk);

       __syncthreads();

      // preload current dO.
      prologueDovNextIteration<true, false>(
        shared_storage, p, query_start, key_start, warp_id, lane_id);
      // Epilogue: add LSE + exp and store that to our shared memory buffer
      // shmem <- (matmul_result -
      // logsumexp[i_start:i_end].unsqueeze(1)).exp()
      int warp_idx_mn_0 =
          warp_id % (Mma::WarpCount::kM * Mma::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % Mma::WarpCount::kM,
          warp_idx_mn_0 / Mma::WarpCount::kM};

      // apply bias if applicable
      if (p.bias_ptr != nullptr) {
        // load bias tile Bij into shared memory
        typename MatmulQK::BiasLoader::GmemTileIterator bias_iter(
            {cutlass::layout::RowMajor(p.bias_strideM)},
            p.bias_ptr + query_start * p.bias_strideM + key_start,
            {num_queries_in_block, num_keys_in_block},
            thread_id);
        cutlass::TensorRef<scalar_t, cutlass::layout::RowMajor> bias_tensor_ref(
            shared_storage.bias().data(),
            cutlass::layout::RowMajor(MatmulQK::ThreadblockShape::kN));
        typename MatmulQK::BiasLoader::SmemTileIterator smem_tile_iter(
            bias_tensor_ref, thread_id);
        MatmulQK::BiasLoader::load(bias_iter, smem_tile_iter);

        // Pij += Bij, where Pij is in register fragment and Bij is in shmem
        auto lane_offset = MatmulQK::AccumLambdaIterator::get_lane_offset(
            lane_id, warp_id, output_tile_coords);
        MatmulQK::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
              if (accum_m < num_queries_in_block && accum_n < num_keys_in_block) {
                accum_qk[idx] += accum_t(bias_tensor_ref.at({accum_m, accum_n}));
              }
            },
            [&](int accum_n) {});
      }
      __syncthreads();

      auto lane_offset = MatmulQK::AccumLambdaIterator::get_lane_offset(
          lane_id, warp_id, output_tile_coords);
      // Apply mask
      if (p.custom_mask_type == CausalFromTopLeft ||
          p.custom_mask_type == CausalFromBottomRight) {
        int shift = query_start - key_start;
        if (p.custom_mask_type == CausalFromBottomRight) {
          shift += p.num_keys - p.num_queries;
        }

        // current_key = key_start + accum_m
        // current_query = query_start + accum_n
        // mask if: `current_key > current_query`
        MatmulQK::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_n) {},
            [&](int accum_m, int accum_n, int idx) {
              if (accum_n > accum_m + shift) {
                accum_qk[idx] =
                    -cutlass::platform::numeric_limits<accum_t>::infinity();
              }
            },
            [&](int accum_n) {});
      }

      MatmulQK::B2bGemm::accumApplyLSEToSmem(
          shared_storage.attn_shared_storage(),
          accum_qk,
          p.logsumexp_ptr + query_start,
          problem_size.m(),
          thread_id,
          warp_id,
          lane_id,
          output_tile_coords);
      __syncthreads();
#if 0
      auto accum_ref_attnT = shared_storage.attn_shared_storage().accum_ref();
      PRINT_TENSOR4x4_T0_L0_START("attn_T", accum_ref_attnT, 0, 0);
#endif

#ifdef HAS_PYTORCH
      if constexpr(kApplyDropout) {
        auto p_dropout_in_uint8_t = uint8_t(std::floor((1 - p.dropout_prob) * 255.0));
        int warp_idx_m = warp_idx_mn_0 % Mma::WarpCount::kM;
        int warp_idx_n = warp_idx_mn_0 / Mma::WarpCount::kM;

        // Need col to be multiples of 16, since we're doing dropout with block of:
        //   PPU: 32 x 16, A100: 16 x 32
        static constexpr int DROPOUT_BLOCK_M = 32;
        static constexpr int DROPOUT_BLOCK_N = 16;
        static constexpr int DROPOUT_ROW_STRIDE = 1;
        int block_row_idx = (query_start + warp_idx_m * MatmulQK::WarpShape::kM) / DROPOUT_BLOCK_M;
        int block_col_idx = (key_start + warp_idx_n * MatmulQK::WarpShape::kN) / DROPOUT_BLOCK_N;

        uint64_t seed, offset;
        auto seeds = at::cuda::philox::unpack(p.rng_engine_inputs);
        if (p.rng_state) {
          seed = p.rng_state[0];
          offset = p.rng_state[1];
        } else {
          seed = std::get<0>(seeds);
          offset = std::get<1>(seeds);
        }
        offset += p.dropout_batch_head_rng_offset * 32 + lane_id;
        // printf("block_row_idx = %d, query_start = %d, warp_idx_m = %d\n", block_row_idx, query_start, warp_idx_m);
        // printf("block_col_idx = %d, key_start = %d, warp_idx_n = %d\n", block_col_idx, key_start, warp_idx_n);
        fmha::apply_dropout<MatmulQK::WarpShape::kM, MatmulQK::WarpShape::kN, fmha::DROPOUT_MODE_SET_BOOLEAN>(
            accum_qk, p_dropout_in_uint8_t,
            seed, offset, block_row_idx, block_col_idx, DROPOUT_ROW_STRIDE, dropout_scale);

        // store accum_copy to tsm
        auto zij = shared_storage.zij().accum_ref();

        MatmulQK::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
              int col = key_start + accum_n;
                zij.at({accum_m, accum_n}) = scalar_t(accum_qk[idx]);
            },
            [&](int accum_m) {});
        __syncthreads();
      }
#endif
    }
    rematerializeThreadIds();

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradV matmul
    //
    // grad_v[j_start:j_end] += attn_T @ do_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    constexpr bool kSingleIterationGradV =
        kMaxK <= MatmulGradV::ThreadblockShape::kN;
    for (int col = 0; col < (kSingleIterationGradV ? 1 : p.head_dim_value);
         col += MatmulGradV::ThreadblockShape::kN) {

      using Mma = typename MatmulGradV::Mma;
      using AccumTileGmem = typename MatmulGradQ::AccumTileGmem;

      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block, p.head_dim_value - col, num_queries_in_block);

      typename Mma::IteratorB iterator_B(
          {int32_t(p.gO_strideM)},
          p.grad_output_ptr + query_start * p.gO_strideM + col,
          {num_queries_in_block, p.head_dim_value - col},
          thread_id,
          no_offset);

      // if dropout: dVj += (Pij.T * Zij) @ dOi
      // otherwise:  dVj += Pij.T @ dOi
      Mma mma(
          // operand A: Pij.T
          shared_storage.attn_shared_storage().accum_ref(),
          // operand A_scale Zij.T:
          // if we're using dropout, operand A is Pij_dropped.T = Pij.T * Zij.T
          // which is computed on the fly as fragments of Pij.T are loaded in
          shared_storage.zij().accum_ref(),
          // operand B: dOi - which was loaded into shared memory previously
          // when we computed dVj
          // shared_storage.mm_gradV().operand_B_ref(),
          shared_storage.mm_doivj_do().ref(),
          thread_id,
          warp_id,
          lane_id);

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      // __syncthreads();
      mma(gemm_k_iterations,
          output_frags.gradV,
          iterator_B,
          output_frags.gradV,
          true);
      __syncthreads();

      // PRINT_ACCUM8_T0_L0("gradV", output_frags.gradV);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // MatmulDOIVJ
    /////////////////////////////////////////////////////////////////////////////////////////////////
    {
      using Mma = typename MatmulDOIVJ::Mma;
      // do_i
      typename Mma::IteratorA iterator_A(
          {int32_t(p.gO_strideM)},
          p.grad_output_ptr + query_start * p.gO_strideM,
          {num_queries_in_block, p.head_dim_value},
          thread_id,
          no_offset);

      // v_j.transpose(-2, -1)
      typename Mma::IteratorB iterator_B(
          {int32_t(p.v_strideM)},
          p.value_ptr + key_start * p.v_strideM,
          {p.head_dim_value, num_keys_in_block},
          thread_id,
          no_offset);
      Mma mma(shared_storage.mm_doivj_do(),
              shared_storage.mm_doivj_v(),
              thread_id,
              warp_id,
              lane_id);
      mma.set_zero_outside_bounds(!skipBoundsChecks);

      typename Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (p.head_dim_value + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum, true);
      __syncthreads();

      prologueGradK(0);
      // prologueGradQ(0);

      int warp_idx_mn_0 =
          warp_id % (Mma::WarpCount::kM * Mma::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % Mma::WarpCount::kM,
          warp_idx_mn_0 / Mma::WarpCount::kM};
      // TODO: This must be terribly inefficient. There must be a better way
      // tmp [RF] <- (accum [RF] - Di [smem] ) * attn_T [smem]
      // attn_shared_storage  [smem] <- tmp.T
      // tmp_shared_storage [smem] <- tmp
      {
        using LambdaIterator = typename MatmulDOIVJ::AccumLambdaIterator;
        auto lane_offset = LambdaIterator::get_lane_offset(
            lane_id, warp_id, output_tile_coords);

        // if dropout was used, compute dPij = dPij_dropped * Zij
        if (kApplyDropout) {
          LambdaIterator::iterateRows(
              lane_offset,
              [&](int accum_m) {},
              [&](int accum_m, int accum_n, int idx) {
                accum[idx] *= accum_qk[idx];
              },
              [&](int accum_m) {});
        }

        // auto attn_T = shared_storage.attn_shared_storage().accum_ref();
        auto attn = shared_storage.attn_shared_storage().accum_ref();
#if 0
        PRINT_B0_T0("doivj_dropped");
        print_warp_accum<LambdaIterator>(accum, lane_offset, 4, 4);
        PRINT_TENSOR4x4_T0_L0("attn_T", attn_T)
#endif
        accum_t current_di;
        // dSij = (dPij - Di) * Pij
        LambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) { current_di = shared_storage.di()[accum_m]; },
            [&](int accum_m, int accum_n, int idx) {
              // TODO: Otherwise we can get nans as we
              // might have infs here (only seen on f16 tho)
              accum[idx] = skipBoundsChecks || (accum_m < num_queries_in_block && accum_n < num_keys_in_block) ?
                            (accum[idx] - current_di) * float(attn.at({accum_m, accum_n})) : 0;
            },
            [&](int accum_m) {});

        // store bias gradient tile dBij to global memory,
        // where dBij = dSij = Pij * (dPij - Di)
        if (p.grad_bias_ptr != nullptr) {
          typename MatmulDOIVJ::BiasGradEpilogue::OutputTileIterator
              output_iter(
                  typename MatmulDOIVJ::BiasGradEpilogue::OutputTileIterator::
                      Params{p.gB_strideM},
                  // grad_bias_ptr is offset to point at beginning of
                  // matrix of shape (queries, keys) for a given
                  // (batch_id, head_id) the pointer arithmetic here produces
                  // a pointer to the start of the current tile within that
                  // matrix
                  p.grad_bias_ptr + query_start * p.gB_strideM + key_start,
                  {num_queries_in_block, num_keys_in_block},
                  thread_id);

          // no-op epilogue operator - just casting and storing contents of
          // accum to global memory
          typename MatmulDOIVJ::BiasGradEpilogue::OutputOp output_op({1, 1});
          typename MatmulDOIVJ::BiasGradEpilogue epilogue(
              shared_storage.gradB_epilogue(), thread_id, warp_id, lane_id);
          epilogue(output_op, output_iter, accum, output_iter);
        }

        accum = accum * scale;

#if 0
        PRINT_B0_T0("(doivj - di) * attn * scale");
        print_warp_accum<LambdaIterator>(accum, lane_offset, 4, 4);
#endif

        __syncthreads();
        if (!MatmulGradK::DefaultMmaFromSmem::kIsTransposedA) {
          auto tmpT = shared_storage.tmpT_shared_storage().accum_ref();
          // attn <- attn_T.T
          LambdaIterator::iterateRows(
              lane_offset,
              [&](int accum_m) {},
              [&](int accum_m, int accum_n, int idx) {
                tmpT.at({accum_n, accum_m}) = scalar_t(accum[idx]);
              },
              [&](int accum_m) {});
        }
      }

      MatmulDOIVJ::B2bGemm::accumToSmem(
          shared_storage.tmp_shared_storage(),
          accum,
          lane_id,
          output_tile_coords);
      __syncthreads();
    }

    // Force `nvcc` to recompute values that depend on the variables just below
    // to use less RF and prevent some spilling
    p.head_dim = warp_uniform(p.head_dim);
    p.k_strideM = warp_uniform(p.k_strideM);
    rematerializeThreadIds();

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradQ matmul
    //
    // grad_q[i_start:i_end] += tmp @ k_j
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Skip the loop & associated branches if we know at compile time the number
    // of iterations
    constexpr bool kSingleIterationGradQ =
        kMaxK <= MatmulGradQ::ThreadblockShape::kN;
    for (int col = 0; col < (kSingleIterationGradQ ? 1 : p.head_dim);
         col += MatmulGradQ::ThreadblockShape::kN) {
      using Mma = typename MatmulGradQ::Mma;
      using AccumTileGmem = typename MatmulGradQ::AccumTileGmem;
      cutlass::gemm::GemmCoord problem_size(
          num_queries_in_block,
          false ? MatmulGradQ::ThreadblockShape::kN : p.head_dim - col,
          num_keys_in_block);

      // k_j
      typename Mma::IteratorB iterator_B(
          {int32_t(p.k_strideM)},
          p.key_ptr + key_start * p.k_strideM + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      // PRINT_TENSOR4x4_T0_L0_START("tmp_shared_storage", shared_storage.tmp_shared_storage().accum_ref(), 0, 0);
      // PRINT_TENSOR4x4_T0_L0_START("mm_gradQ", shared_storage.mm_gradQ().operand_B_ref(), 0, 0);

      Mma mma(
          // operand A: dSij
          shared_storage.tmp_shared_storage().accum_ref(),
          // operand B: Kj
          shared_storage.mm_qk_k().ref(),
          // shared_storage.mm_gradQ().operand_B_ref(),
          thread_id,
          warp_id,
          lane_id);

      typename Mma::FragmentC accum;

      int col_id = col / MatmulGradQ::ThreadblockShape::kN;
      int num_cols = kSingleIterationGradQ
          ? 1
          : ceil_div(p.head_dim, MatmulGradQ::ThreadblockShape::kN);
      int storage_id = (col_id + query_start / kBlockSizeI * num_cols);

      // cutlass::Semaphore semaphore(&p.workspace_gq[storage_id].lock, thread_id);
      // if (p.num_splits_key_device() > 1) {
      //   semaphore.fetch();
      //   semaphore.wait(p.split_key_device());

      //   // Make sure we can see other block's output
      //   __threadfence();
      // }

      AccumTileGmem gmem_tile{&p.workspace_gq[storage_id].buffer[0]};
      // gmem_tile.load(accum, thread_id);

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      // __syncthreads();

      mma(gemm_k_iterations, accum, iterator_B, accum, true);
      // __syncthreads();

      // PRINT_ACCUM8_T0_L0("gradQ", accum);
      
      gmem_tile.storeAtomicAdd(accum, thread_id);

      // gmem_tile.store(accum, thread_id);

      // if (p.num_splits_key_device() > 1) {
      //   int lock = 0;
      //   if (p.split_key_device() == p.num_splits_key_device() + 1) {

      //     // The final threadblock resets the semaphore for subsequent grids.
      //     lock = 0;
      //   }
      //   else {
      //     // Otherwise, the semaphore is incremented
      //     lock = p.split_key_device() + 1;
      //   }

      //   __threadfence();
      //   semaphore.release(lock);
      // }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // GradK matmul
    //
    // grad_k[i_start:i_end] += tmp.transpose(-2, -1) @ q_i
    /////////////////////////////////////////////////////////////////////////////////////////////////
    rematerializeThreadIds();

    constexpr bool kSingleIterationGradK =
        kMaxK <= MatmulGradK::ThreadblockShape::kN;
    for (int col = 0; col < (kSingleIterationGradK ? 1 : p.head_dim);
         col += MatmulGradK::ThreadblockShape::kN) {
      using Mma = typename MatmulGradK::Mma;
      using AccumTileGmem = typename MatmulGradQ::AccumTileGmem;

      cutlass::gemm::GemmCoord problem_size(
          num_keys_in_block,
          false ? MatmulGradK::ThreadblockShape::kN : p.head_dim - col,
          num_queries_in_block);

      // q_i
      typename Mma::IteratorB iterator_B(
          {int32_t(p.q_strideM)},
          p.query_ptr + query_start * p.q_strideM + col,
          {problem_size.k(), problem_size.n()},
          thread_id,
          no_offset);

      auto getTmp = [&](int) { return &shared_storage.tmp_shared_storage(); };
      auto getTmpT = [&](int) { return &shared_storage.tmpT_shared_storage(); };
      // this is basically:
      // opA = kIsTransposedA ? getTmp() : getTmpT();
      bool constexpr kIsTransposedA =
          MatmulGradK::DefaultMmaFromSmem::kIsTransposedA;

      auto& opA = *call_conditional<
          kIsTransposedA,
          decltype(getTmp),
          decltype(getTmpT)>::apply(getTmp, getTmpT, 0);
      Mma mma(
          // operand A: dSij.T
          opA.accum_ref(),
          // shared_storage.tmp_shared_storage().accum_ref(),
          // operand B: Qi
          shared_storage.mm_gradK().operand_B_ref(),
          // shared_storage.mm_qk_q().ref(),
          thread_id,
          warp_id,
          lane_id);

      auto gemm_k_iterations =
          (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      // __syncthreads();

      mma(gemm_k_iterations,
          output_frags.gradK,
          iterator_B,
          output_frags.gradK,
          true);
      __syncthreads();

      //preload next qeury
      int32_t next_query, next_key;
      incrIteration(p, query_start, key_start, next_query, next_key);
      prologueQkNextIteration<true, false>(
                shared_storage, p, next_query, next_key, warp_id, lane_id);
    }
  }

  static CUTLASS_DEVICE int32_t getQueryStartShift(Params const& p) {
    if (p.custom_mask_type == NoCustomMask && p.num_splits_key_device() > 1) {
      return (p.split_key_device() * kBlockSizeI) % getQueryEnd(p);
    }
    return 0;
  }

  // Iteration order logic
  static CUTLASS_DEVICE int32_t
  getQueryStart(Params const& p, int32_t key_start) {
    return getSmallestQueryForKey(p, key_start) + getQueryStartShift(p);
  };

  static CUTLASS_DEVICE int32_t getQueryEnd(Params const& p) {
    return align_up(p.num_queries, kBlockSizeI);
  };

  static CUTLASS_DEVICE int32_t
  getSmallestQueryForKey(Params const& p, int32_t key_start) {
    if (p.custom_mask_type == CausalFromTopLeft) {
      return (key_start / kBlockSizeI) * kBlockSizeI;
    } else if (p.custom_mask_type == CausalFromBottomRight) {
      int first_query =
          cutlass::fast_max(0, key_start - p.num_keys + p.num_queries);
      return (first_query / kBlockSizeI) * kBlockSizeI;
    }
    return 0;
  };

  // Returns how many kernel blocks will write to a given block in `grad_query`
  // This is usually equal to the number of key splits, but can be different
  // for instance in the causal case, or varying seqlen
  static CUTLASS_DEVICE int32_t
  getNumParallelBlocksForQuery(Params const& p, int32_t query_start) {
    int16_t num_key_blocks = ceil_div(p.num_keys, kBlockSizeJ);
    if (p.custom_mask_type == CausalFromTopLeft) {
      int32_t last_key_for_block = query_start + kBlockSizeI - 1;
      last_key_for_block = cutlass::fast_min(last_key_for_block, p.num_keys);
      num_key_blocks = ceil_div(last_key_for_block, kBlockSizeJ);
    } else if (p.custom_mask_type == CausalFromBottomRight) {
      int32_t last_key_for_block =
          query_start + (kBlockSizeI - 1) + (1 + p.num_keys - p.num_queries);
      last_key_for_block = cutlass::fast_min(last_key_for_block, p.num_keys);
      num_key_blocks = ceil_div(last_key_for_block, kBlockSizeJ);
    }
    return cutlass::fast_min(p.num_splits_key_device(), num_key_blocks);
  };

  // Returns the next block to process
  static CUTLASS_DEVICE void incrIteration(
      Params const& p,
      int32_t query_start,
      int32_t key_start,
      int32_t& next_query,
      int32_t& next_key) {
    next_query = query_start + kBlockSizeI;
    next_key = key_start;
    auto query_shift = getQueryStartShift(p);
    // Wrap around
    if (query_shift) {
      if (next_query >= p.num_queries) {
        next_query = getSmallestQueryForKey(p, key_start);
        return;
      } else if (query_start < query_shift && query_shift <= next_query) {
        // jump to next key
      } else {
        return;
      }
    } else {
      if (next_query < p.num_queries) {
        return;
      }
      // jump to next key
    }
    // Next key
    next_key = key_start + p.num_splits_key_device() * kBlockSizeJ;
    next_query = getQueryStart(p, next_key);
  }

  template <bool kLoadA = true, bool kLoadB = true>
  static CUTLASS_DEVICE void prologueQkNextIteration(
      SharedStorage& shared_storage,
      Params const& p,
      int32_t query_start,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    if (query_start >= p.num_queries || key_start >= p.num_keys) {
      return;
    }

    int thread_id = 32 * warp_id + lane_id;
    typename MatmulQK::Mma::IteratorA iterator_A(
        {int32_t(p.q_strideM)},
        p.query_ptr + query_start * p.q_strideM,
        {p.num_queries - query_start, p.head_dim},
        thread_id,
        cutlass::MatrixCoord{0, 0});

    typename MatmulQK::Mma::IteratorB iterator_B(
        {int32_t(p.k_strideM)},
        p.key_ptr + key_start * p.k_strideM,
        {p.head_dim, p.num_keys - key_start},
        thread_id,
        cutlass::MatrixCoord{0, 0});


    MatmulQK::Mma::template prologue<kLoadA, kLoadB>(
        shared_storage.mm_qk_q(),
        shared_storage.mm_qk_k(),
        iterator_A,
        iterator_B,
        thread_id,
        p.head_dim);
  }

template <bool kLoadA = true, bool kLoadB = true>
  static CUTLASS_DEVICE void prologueDovNextIteration(
      SharedStorage& shared_storage,
      Params const& p,
      int32_t query_start,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    if (query_start >= p.num_queries || key_start >= p.num_keys) {
      return;
    }

    int thread_id = 32 * warp_id + lane_id;
    typename MatmulDOIVJ::Mma::IteratorA iterator_A(
        {int32_t(p.gO_strideM)},
        p.grad_output_ptr + query_start * p.gO_strideM,
        {p.num_queries - query_start, p.head_dim_value},
        thread_id,
        cutlass::MatrixCoord{0, 0});
    typename MatmulDOIVJ::Mma::IteratorB iterator_B(
        {int32_t(p.v_strideM)},
        p.value_ptr + key_start * p.v_strideM,
        {p.head_dim_value, p.num_keys - key_start},
        thread_id,
        cutlass::MatrixCoord{0, 0});
    MatmulDOIVJ::Mma::template prologue<kLoadA, kLoadB>(
        shared_storage.mm_doivj_do(),
        shared_storage.mm_doivj_v(),
        iterator_A,
        iterator_B,
        thread_id,
        p.head_dim_value);
  }

  template <bool skipBoundsChecks>
  static CUTLASS_DEVICE void writeFragsToGmem(
      SharedStorage& shared_storage,
      OutputFragments& output_frags,
      Params const& p,
      int32_t key_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    uint16_t thread_id = 32 * warp_id + lane_id;
    int32_t num_keys_in_block = skipBoundsChecks
        ? MatmulQK::Mma::Shape::kN
        : cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kN, p.num_keys - key_start);

    typename MatmulGradV::OutputTileIterator outputV_it(
        typename MatmulGradV::OutputTileIterator::Params{p.gV_strideM()},
        p.grad_value_ptr + key_start * p.gV_strideM(),
        {num_keys_in_block, p.head_dim_value},
        thread_id);
    accumulateInGmem<MatmulGradV>(
        shared_storage.gradV_epilogue_final(),
        output_frags.gradV,
        outputV_it,
        true,
        warp_id,
        lane_id);

    typename MatmulGradK::OutputTileIterator outputK_it(
        typename MatmulGradK::OutputTileIterator::Params{p.gK_strideM()},
        p.grad_key_ptr + key_start * p.gK_strideM(),
        {num_keys_in_block,
         false ? MatmulGradK::ThreadblockShape::kN : p.head_dim},
        thread_id);
    accumulateInGmem<MatmulGradK>(
        shared_storage.gradK_epilogue_final(),
        output_frags.gradK,
        outputK_it,
        true,
        warp_id,
        lane_id);
  }


  template <typename MatmulT>
  static CUTLASS_DEVICE void accumulateInGmem(
      typename MatmulT::DefaultEpilogue::SharedStorage& epilogue_smem,
      typename MatmulT::Mma::FragmentC const& accum,
      typename MatmulT::OutputTileIterator output_it,
      bool first,
      uint8_t warp_id,
      uint8_t lane_id) {
    using DefaultEpilogue = typename MatmulT::DefaultEpilogue;
    using DefaultOutputOp = typename MatmulT::DefaultOutputOp;
    using Mma = typename MatmulT::Mma;
    int thread_id = 32 * warp_id + lane_id;

    DISPATCH_BOOL(
        first, kIsFirst, ([&]() {
          static constexpr auto ScaleType = kIsFirst
              ? cutlass::epilogue::thread::ScaleType::Nothing
              : cutlass::epilogue::thread::ScaleType::NoBetaScaling;
          using EpilogueOutputOp =
              typename cutlass::epilogue::thread::LinearCombination<
                  typename DefaultOutputOp::ElementOutput,
                  DefaultOutputOp::kCount,
                  typename DefaultOutputOp::ElementAccumulator,
                  typename DefaultOutputOp::ElementCompute,
                  ScaleType>;
          using Epilogue =
              typename cutlass::epilogue::threadblock::EpiloguePipelined<
                  typename DefaultEpilogue::Shape,
                  typename Mma::Operator,
                  DefaultEpilogue::kPartitionsK,
                  typename MatmulT::OutputTileIterator,
                  typename DefaultEpilogue::AccumulatorFragmentIterator,
                  typename DefaultEpilogue::WarpTileIterator,
                  typename DefaultEpilogue::SharedLoadIterator,
                  EpilogueOutputOp,
                  typename DefaultEpilogue::Padding,
                  DefaultEpilogue::kFragmentsPerIteration,
                  true // IterationsUnroll
                  >;

          EpilogueOutputOp rescale({1, 0});
          Epilogue epilogue(epilogue_smem, thread_id, warp_id, lane_id);
          epilogue(rescale, output_it, accum);
        }));
  }

  static CUTLASS_DEVICE void convert_kernel(Params p) {
    extern __shared__ char smem_buffer[];

    using gradQ_epilogue_lastIter = typename MatmulGradQ::DefaultEpilogue::SharedStorage;
    gradQ_epilogue_lastIter& shared_storage = *((gradQ_epilogue_lastIter*)smem_buffer);
    
    uint16_t thread_id = threadIdx.x;
    uint8_t  warp_id = warp_uniform(thread_id / 32);
    uint8_t  lane_id = thread_id % 32;
    uint16_t block_id = blockIdx.x;

    int32_t query_start = block_id * MatmulQK::Mma::Shape::kM;

    int32_t num_queries_in_block = kKeysQueriesAlignedToBlockSize
        ? MatmulQK::Mma::Shape::kM
        : warp_uniform(cutlass::fast_min(
              (int32_t)MatmulQK::Mma::Shape::kM, p.num_queries - query_start));

    constexpr bool kSingleIterationGradQ =
      kMaxK <= MatmulGradQ::ThreadblockShape::kN;

    for (int col = 0; col < (kSingleIterationGradQ ? 1 : p.head_dim);
      col += MatmulGradQ::ThreadblockShape::kN) {

      int col_id = col / MatmulGradQ::ThreadblockShape::kN;
      int num_cols = kSingleIterationGradQ
        ? 1
        : ceil_div(p.head_dim, MatmulGradQ::ThreadblockShape::kN);
      int storage_id = (col_id + query_start / kBlockSizeI * num_cols);

      typename MatmulGradQ::Mma::FragmentC accum;
      typename MatmulGradQ::AccumTileGmem gmem_tile{&p.workspace_gq[storage_id].buffer[0]};
      gmem_tile.load(accum, thread_id);

      typename MatmulGradQ::OutputTileIterator output_it(
          typename MatmulGradQ::OutputTileIterator::Params{p.gQ_strideM()},
          p.grad_query_ptr + query_start * p.gQ_strideM() + col,
          {num_queries_in_block, p.head_dim},
          thread_id);

      // PRINT_ACCUM8_T0_L0("gradQ-convert", accum);

      constexpr bool storage_contains_zeros = true;
      accumulateInGmem<MatmulGradQ>(
          shared_storage,
          accum,
          output_it,
          storage_contains_zeros,
          warp_id,
          lane_id);
    }
  }

  template <int kElementsPerAccess>
  static CUTLASS_DEVICE void computeDelta(
      Params const& p,
      int32_t query_start,
      uint8_t warp_id,
      uint8_t lane_id) {
    // Each thread computes one value for Delta
    // Depending on warp configuration, we might have multiple
    // threads of the same warp working on the same row
    using AccessType = cutlass::Array<scalar_t, kElementsPerAccess>;
    static_assert(kNumThreads >= kBlockSizeI, "");
    static constexpr int kNumThreadsPerLine = kNumThreads / kBlockSizeI;
    int16_t thread_id = 32 * warp_id + lane_id;

    int16_t laneFirstCol = kElementsPerAccess * (lane_id % kNumThreadsPerLine);
    int16_t laneRow = thread_id / kNumThreadsPerLine;
    bool rowPred = (query_start + laneRow) < p.num_queries;
    bool pred = rowPred;

    // on windows, previous syntax __restrict__ AccessType*
    // resulted in error: "restrict" is not allowed
    const AccessType* __restrict__ grad_output_ptr =
        reinterpret_cast<const AccessType*>(
            p.grad_output_ptr + (query_start + laneRow) * p.gO_strideM +
            laneFirstCol);
    const AccessType* __restrict__ output_ptr =
        reinterpret_cast<const AccessType*>(
            p.output_ptr + (query_start + laneRow) * p.o_strideM() +
            laneFirstCol);

    static constexpr int64_t kMaxIters =
        kMaxK / (kElementsPerAccess * kNumThreadsPerLine);
    constexpr int kPipelineStages = 2;
    accum_t delta_value = accum_t(0);
    using GlobalLoad =
        cutlass::arch::global_load<AccessType, sizeof(AccessType)>;
    AccessType frag_grad_output[kPipelineStages];
    AccessType frag_output[kPipelineStages];

    auto loadAndIncrement = [&](int ld_pos, bool is_valid) {
      frag_grad_output[ld_pos].clear();
      frag_output[ld_pos].clear();
      GlobalLoad(frag_grad_output[ld_pos], grad_output_ptr, is_valid);
      GlobalLoad(frag_output[ld_pos], output_ptr, is_valid);
      grad_output_ptr += kNumThreadsPerLine;
      output_ptr += kNumThreadsPerLine;
    };

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < kPipelineStages - 1; ++iter) {
      int ld_pos = iter % kPipelineStages;
      pred = pred &&
          (laneFirstCol + iter * kElementsPerAccess * kNumThreadsPerLine) <
              p.head_dim_value;
      loadAndIncrement(ld_pos, pred);
    }
    auto columnIteration = [&](int iter) {
      // Load for next iter
      int ld_pos = (iter + kPipelineStages - 1) % kPipelineStages;
      pred = pred &&
          (laneFirstCol +
           (iter + kPipelineStages - 1) * kElementsPerAccess *
               kNumThreadsPerLine) < p.head_dim_value;
      loadAndIncrement(ld_pos, pred);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < AccessType::kElements; ++i) {
        delta_value += accum_t(frag_output[iter % kPipelineStages][i]) *
            accum_t(frag_grad_output[iter % kPipelineStages][i]);
      }
    };

    // If we have a small lower-bound for K, we can unroll the loop
    if (kMaxK <= 256) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < kMaxIters; ++iter) {
        columnIteration(iter);
      }
    } else {
      int num_iters =
          ceil_div(p.head_dim_value, kElementsPerAccess * kNumThreadsPerLine) *
          (kElementsPerAccess * kNumThreadsPerLine);
      for (int iter = 0; iter < num_iters; ++iter) {
        columnIteration(iter);
      }
    }

    // Reduce between workers
    // static_assert(
    //     kNumThreadsPerLine == 1 || kNumThreadsPerLine == 2 ||
    //         kNumThreadsPerLine == 4,
    //     "");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < kNumThreadsPerLine; i *= 2) {
      delta_value = delta_value + __shfl_xor_sync(0xffffffff, delta_value, i);
    }

    // Store in gmem
    if (rowPred) {
      p.delta_ptr[query_start + laneRow] = delta_value;
    }
  }

  static CUTLASS_DEVICE void dov_kernel(Params p) {
    uint16_t thread_id = threadIdx.x;
    uint8_t  warp_id = warp_uniform(thread_id / 32);
    uint8_t  lane_id = thread_id % 32;
    int64_t  query_id = blockIdx.x;
    int64_t  batch_id = blockIdx.y;
    int32_t  head_id = blockIdx.z;

    // Computes (dO*out).sum(-1) and writes it to `p.delta_ptr`
    constexpr int kOptimalElements =
        128 / cutlass::sizeof_bits<scalar_t>::value;
    computeDelta<kOptimalElements>(p, query_id * kBlockSizeI, warp_id, lane_id);

    // zero for workspace
    if (p.workspace_gq) {
      constexpr int AccessByte = 32;
      constexpr int kElements = AccessByte / sizeof(output_accum_t);
      using AccessType = cutlass::Array<output_accum_t, kElements>;
      AccessType frags;
      frags.clear();

      int64_t query_id = blockIdx.x;
      output_accum_t *workspace_per_query =
          reinterpret_cast<output_accum_t*>(p.workspace_gq)
            + (query_id * p.workspace_elements_cols());
      workspace_per_query = warp_uniform(workspace_per_query);

      CUTLASS_PRAGMA_UNROLL
      for (int loop = 0; 
            loop < cutlass::ceil_div(p.workspace_elements_cols() / kElements, kNumThreads);
            loop++) {
        size_t offset = loop * kNumThreads * kElements + thread_id * kElements;
        cutlass::arch::global_store<AccessType, AccessByte>(
            frags,
            reinterpret_cast<void*>(workspace_per_query + offset),offset <= p.workspace_elements_cols());
      }
    }
    __syncthreads();
  }
};

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_backward_batched_impl_v2(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::attention_kernel(p);
}

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_backward_convert_dq(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::convert_kernel(p);
}

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_backward_dov(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::dov_kernel(p);
}


