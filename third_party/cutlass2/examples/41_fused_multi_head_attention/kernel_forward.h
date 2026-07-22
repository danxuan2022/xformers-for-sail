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
#include "kernel_forward_base.h"

template <
    // The datatype of Q/K/V
    typename scalar_t_,
    // Architecture we are targeting (eg `cutlass::arch::Sm80`)
    typename ArchTag,
    // If Q/K/V are correctly aligned in memory and we can run a fast kernel
    bool isAligned_,
    int kQueriesPerBlock,
    int kKeysPerBlock_,
    bool kSingleValueIteration_, // = `value.shape[-1] <= kKeysPerBlock`
    // This is quite slower on V100 for some reason
    // Set to false if you know at compile-time you will never need dropout
    bool kSupportsDropout_ = false,
    bool kSupportsBias_ = false,
    int kMaxK_ = (int)cutlass::platform::numeric_limits<uint32_t>::max(),
    bool kReturnSoftmax = false>
struct AttentionKernel {
  enum CustomMaskType {
    NoCustomMask = 0,
    CausalFromTopLeft = 1,
    CausalFromBottomRight = 2,
    NumCustomMaskTypes,
  };

  using scalar_t = scalar_t_;
  using accum_t = float;
  using lse_scalar_t = float;
  using output_t = scalar_t;
  // Accumulator between 2 iterations
  // Using `accum_t` improves perf on f16 at the cost of
  // numerical errors
  using output_accum_t = accum_t;
  static constexpr int kMaxK = kMaxK_;
  static constexpr bool kSupportsDropout = kSupportsDropout_;
  static constexpr bool kSupportsBias = kSupportsBias_;
  static constexpr int kKeysPerBlock = kKeysPerBlock_;
  static constexpr bool kIsAligned = isAligned_;
  static constexpr bool kSingleValueIteration = kSingleValueIteration_;
  static constexpr int32_t kAlignLSE = 32; // block size of backward
  static constexpr bool kPreloadV = ArchTag::kMinComputeCapability >= 80 &&
      cutlass::sizeof_bits<scalar_t>::value == 16;
  static constexpr bool kKeepOutputInRF = kSingleValueIteration;
  static constexpr bool kNeedsOutputAccumulatorBuffer = !kKeepOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  static_assert(kQueriesPerBlock % 32 == 0, "");
  static_assert(kKeysPerBlock % 32 == 0, "");
  static constexpr int kNumWarpsPerBlock =
      kQueriesPerBlock * kKeysPerBlock / (32 * 32);
  static constexpr int kWarpSize = 32;

  // Launch bounds
  static constexpr int kNumThreads = kWarpSize * kNumWarpsPerBlock;
  static constexpr int kMinBlocksPerSm =
      getWarpsPerSm<scalar_t, ArchTag>() / kNumWarpsPerBlock;

  struct Params {
    // Input tensors
    scalar_t* query_ptr; // [num_queries, num_heads, head_dim]
    scalar_t* key_ptr; // [num_keys, num_heads, head_dim]
    scalar_t* value_ptr; // [num_keys, num_heads, head_dim_value]
    scalar_t* attn_bias_ptr = nullptr; // [num_heads, num_queries, num_keys]
    int32_t* seqstart_q_ptr = nullptr;
    int32_t* seqstart_k_ptr = nullptr;

    // pointer to save result after softmax and dropout
    scalar_t* s_dmask_ptr = nullptr; // [batch_size, num_heads, num_queries, num_keys]

    int32_t* causal_diagonal_ptr = nullptr;
    int32_t* seqlen_k_ptr = nullptr;
    uint32_t causal_diagonal_offset = 0;

    // Output tensors
    output_t* output_ptr; // [num_queries, num_heads, head_dim_value]
    output_accum_t*
        output_accum_ptr; // [num_queries, num_heads, head_dim_value]
    lse_scalar_t* logsumexp_ptr; // [num_heads, num_queries] - can be null
    // Scale
    accum_t scale = 0;
    // Dimensions/strides
    int32_t head_dim = 0;
    int32_t head_dim_value = 0;
    int32_t num_queries = 0;
    int32_t num_keys = 0;

    int32_t num_keys_absolute = 0;
    int32_t num_keys_softmax  = 0;

    // keep this param for compatibility
    int32_t* cu_seqlens_q_ptr = nullptr;
    int32_t* cu_seqlens_k_ptr = nullptr;
    bool causal;

    uint8_t custom_mask_type = NoCustomMask;

    int32_t q_strideM = 0;
    int32_t k_strideM = 0;
    int32_t v_strideM = 0;
    int32_t bias_strideM = 0;

    int32_t o_strideM = 0;

    // Everything below is only used in `advance_to_block`
    // and shouldn't use registers
    int32_t q_strideH = 0;
    int32_t k_strideH = 0;
    int32_t v_strideH = 0;
    int32_t o_strideH = 0;
    int64_t bias_strideH = 0;

    int64_t q_strideB = 0;
    int64_t k_strideB = 0;
    int64_t v_strideB = 0;
    int64_t o_strideB = 0;
    int64_t bias_strideB = 0;

    int32_t num_batches = 0;
    int32_t num_heads = 0;

    // dropout
    bool use_dropout = false;
    unsigned long long dropout_batch_head_rng_offset = 0;
    float dropout_prob = 0.0f;
#ifdef HAS_PYTORCH
    at::PhiloxCudaState rng_engine_inputs = at::PhiloxCudaState(0, 0);
    int64_t* extragraph_offset = nullptr;
    int64_t* seed = nullptr;
#endif

    // Moves pointers to what we should process
    // Returns "false" if there is no work to do
    CUTLASS_DEVICE bool advance_to_block() {
      auto batch_id = blockIdx.z;
      auto head_id = blockIdx.y;
      auto query_start = blockIdx.x * kQueriesPerBlock;

      auto lse_dim = ceil_div((int32_t)num_queries, kAlignLSE) * kAlignLSE;

      if (kSupportsDropout) {
        dropout_batch_head_rng_offset =
            batch_id * num_heads * num_queries * num_keys +
            head_id * num_queries * num_keys;
      }

      if (kReturnSoftmax) {
        if (s_dmask_ptr != nullptr) {
          s_dmask_ptr += batch_id * num_heads * num_queries * num_keys;
          s_dmask_ptr += head_id * num_queries * num_keys;
        }
      }

      if (scale == 0) {
        scale = 1.0f / cutlass::fast_sqrt(float(head_dim));
      }

      num_keys_softmax = num_keys;

      int64_t q_start, k_start;
      // Advance to current batch - in case of different sequence lengths
      if (seqstart_q_ptr != nullptr) {
        assert(seqstart_k_ptr != nullptr);
        seqstart_q_ptr += batch_id;

        q_start = seqstart_q_ptr[0];
        int64_t q_next_start = seqstart_q_ptr[1];
        int64_t k_end;
        seqstart_k_ptr += batch_id;

        if (seqlen_k_ptr) {
          k_start = seqstart_k_ptr[0];
          k_end = k_start + seqlen_k_ptr[batch_id];
        } else {
          k_start = seqstart_k_ptr[0];
          k_end = seqstart_k_ptr[1];
        }

        num_queries = q_next_start - q_start;
        num_keys = k_end - k_start;

        if (query_start >= num_queries) {
          return false;
        }
      } else {
        query_ptr += batch_id * q_strideB;
        key_ptr += batch_id * k_strideB;
        value_ptr += batch_id * v_strideB;
        output_ptr += batch_id * o_strideB;
        if (output_accum_ptr != nullptr) {
          output_accum_ptr += batch_id * o_strideB;
        }
        q_start = 0;
        k_start = 0;
      }

      // Advance to the current batch / head / query_start
      query_ptr += int64_t(q_start + query_start) * q_strideM + head_id * q_strideH;
      key_ptr += k_start * k_strideM + head_id * k_strideH;

      value_ptr += k_start * v_strideM + head_id * v_strideH;
      output_ptr += int64_t(q_start + query_start) * o_strideM +
          head_id * o_strideH;
      if (kSupportsBias && attn_bias_ptr != nullptr) {
        attn_bias_ptr += (batch_id * bias_strideB) + (head_id * bias_strideH);
      }
      if (output_accum_ptr != nullptr) {
        output_accum_ptr += int64_t(q_start + query_start) * o_strideM +
            head_id * o_strideH;
      } else {
        // Accumulate directly in the destination buffer (eg for f32)
        output_accum_ptr = (accum_t*)output_ptr;
      }

      if (logsumexp_ptr != nullptr) {
        // lse[batch_id, head_id, query_start]
        logsumexp_ptr +=
            batch_id * lse_dim * num_heads + head_id * lse_dim + query_start;
      }

      num_keys_absolute = num_keys;

      // Custom masking
      if (causal_diagonal_ptr) {
        causal_diagonal_offset = causal_diagonal_ptr[batch_id];
      }
      if (custom_mask_type == CausalFromBottomRight) {
        causal_diagonal_offset += num_keys - num_queries;
      }
      if (custom_mask_type == CausalFromTopLeft ||
          custom_mask_type == CausalFromBottomRight) {
        // the bottom row of the current block is query_start + kQueriesPerBlock
        // the last active key is then query_start + causal_diagonal_offset +
        // kQueriesPerBlock so num_keys is the min between actual num_keys and
        // this to avoid extra computations
        num_keys = cutlass::fast_min(
            int32_t(query_start + causal_diagonal_offset + kQueriesPerBlock),
            num_keys);
      }

      num_queries -= query_start;
      num_batches = 0; // no longer used after

      // If num_queries == 1, and there is only one key head we're wasting
      // 15/16th of tensor core compute In that case :
      //  - we only launch kernels for head_id % kQueriesPerBlock == 0
      //  - we iterate over heads instead of queries (strideM = strideH)
      if (num_queries == 1 && k_strideH == 0 && v_strideH == 0) {
        if (head_id % kQueriesPerBlock != 0)
          return false;
        q_strideM = q_strideH;
        bias_strideM = bias_strideH;
        num_queries = num_heads;
        num_heads = 1; // unused but here for intent
        // remove causal since n_query = 1
        // otherwise, offset would change with head !
        custom_mask_type = NoCustomMask;
        o_strideM = head_dim_value;
      }

      // Make sure the compiler knows these variables are the same on all
      // the threads of the warp.
      // Only worth doing if they could have been modified above.
      query_ptr = warp_uniform(query_ptr);
      key_ptr = warp_uniform(key_ptr);
      value_ptr = warp_uniform(value_ptr);
      if (kSupportsBias) {
        attn_bias_ptr = warp_uniform(attn_bias_ptr);
      }
      output_ptr = warp_uniform(output_ptr);
      output_accum_ptr = warp_uniform(output_accum_ptr);
      logsumexp_ptr = warp_uniform(logsumexp_ptr);
      if (kReturnSoftmax) {
        s_dmask_ptr = warp_uniform(s_dmask_ptr);
      }
      num_queries = warp_uniform(num_queries);
      num_keys = warp_uniform(num_keys);
      num_heads = warp_uniform(num_heads);
      o_strideM = warp_uniform(o_strideM);
      custom_mask_type = warp_uniform(custom_mask_type);
      return true;
    }

    __host__ dim3 getBlocksGrid() const {
      return dim3(
          ceil_div(num_queries, (int32_t)kQueriesPerBlock),
          num_heads,
          num_batches);
    }

    __host__ dim3 getThreadsGrid() const {
      return dim3(kWarpSize, kNumWarpsPerBlock, 1);
    }
  };

  struct MM0 {
    /*
      In this first matmul, we compute a block of `Q @ K.T`.
      While the calculation result is still hot in registers, we update
      `mi`, `m_prime`, `s_prime` in shared-memory, and then store this value
      into a shared-memory ("AccumulatorSharedStorage") that is used later as
      operand A for the second matmul (see MM1)
    */
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            scalar_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int kAlignmentA =
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment;
    static constexpr int kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        cutlass::layout::ColumnMajor, // LayoutB,
        kAlignmentB,
        accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        OpClass,
        ArchTag, // ArchTag
        ThreadblockShape, // ThreadblockShape
        WarpShape, // WarpShape
        typename GemmType::InstructionShape, // InstructionShape
        DefaultConfig::kStages, // Should use `DefaultConfig::kStages`, but that
                                // uses too much smem
        typename GemmType::Operator // Operator
        >::DefaultMma;
    using MmaCore = typename DefaultMma::MmaCore;
    using IteratorA = typename DefaultMma::IteratorA;
    using IteratorB = typename DefaultMma::IteratorB;
    using Mma = typename DefaultMma::ThreadblockMma;
    using ScalingCoefsUpdater = typename DefaultAttentionScalingCoefsUpdater<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Updater;
    using AccumLambdaIterator = typename DefaultMmaAccumLambdaIterator<
        typename Mma::Operator::IteratorC,
        accum_t,
        kWarpSize>::Iterator;
    static_assert(
        MmaCore::WarpCount::kM * MmaCore::WarpCount::kN *
                MmaCore::WarpCount::kK ==
            kNumWarpsPerBlock,
        "");

    // used for efficient load of bias tile Bij from global to shared memory
    using BiasLoader = TileSmemLoader<
        scalar_t,
        cutlass::MatrixShape<kQueriesPerBlock, kKeysPerBlock>,
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
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MM1 {
    /**
      Second matmul: perform `attn @ V` where `attn` is the attention (not
      normalized) and stored in shared memory
    */
    using GemmType = DefaultGemmType<ArchTag, scalar_t>;

    using OpClass = typename GemmType::OpClass;
    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            scalar_t,
            scalar_t,
            output_accum_t, // ElementC
            accum_t // ElementAccumulator
            >;
    static constexpr int kAlignmentA = DefaultConfig::kAlignmentA; // from smem
    static constexpr int kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;
    using ThreadblockShape = cutlass::gemm::
        GemmShape<kQueriesPerBlock, kKeysPerBlock, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    using LayoutB = cutlass::layout::RowMajor;
    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<
        scalar_t, // ElementA,
        cutlass::layout::RowMajor, // LayoutA,
        kAlignmentA,
        scalar_t, // ElementB,
        LayoutB, // LayoutB,
        kAlignmentB,
        output_accum_t,
        cutlass::layout::RowMajor, // LayoutC,
        accum_t,
        OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        typename GemmType::InstructionShape,
        typename DefaultConfig::EpilogueOutputOp,
        void, // ThreadblockSwizzle - not used
        DefaultConfig::kStages,
        false, // SplitKSerial
        typename GemmType::Operator>;

    using WarpIteratorA = typename cutlass::gemm::threadblock::
        DefaultWarpIteratorAFromSharedMemory<
            typename DefaultGemm::Mma::Policy::Operator::Shape, // WarpShape
            typename DefaultGemm::Mma::Policy::Operator::InstructionShape,
            typename DefaultGemm::Mma::Policy::Operator::IteratorA,
            typename DefaultGemm::Mma::Policy>::WarpIterator;
    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            MM0::AccumulatorSharedStorage::Shape::kN, // kMaxK
            WarpIteratorA,
            typename DefaultGemm::Mma::Policy,
            false>; // kScaleOperandA
    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;
    static_assert(
        WarpCount::kM * WarpCount::kN * WarpCount::kK == kNumWarpsPerBlock,
        "");

    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_t>;
    using OutputTileIteratorAccum =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_accum_t>;
  };

  static constexpr int64_t kAlignmentQ = MM0::kAlignmentA;
  static constexpr int64_t kAlignmentK = MM0::kAlignmentB;
  static constexpr int64_t kAlignmentV = 1;

  static constexpr int MM0_WarpCount = MM0::Mma::WarpCount::kM * MM0::Mma::WarpCount::kN;

  // Shared storage - depends on kernel params
  // s_prime and mi need more shared memory, as reduce is required between WarpCount::kN
  struct ScalingCoefs {
    cutlass::Array<accum_t, kQueriesPerBlock> m_prime;
    cutlass::Array<accum_t, kQueriesPerBlock * MM0_WarpCount> s_prime;
    cutlass::Array<accum_t, kQueriesPerBlock> mi;
    cutlass::Array<accum_t, kQueriesPerBlock> out_rescale;
    cutlass::Array<accum_t, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>
        addition_storage;
  };

  struct SharedStorageEpilogueAtEnd : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      union {
        typename MM0::BiasLoader::SmemTile bias;
        typename MM0::AccumulatorSharedStorage si;
      };
      typename MM1::Mma::SharedStorage mm1;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return epilogue;
    }
  };

  struct SharedStorageEpilogueInLoop : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      union {
        typename MM0::BiasLoader::SmemTile bias;
        typename MM0::AccumulatorSharedStorage si;
      };
      typename MM1::Mma::SharedStorage mm1;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return after_mm0.epilogue;
    }
  };

  using SharedStorage = typename cutlass::platform::conditional<
      kSingleValueIteration || kKeepOutputInRF,
      SharedStorageEpilogueAtEnd,
      SharedStorageEpilogueInLoop>::type;

  static bool __host__ check_supported(Params const& p) {
    CHECK_ALIGNED_PTR(p.query_ptr, kAlignmentQ);
    CHECK_ALIGNED_PTR(p.key_ptr, kAlignmentK);
    CHECK_ALIGNED_PTR(p.value_ptr, kAlignmentV);
    if (kSupportsBias) {
      CHECK_ALIGNED_PTR(p.attn_bias_ptr, kAlignmentQ);
      XFORMERS_CHECK(
          p.num_batches <= 1 || p.bias_strideB % kAlignmentQ == 0,
          "attn_bias is not correctly aligned (strideB)");
      XFORMERS_CHECK(
          p.num_heads <= 1 || p.bias_strideH % kAlignmentQ == 0,
          "attn_bias is not correctly aligned (strideH)");
      XFORMERS_CHECK(
          p.num_queries <= 1 || p.bias_strideM % kAlignmentQ == 0,
          "attn_bias is not correctly aligned (strideM)");
    }
    XFORMERS_CHECK(
        p.q_strideM % kAlignmentQ == 0,
        "query is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.k_strideM % kAlignmentK == 0,
        "key is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.v_strideM % kAlignmentV == 0,
        "value is not correctly aligned (strideM)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.q_strideH % kAlignmentQ == 0,
        "query is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.k_strideH % kAlignmentK == 0,
        "key is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.num_heads <= 1 || p.v_strideH % kAlignmentV == 0,
        "value is not correctly aligned (strideH)");
    XFORMERS_CHECK(
        p.causal_diagonal_ptr == nullptr || p.custom_mask_type != NoCustomMask,
        "`causal_diagonal_ptr` is only useful when `custom_mask_type` is causal");
    XFORMERS_CHECK(
        p.custom_mask_type < NumCustomMaskTypes,
        "invalid value for `custom_mask_type`");
    return true;
  }

  static void CUTLASS_DEVICE attention_kernel(Params& p) {
    // In this block, we will only ever:
    // - read query[query_start:query_end, :]
    // - write to output[query_start:query_end, :]
#if ENABLE_AIU
    extern __shared__ __align__(128) char smem_buffer[];
#else
    extern __shared__ char smem_buffer[];
#endif
    SharedStorage& shared_storage = *((SharedStorage*)smem_buffer);
    auto& m_prime = shared_storage.m_prime;
    auto& s_prime_tmp = shared_storage.s_prime;
    auto& s_prime = *reinterpret_cast<cutlass::Array<accum_t, kQueriesPerBlock>*>(s_prime_tmp.data());
    auto& si = shared_storage.after_mm0.si;
    auto& mi = shared_storage.mi;
    auto& out_rescale = shared_storage.out_rescale;
    const uint32_t query_start = blockIdx.x * kQueriesPerBlock;
    auto s_dmask_ptr = p.s_dmask_ptr;

    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    if (thread_id() < kQueriesPerBlock) {
      s_prime[thread_id()] = accum_t(0);
      out_rescale[thread_id()] = accum_t(1.0);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < MM0_WarpCount; i++) {
        s_prime[thread_id() + i * kQueriesPerBlock] = accum_t(0);
      }
      m_prime[thread_id()] =
          -cutlass::platform::numeric_limits<accum_t>::infinity();
      mi[thread_id()] = -cutlass::platform::numeric_limits<accum_t>::infinity();
    }
    typename MM1::Mma::FragmentC accum_o;
    accum_o.clear();

    auto createOutputIter = [&](int col) -> typename MM1::OutputTileIterator {
      using OutputTileIterator = typename MM1::OutputTileIterator;
      return OutputTileIterator(
          typename OutputTileIterator::Params{(int32_t)p.o_strideM},
          p.output_ptr,
          typename OutputTileIterator::TensorCoord{
              p.num_queries, p.head_dim_value},
          thread_id(),
          {0, col});
    };

    auto createOutputAccumIter = [&](int col) ->
        typename MM1::OutputTileIteratorAccum {
          using OutputTileIteratorAccum = typename MM1::OutputTileIteratorAccum;
          return OutputTileIteratorAccum(
              typename OutputTileIteratorAccum::Params{(int32_t)p.o_strideM},
              p.output_accum_ptr,
              typename OutputTileIteratorAccum::TensorCoord{
                  p.num_queries, p.head_dim_value},
              thread_id(),
              {0, col});
        };

#ifdef HAS_PYTORCH
    curandStatePhilox4_32_10_t curand_state_init;
    if (kSupportsDropout && p.use_dropout) {
      const auto seeds = at::cuda::philox::unpack(p.rng_engine_inputs);
      if (p.rng_engine_inputs.captured_ && p.seed && p.extragraph_offset) {
        // See Note [Seed and Offset Device]
        // When we are in cuda graph capture mode the seed and offset are stored
        // on device We pass in int64_t* seed, and int64_t* offset to act as
        // scratch space for storing the rng state during the forward pass and
        // saving for backwards.
        auto [seed, offset] = seeds;
        *p.seed = seed;
        *p.extragraph_offset = offset;
      }
      // each element of the attention matrix P with shape
      // (batch_sz, n_heads, n_queries, n_keys) is associated with a single
      // offset in RNG sequence. we initialize the RNG state with offset that
      // starts at the beginning of a (n_queries, n_keys) matrix for this
      // block's batch_id and head_id
      // initializing rng state is very expensive, so we run once per kernel,
      // rather than once per iteration. each iteration takes a copy of the
      // initialized RNG state and offsets it as needed.
      curand_init(
          std::get<0>(seeds),
          0,
          std::get<1>(seeds) + p.dropout_batch_head_rng_offset,
          &curand_state_init);
    }
#endif

    // Iterate through keys
    for (int32_t iter_key_start = 0; iter_key_start < p.num_keys;
         iter_key_start += kKeysPerBlock) {
      int32_t problem_size_0_m =
          cutlass::fast_min((int32_t)kQueriesPerBlock, p.num_queries);
      int32_t problem_size_0_n = cutlass::fast_min(
          int32_t(kKeysPerBlock), p.num_keys - iter_key_start);
      int32_t const& problem_size_0_k = p.head_dim;
      int32_t const& problem_size_1_n = p.head_dim_value;
      int32_t const& problem_size_1_k = problem_size_0_n;

      auto prologueV = [&](int blockN) {
        typename MM1::Mma::IteratorB iterator_V(
            typename MM1::IteratorB::Params{MM1::LayoutB(p.v_strideM)},
            p.value_ptr + iter_key_start * p.v_strideM,
            {problem_size_1_k, problem_size_1_n},
            thread_id(),
            cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
        MM1::Mma::prologue(
            shared_storage.after_mm0.mm1,
            iterator_V,
            thread_id(),
            problem_size_1_k);
      };

      __syncthreads(); // Need to have shared memory initialized, and `m_prime`
                       // updated from end of prev iter
      //
      // MATMUL: Q.K_t
      //
      // Computes the block-matrix product of:
      // (a) query[query_start:query_end, :]
      // with
      // (b) key[iter_key_start:iter_key_start + kKeysPerBlock]
      // and stores that into `shared_storage.si`
      //

      // Compute threadblock location
      cutlass::gemm::GemmCoord tb_tile_offset = {0, 0, 0};

      cutlass::MatrixCoord tb_offset_A{
          tb_tile_offset.m() * MM0::Mma::Shape::kM, tb_tile_offset.k()};

      cutlass::MatrixCoord tb_offset_B{
          tb_tile_offset.k(), tb_tile_offset.n() * MM0::Mma::Shape::kN};

      // Construct iterators to A and B operands
      typename MM0::IteratorA iterator_A(
          typename MM0::IteratorA::Params(
              typename MM0::MmaCore::LayoutA(p.q_strideM)),
          p.query_ptr,
          {problem_size_0_m, problem_size_0_k},
          thread_id(),
          tb_offset_A);

      typename MM0::IteratorB iterator_B(
          typename MM0::IteratorB::Params(
              typename MM0::MmaCore::LayoutB(p.k_strideM)),
          p.key_ptr + iter_key_start * p.k_strideM,
          {problem_size_0_k, problem_size_0_n},
          thread_id(),
          tb_offset_B);

      auto my_warp_id = warp_id();
      auto my_lane_id = lane_id();

      // Construct thread-scoped matrix multiply
      typename MM0::Mma mma(
          shared_storage.mm0, thread_id(), my_warp_id, my_lane_id);

      typename MM0::Mma::FragmentC accum;

      accum.clear();

      auto gemm_k_iterations =
          (problem_size_0_k + MM0::Mma::Shape::kK - 1) / MM0::Mma::Shape::kK;

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
      MM1::Mma::drain_cp_asyncs();
      __syncthreads();

      if (kPreloadV) {
        prologueV(0);
      }

      typename MM0::Mma::Operator::IteratorC::TensorCoord
          iteratorC_tile_offset = {
              (tb_tile_offset.m() * MM0::Mma::WarpCount::kM) +
                  (my_warp_id % MM0::Mma::WarpCount::kM),
              (tb_tile_offset.n() * MM0::Mma::WarpCount::kN) +
                  (my_warp_id / MM0::Mma::WarpCount::kM)};

      // multiply by scaling factor
      if (kSupportsBias) {
        accum =
            cutlass::multiplies<typename MM0::Mma::FragmentC>()(p.scale, accum);
      }

      // apply attention bias if applicable
      if (kSupportsBias && p.attn_bias_ptr != nullptr) {
        // load bias tile Bij into shared memory
        typename MM0::BiasLoader::GmemTileIterator bias_iter(
            {cutlass::layout::RowMajor(p.bias_strideM)},
            // attn_bias_pointer points to matrix of size (n_queries, n_keys)
            // for the relevant batch_id and head_id
            p.attn_bias_ptr + query_start * p.bias_strideM + iter_key_start,
            {problem_size_0_m, problem_size_0_n},
            thread_id());
        cutlass::TensorRef<scalar_t, cutlass::layout::RowMajor> bias_tensor_ref(
            shared_storage.after_mm0.bias.data(),
            cutlass::layout::RowMajor(MM0::ThreadblockShape::kN));
        typename MM0::BiasLoader::SmemTileIterator smem_tile_iter(
            bias_tensor_ref, thread_id());
        MM0::BiasLoader::load(bias_iter, smem_tile_iter);

        // Pij += Bij, Pij is in register fragment and Bij is in shared memory
        auto lane_offset = MM0::AccumLambdaIterator::get_lane_offset(
            lane_id(), warp_id(), iteratorC_tile_offset);
        MM0::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
              if (accum_m < problem_size_0_m && accum_n < problem_size_0_n) {
                accum[idx] += (accum_t)bias_tensor_ref.at({accum_m, accum_n});
              }
            },
            [&](int accum_m) {});
      }

      // Mask out last if causal
      // This is only needed if upper-right corner of current query / key block
      // intersects the mask Coordinates of upper-right corner of current block
      // is y=query_start x=min(iter_key_start + kKeysPerBlock, num_keys)) The
      // first masked element is x = y + offset -> query_start + offset There is
      // intersection (and we need to mask) if min(iter_key_start +
      // kKeysPerBlock, num_keys)) >= query_start + offset
      if (p.custom_mask_type &&
          cutlass::fast_min(iter_key_start + kKeysPerBlock, p.num_keys) >=
              (query_start + p.causal_diagonal_offset)) {
        auto query_start = blockIdx.x * kQueriesPerBlock;
        auto lane_offset = MM0::AccumLambdaIterator::get_lane_offset(
            lane_id(), warp_id(), iteratorC_tile_offset);
        int32_t last_col;
        MM0::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {
              // last absolute col is (last absolute query + offset)
              // last local col is (last absolute query + offset -
              // iter_key_start)
              last_col = query_start + accum_m + p.causal_diagonal_offset -
                  iter_key_start;
            },
            [&](int accum_m, int accum_n, int idx) {
              if (accum_n > last_col) {
                accum[idx] =
                    -cutlass::platform::numeric_limits<accum_t>::infinity();
              }
            },
            [&](int accum_m) {});
      }
      // Update `mi` from accum stored in registers
      // Also does accum[i] <- exp(accum[i] - mi)
      iterative_softmax<typename MM0::Mma::Operator::IteratorC>(
          accum_o,
          accum,
          mi,
          m_prime,
          s_prime,
          out_rescale,
          shared_storage.addition_storage,
          my_lane_id,
          thread_id(),
          my_warp_id,
          p.num_keys - iter_key_start,
          iter_key_start == 0,
          iteratorC_tile_offset,
          kSupportsBias ? 1.0f : p.scale);

      // Output results to shared-memory
      int warp_idx_mn_0 = my_warp_id %
          (MM0::Mma::WarpCount::kM * MM0::Mma::WarpCount::kN);
      auto output_tile_coords = cutlass::MatrixCoord{
          warp_idx_mn_0 % MM0::Mma::WarpCount::kM,
          warp_idx_mn_0 / MM0::Mma::WarpCount::kM};

      MM0::B2bGemm::accumToSmem(
          shared_storage.after_mm0.si, accum, my_lane_id, output_tile_coords);

      __syncthreads();

#ifdef HAS_PYTORCH
      // apply dropout (if applicable) after we've written Pij to smem.
      // dropout is applied by multiplying each element of Pij by:
      // - 0 with probability dropout_p
      // - 1 / (1 - dropout_p) with probability 1 - dropout_p
      //
      // for backward purposes we want to be able to map each element of the
      // attention matrix to the same random uniform number as the one we used
      // in forward, without needing to use the same iteration order or having
      // to store the dropout matrix. its possible to do this in registers but
      // it ends up being very slow because each thread having noncontiguous
      // strips of the Pij tile means we have to skip around a lot, and also
      // have to generate a single random number at a time
      if (kSupportsDropout && p.use_dropout) {
        auto si = shared_storage.after_mm0.si.accum_ref();
        // each thread handles a contiguous sequence of elements from Sij, all
        // coming from the same row. the reason they have to come from the same
        // row is that the sampling random numbers from a contiguous random
        // number sequence is much more efficient than jumping around, and the
        // linear offset of each element of S (the global matrix) maps to an
        // offset in a random number sequence. for S, the end of a row and the
        // beginning of the next have adjacent offsets, but for Sij, this is not
        // necessarily the case.
        const int num_threads = blockDim.x * blockDim.y * blockDim.z;
        const int threads_per_row =
            cutlass::fast_min(num_threads / problem_size_0_m, problem_size_0_n);
        const int elts_per_thread = cutlass::round_nearest(
            cutlass::ceil_div(problem_size_0_n, threads_per_row), 4);

        const int thread_i = thread_id() / threads_per_row;
        const int thread_start_j =
            (thread_id() % threads_per_row) * elts_per_thread;

        if (thread_i < problem_size_0_m && thread_start_j < problem_size_0_n) {
          curandStatePhilox4_32_10_t curand_state = curand_state_init;
          skipahead(
              static_cast<unsigned long long>(
                  (query_start + thread_i) * p.num_keys_absolute +
                  (iter_key_start + thread_start_j)),
              &curand_state);
          const float dropout_scale = 1.0f / (1.0f - p.dropout_prob);

          // apply dropout scaling to elements this thread is responsible for,
          // in chunks of 4
          for (int sij_start_col_idx = thread_start_j; sij_start_col_idx <
               cutlass::fast_min(thread_start_j + elts_per_thread,
                                 problem_size_0_n);
               sij_start_col_idx += 4) {
            const float4 rand_uniform_quad = curand_uniform4(&curand_state);

            CUTLASS_PRAGMA_UNROLL
            for (int quad_idx = 0; quad_idx < 4; ++quad_idx) {
              if (kReturnSoftmax) {
                if (sij_start_col_idx + quad_idx < problem_size_0_n) {
                  int s_index = (query_start + thread_i) * p.num_keys_softmax + iter_key_start + sij_start_col_idx + quad_idx;
                  s_dmask_ptr[s_index] = si.at({thread_i, sij_start_col_idx + quad_idx}) * (
                      (scalar_t)(((&rand_uniform_quad.x)[quad_idx] > p.dropout_prob) ? 1.0f : -1.0f)
                      );
                }
              }

              si.at({thread_i, sij_start_col_idx + quad_idx}) *=
                  static_cast<scalar_t>(
                      dropout_scale *
                      ((&rand_uniform_quad.x)[quad_idx] > p.dropout_prob));
            }
          }
        }
        __syncthreads(); // p.use_dropout should have same value kernel-wide
      }
#endif

      //
      // MATMUL: Attn . V
      // Run the matmul `attn @ V` for a block of attn and V.
      // `attn` is read from shared memory (in `shared_storage_si`)
      // `V` is read from global memory (with iterator_B)
      //

      const int64_t nBlockN = kSingleValueIteration
          ? 1
          : ceil_div(
                (int64_t)problem_size_1_n, int64_t(MM1::ThreadblockShape::kN));
      for (int blockN = 0; blockN < nBlockN; ++blockN) {
        int gemm_k_iterations =
            (problem_size_1_k + MM1::Mma::Shape::kK - 1) / MM1::Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add and store it in accum
        // (in registers)
        if (!kPreloadV) {
          __syncthreads(); // we share shmem between mma and epilogue
        }

        typename MM1::Mma::IteratorB iterator_V(
            typename MM1::IteratorB::Params{MM1::LayoutB(p.v_strideM)},
            p.value_ptr + iter_key_start * p.v_strideM,
            {problem_size_1_k, problem_size_1_n},
            thread_id(),
            cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});
        typename MM1::Mma mma_pv(
            // operand A: Pij_dropped in shared memory
            shared_storage.after_mm0.si.accum_ref(),
            // operand B: shared memory staging area for Vj, which is loaded
            // from global memory
            shared_storage.after_mm0.mm1.operand_B_ref(),
            (int)thread_id(),
            (int)my_warp_id,
            (int)my_lane_id);
        mma_pv.set_prologue_done(kPreloadV);
        if (!kKeepOutputInRF) {
          accum_o.clear();
        }
        mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o);
        MM1::Mma::drain_cp_asyncs();
        __syncthreads();

        if (kPreloadV && !kSingleValueIteration && blockN + 1 < nBlockN) {
          prologueV(blockN + 1);
        }

        if (!kKeepOutputInRF) {
          DISPATCH_BOOL(
              iter_key_start == 0, kIsFirst, ([&] {
                DISPATCH_BOOL(
                    (iter_key_start + kKeysPerBlock) >= p.num_keys,
                    kIsLast,
                    ([&] {
                      using DefaultEpilogue = typename MM1::DefaultEpilogue;
                      using DefaultOp =
                          typename MM1::DefaultConfig::EpilogueOutputOp;
                      using ElementCompute = typename DefaultOp::ElementCompute;
                      using EpilogueOutputOp = typename cutlass::epilogue::
                          thread::MemoryEfficientAttentionNormalize<
                              typename cutlass::platform::conditional<
                                  kIsLast,
                                  output_t,
                                  output_accum_t>::type,
                              output_accum_t,
                              DefaultOp::kCount,
                              typename DefaultOp::ElementAccumulator,
                              ElementCompute,
                              kIsFirst,
                              kIsLast,
                              cutlass::Array<ElementCompute, kQueriesPerBlock>>;
                      using Epilogue = typename cutlass::epilogue::threadblock::
                          EpiloguePipelined<
                              typename DefaultEpilogue::Shape,
                              typename MM1::Mma::Operator,
                              DefaultEpilogue::kPartitionsK,
                              typename cutlass::platform::conditional<
                                  kIsLast,
                                  typename MM1::OutputTileIterator,
                                  typename MM1::OutputTileIteratorAccum>::type,
                              typename DefaultEpilogue::
                                  AccumulatorFragmentIterator,
                              typename DefaultEpilogue::WarpTileIterator,
                              typename DefaultEpilogue::SharedLoadIterator,
                              EpilogueOutputOp,
                              typename DefaultEpilogue::Padding,
                              DefaultEpilogue::kFragmentsPerIteration,
                              true, // IterationsUnroll
                              typename MM1::OutputTileIteratorAccum // Read
                                                                    // iterator
                              >;

                      int col = blockN * MM1::Mma::Shape::kN;
                      auto source_iter = createOutputAccumIter(col);
                      auto dest_iter = call_conditional<
                          kIsLast,
                          decltype(createOutputIter),
                          decltype(createOutputAccumIter)>::
                          apply(createOutputIter, createOutputAccumIter, col);
                      EpilogueOutputOp rescale(s_prime, out_rescale);
                      Epilogue epilogue(
                          shared_storage.epilogue_shared_storage(),
                          thread_id(),
                          warp_id(),
                          lane_id());
                      epilogue(rescale, dest_iter, accum_o, source_iter);
                    }));
              }));
          if (!kSingleValueIteration) {
            __syncthreads();
          }
        }
      }
      __syncthreads(); // we modify `m_prime` after
    }

    if (kKeepOutputInRF) {
      constexpr bool kIsFirst = true;
      constexpr bool kIsLast = true;
      using DefaultEpilogue = typename MM1::DefaultEpilogue;
      using DefaultOp = typename MM1::DefaultConfig::EpilogueOutputOp;
      using ElementCompute = typename DefaultOp::ElementCompute;
      using EpilogueOutputOp =
          typename cutlass::epilogue::thread::MemoryEfficientAttentionNormalize<
              output_t, // output
              output_accum_t, // source
              DefaultOp::kCount,
              typename DefaultOp::ElementAccumulator, // accum
              output_accum_t, // compute
              kIsFirst,
              kIsLast,
              cutlass::Array<ElementCompute, kQueriesPerBlock>>;
      using Epilogue =
          typename cutlass::epilogue::threadblock::EpiloguePipelined<
              typename DefaultEpilogue::Shape,
              typename MM1::Mma::Operator,
              DefaultEpilogue::kPartitionsK,
              typename MM1::OutputTileIterator, // destination
              typename DefaultEpilogue::AccumulatorFragmentIterator,
              typename DefaultEpilogue::WarpTileIterator,
              typename DefaultEpilogue::SharedLoadIterator,
              EpilogueOutputOp,
              typename DefaultEpilogue::Padding,
              DefaultEpilogue::kFragmentsPerIteration,
              true, // IterationsUnroll
              typename MM1::OutputTileIteratorAccum // source tile
              >;
      auto dest_iter = createOutputIter(0);
      EpilogueOutputOp rescale(s_prime, out_rescale);
      Epilogue epilogue(
          shared_storage.epilogue_shared_storage(),
          thread_id(),
          warp_id(),
          lane_id());
      MM1::Mma::drain_cp_asyncs();
      epilogue(rescale, dest_iter, accum_o);
    }

    // 7. Calculate logsumexp
    // To make the backward easier, we pad logsumexp with `inf`
    // this avoids a few bound checks, and is not more expensive during fwd
    static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    if (p.logsumexp_ptr && thread_id() < kQueriesPerBlock) {
      auto lse_dim = ceil_div((int32_t)p.num_queries, kAlignLSE) * kAlignLSE;
      constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
      if (thread_id() < p.num_queries) {
        // We set fully masked out rows to 0, the sumexp for masked out rows will be 0
        // We update it to be 1 prior to calling log so that log(1) = 0
        s_prime[thread_id()] = (s_prime[thread_id()] == 0) ? 1: s_prime[thread_id()];
        mi[thread_id()] = (mi[thread_id()] == -cutlass::platform::numeric_limits<accum_t>::infinity()) ? 0: mi[thread_id()];
        p.logsumexp_ptr[thread_id()] = accum_t(mi[thread_id()] / kLog2e) +
            cutlass::fast_log(accum_t(s_prime[thread_id()]));
      } else if (thread_id() < lse_dim) {
        p.logsumexp_ptr[thread_id()] =
            cutlass::platform::numeric_limits<accum_t>::infinity();
      }
    }
  }

template <typename WarpIteratorC>
  CUTLASS_DEVICE static void iterative_softmax(
      typename WarpIteratorC::Fragment& frag_o, // output so far
      typename WarpIteratorC::Fragment& frag,
      cutlass::Array<accum_t, kQueriesPerBlock>& mi,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& s_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& out_rescale,
      cutlass::Array<accum_t, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>&
          addition_storage,
      int8_t lane_id,
      int8_t thread_id,
      int8_t warp_id,
      int max_col,
      bool is_first,
      typename WarpIteratorC::TensorCoord const& tile_offset,
      float scaling) {
    /* Iterates on the accumulator and corresponding position on result matrix

    (1) Update `mi[r]` to the max value of the row `r`
    (2) In a second iteration do the following:
        (a) accum   <- exp(accum - mi)
        (b) m_prime <- exp(m_prime - mi)
        (c) s_prime <- s_prime * m_prime + sum(accum)

    All of this is done on registers, before we store all of this
    on shared memory for the next matmul with Value.
    */
    using Fragment = typename WarpIteratorC::Fragment;
    using LambdaIterator = typename DefaultMmaAccumLambdaIterator<
        WarpIteratorC,
        accum_t,
        kWarpSize>::Iterator;
    // Convert to `accum_t` (rather than double)
    constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E

    static_assert(kQueriesPerBlock % kNumWarpsPerBlock == 0, "");
    static constexpr int kLinesPerWarp = kQueriesPerBlock / kNumWarpsPerBlock;

    frag = cutlass::multiplies<Fragment>()(scaling * kLog2e, frag);

    auto lane_offset =
        LambdaIterator::get_lane_offset(lane_id, warp_id, tile_offset);

    // First update `mi` to the max per-row
    {
      accum_t max;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {
            max = -cutlass::platform::numeric_limits<accum_t>::infinity();
          },
          [&](int accum_m, int accum_n, int idx) {
            if (accum_n < max_col) {
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
        out_rescale[id] = m_prime_exp;
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
        out_rescale[id] = 1.0f;
      }
    }
    __syncthreads(); // Update output fragments
    if (kKeepOutputInRF && !is_first) {
      accum_t line_rescale;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { line_rescale = out_rescale[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag_o[idx] = frag_o[idx] * line_rescale;
          },
          [&](int accum_m) {});
    }
    // Update accum_m, accum_n, ...
    {
      accum_t mi_row, total_row;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { mi_row = mi[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag[idx] =
                (accum_n < max_col) ? exp2f(frag[idx] - mi_row) : accum_t(0.0);
          },
          [&](int accum_m) {});
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { total_row = 0.0; },
          [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
          [&](int accum_m) {
            if (LambdaIterator::reduceSameRow(
                    lane_id, total_row, [](accum_t a, accum_t b) {
                      return a + b;
                    })) {
              // NOTE: we could atomically add `total_row` to `s_prime`, but
              // it's faster (and deterministic) to avoid atomics here
              addition_storage
                  [accum_m + kQueriesPerBlock * tile_offset.column()] =
                      total_row;
            }
          });
    }
    __syncthreads();
    if (lane_id < kLinesPerWarp) {
      int id = warp_id * kLinesPerWarp + lane_id;
      accum_t total_row = s_prime[id];
      if (restore_mi_to_minus_inf) {
        // Restore `mi`, see above when we set `restore_mi_to_minus_inf=true`
        mi[id] = -cutlass::platform::numeric_limits<accum_t>::infinity();
      } else {
        m_prime[id] = mi[id];
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < MM0::MmaCore::WarpCount::kN; ++i) {
        total_row += addition_storage[id + kQueriesPerBlock * i];
      }
      s_prime[id] = total_row;
    }
  }

  static CUTLASS_DEVICE int8_t lane_id() {
    return threadIdx.x;
  }
  static CUTLASS_DEVICE int8_t warp_id() {
    return threadIdx.y;
  }
  static CUTLASS_DEVICE int16_t thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x;
  }
};

template <typename AK>
__global__ void //__launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_batched_impl(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::attention_kernel(p);
}

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_batched(typename AK::Params params);
