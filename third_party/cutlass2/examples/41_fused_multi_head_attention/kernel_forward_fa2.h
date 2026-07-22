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
    int kHeadDim_,
    int kQueriesPerBlock,
    int kKeysPerBlock_,
    int kQueriesPerWarp,
    int kKeysPerWarp,
    // Set to false if you know at compile-time you will never need dropout
    bool kSupportsDropout_ = false,
    bool kSupportsBias_ = false,
    bool kReturnSoftmax = false>
struct AttentionKernelFA2 {
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
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr bool kSupportsDropout = kSupportsDropout_;
  static constexpr bool kSupportsBias = kSupportsBias_;
  static constexpr int kKeysPerBlock = kKeysPerBlock_;
  static constexpr bool kIsAligned = isAligned_;
  static constexpr int kMaxK = kHeadDim_;
  static constexpr bool kSingleValueIteration = true;
  static constexpr int32_t kAlignLSE = 32; // block size of backward
  static constexpr bool kPreloadV = ArchTag::kMinComputeCapability >= 80 &&
      cutlass::sizeof_bits<scalar_t>::value == 16;
  static constexpr bool kKeepOutputInRF = kSingleValueIteration;
  static constexpr bool kNeedsOutputAccumulatorBuffer = !kKeepOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  static_assert(kQueriesPerBlock % 32 == 0, "");
  static_assert(kKeysPerBlock % 32 == 0, "");
  static constexpr int kWarpSize = 32;

#if !LAYOUT_CONVERT_ON_SHARED
  static_assert(kKeysPerBlock == kKeysPerWarp);
#endif

  static constexpr int kNumWarpsM = kQueriesPerBlock / kQueriesPerWarp;
  static constexpr int kNumWarpsN = kKeysPerBlock / kKeysPerWarp;

  static constexpr int kNumWarpsPerBlock = kNumWarpsM * kNumWarpsN;

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
    int32_t head_dim;
    int32_t head_dim_value;
    int32_t num_queries;
    int32_t num_keys;

    int32_t num_keys_absolute = 0;
    int32_t num_keys_softmax  = 0;

    // keep this param for compatibility
    int32_t* cu_seqlens_q_ptr = nullptr;
    int32_t* cu_seqlens_k_ptr = nullptr;
    bool causal;

    uint8_t custom_mask_type = NoCustomMask;

    int32_t q_strideM;
    int32_t k_strideM;
    int32_t v_strideM;
    int32_t bias_strideM = 0;

    int32_t o_strideM = 0;

    // Everything below is only used in `advance_to_block`
    // and shouldn't use registers
    int32_t q_strideH;
    int32_t k_strideH;
    int32_t v_strideH;
    int32_t o_strideH;
    int64_t bias_strideH = 0;

    int64_t q_strideB;
    int64_t k_strideB;
    int64_t v_strideB;
    int64_t o_strideB;
    int64_t bias_strideB = 0;

    int32_t num_batches;
    int32_t num_heads;
    int32_t num_heads_kv = 0;

    // dropout
    bool use_dropout;
    unsigned long long dropout_batch_head_rng_offset;
    float dropout_prob;
#ifdef HAS_PYTORCH
    at::PhiloxCudaState rng_engine_inputs;
    uint64_t * rng_state = nullptr;
    int64_t* extragraph_offset = nullptr;
    int64_t* seed = nullptr;
#endif

    // Moves pointers to what we should process
    // Returns "false" if there is no work to do
    CUTLASS_DEVICE bool advance_to_block() {
      auto batch_id = blockIdx.z;
      auto head_id = blockIdx.y;
      auto query_start = blockIdx.x * kQueriesPerBlock;

      if (num_heads_kv == 0) { num_heads_kv = num_heads; }
      int head_id_kv = head_id / (num_heads / num_heads_kv);

      auto lse_dim = ceil_div((int32_t)num_queries, kAlignLSE) * kAlignLSE;

      if (kSupportsDropout) {
        dropout_batch_head_rng_offset = batch_id * num_heads + head_id;
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
      key_ptr += k_start * k_strideM + head_id_kv * k_strideH;

      value_ptr += k_start * v_strideM + head_id_kv * v_strideH;
      output_ptr += int64_t(q_start + query_start) * o_strideM +
          head_id * o_strideH;
      if (kSupportsBias && attn_bias_ptr != nullptr) {
        attn_bias_ptr += (batch_id * bias_strideB) + (head_id_kv * bias_strideH);
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
        num_queries = num_heads;
        num_heads = 1; // unused but here for intent
        // remove causal since n_query = 1
        // otherwise, offset would change with head !
        custom_mask_type = NoCustomMask;
        o_strideM = head_dim_value;
      }

      // Make sure the compiler knows these variables are the same on all
      // the threads of the warp.
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
      head_dim = warp_uniform(head_dim);
      head_dim_value = warp_uniform(head_dim_value);
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

  template <
      typename Shape_,
      typename Element_,
      typename Layout_ = cutlass::layout::RowMajor,
      typename Padding_ = cutlass::MatrixShape<0, 0>>
  class SharedStorage_ {
  public:
    //
    // Type definitions
    //
    using Shape = Shape_;
    using Element = Element_;
    using Layout = Layout_;
    using Padding = Padding_;

    /// Tensor reference to the accumulator
    using TensorRefAccum = cutlass::TensorRef<Element, Layout>;

    /// Shape of the accumulator matrix in shared memory
    using ShapeAccum = cutlass::
        MatrixShape<Shape::kRow + Padding::kRow, Shape::kColumn + Padding::kColumn>;

  public:
    //
    // Data members
    //

    /// Buffer for accumulator
    cutlass::AlignedBuffer<Element, ShapeAccum::kCount> buffer;

  public:
    //
    // Methods
    //

    /// Returns a layout object for the Accum matrix
    CUTLASS_DEVICE
    static Layout LayoutAccum() {
      return Layout::packed({ShapeAccum::kRow, ShapeAccum::kColumn});
    }

    /// Returns a TensorRef to the Accumulator
    CUTLASS_HOST_DEVICE
    TensorRefAccum buffer_ref() {
      return TensorRefAccum{buffer.data(), LayoutAccum()};
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
        GemmShape<kQueriesPerBlock, kKeysPerBlock, kHeadDim_>;
    using WarpShape = cutlass::gemm::GemmShape<kQueriesPerWarp, kKeysPerWarp, kHeadDim_>;
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
        GemmShape<kQueriesPerBlock, kHeadDim_, kKeysPerBlock>;
    using WarpShape = cutlass::gemm::GemmShape<kQueriesPerWarp, kHeadDim_ / kNumWarpsN, kKeysPerBlock>;
    using InstructionShape = typename GemmType::InstructionShape;

    static constexpr int align_bits = 128; //(kHeadDim_ == 80) ? 64 : 128;

    using EpilogueOutputOp = typename cutlass::epilogue::thread::LinearCombination<
      output_accum_t,
      align_bits / cutlass::sizeof_bits<output_accum_t>::value,
      accum_t,
      accum_t
    >;

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
        EpilogueOutputOp,
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

  using ScalingCoefsUpdater = typename DefaultAttentionScalingCoefsUpdaterFA2<
      typename MM0::Mma::Operator::IteratorC,
      typename MM1::Mma::Operator::IteratorC,
      accum_t,
      kWarpSize>::Updater;

  static constexpr int64_t kAlignmentQ = MM0::kAlignmentA;
  static constexpr int64_t kAlignmentK = MM0::kAlignmentB;
  static constexpr int64_t kAlignmentV = 1;

  static constexpr int MM0_WarpCount = MM0::Mma::WarpCount::kM * MM0::Mma::WarpCount::kN;

  using SmemQ = SharedStorage_<cutlass::MatrixShape<kQueriesPerBlock, kHeadDim_>,  // M x K
                               scalar_t_
#if !ENABLE_AIU
                               , typename MM0::Mma::SmemIteratorA::Layout>;
#else
                               >;
#endif
  using SmemK = SharedStorage_<cutlass::MatrixShape<kHeadDim_, kKeysPerBlock_>,    // K x N
                               scalar_t_
#if !ENABLE_AIU
                               ,typename MM0::Mma::SmemIteratorB::Layout>;
#else
                               >;
#endif
  using SmemV = SharedStorage_<cutlass::MatrixShape<kKeysPerBlock_, kHeadDim_>,    // K x N
                               scalar_t_
#if !ENABLE_AIU
                               ,typename MM1::Mma::SmemIteratorB1::Layout>;
#else
                               >;
#endif

  struct ScalingCoefs {

  };

  struct SharedStorageEpilogueAtEnd : ScalingCoefs {
    struct SharedMainloop {
      union {
        typename MM0::BiasLoader::SmemTile bias;
#if LAYOUT_CONVERT_ON_SHARED
        typename MM0::AccumulatorSharedStorage si;
#endif
      };
      SmemQ smem_q;
      SmemK smem_k;
      SmemV smem_v;
    };

    cutlass::Array<accum_t, kQueriesPerBlock> m_prime;
    cutlass::Array<accum_t, kQueriesPerBlock * MM0::Mma::WarpCount::kN> s_prime;
    cutlass::Array<accum_t, kQueriesPerBlock> mi;
    union {
      SharedMainloop mainloop;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return epilogue;
    }

    CUTLASS_DEVICE void print_size() {
      printf("smem_q = %d\n", mainloop.smem_q.buffer.size() * sizeof(scalar_t));
      printf("smem_k = %d\n", mainloop.smem_k.buffer.size() * sizeof(scalar_t));
      printf("smem_v = %d\n", mainloop.smem_v.buffer.size() * sizeof(scalar_t));
      printf("m_prime = %d\n", m_prime.size() * sizeof(accum_t));
      printf("s_prime = %d\n", s_prime.size() * sizeof(accum_t));
      printf("mi = %d\n", mi.size() * sizeof(accum_t));
      printf("bias = %d\n", mainloop.bias.size() * sizeof(accum_t));
#if LAYOUT_CONVERT_ON_SHARED
      printf("si = %d\n", mainloop.si.accum.size() * sizeof(typename MM0::AccumulatorSharedStorage::Element));
#endif
      printf("epilogue = %d\n", sizeof(epilogue));
      int sum =   mainloop.smem_q.buffer.size() * sizeof(scalar_t)
                + mainloop.smem_k.buffer.size() * sizeof(scalar_t)
                + mainloop.smem_v.buffer.size() * sizeof(scalar_t)
                + m_prime.size() * sizeof(accum_t)
                + s_prime.size() * sizeof(accum_t)
                + mi.size() * sizeof(accum_t)
#if LAYOUT_CONVERT_ON_SHARED
                + mainloop.si.accum.size() * sizeof(typename MM0::AccumulatorSharedStorage::Element)
#endif
                + sizeof(epilogue);
      printf("sum = %d\n", sum);
    }
  };

  struct SharedStorageEpilogueInLoop {
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
          p.bias_strideM % kAlignmentQ == 0,
          "attn_bias is not correctly aligned");
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
    XFORMERS_CHECK(
        (p.num_heads_kv == 0) || (p.num_heads_kv > 0 && p.num_heads % p.num_heads_kv == 0),
        "p.num_heads must be dividable by p.num_heads_kv");
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

    auto& mi = shared_storage.mi;

    // if (thread_id() == 0) {
    //   shared_storage.print_size();
    // }

    const uint32_t query_start = blockIdx.x * kQueriesPerBlock;
    auto s_dmask_ptr = p.s_dmask_ptr;

    static_assert(kQueriesPerBlock <= kNumWarpsPerBlock * kWarpSize, "");
    if (thread_id() < kQueriesPerBlock) {
      //s_prime[thread_id()] = accum_t(0);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < MM0::Mma::WarpCount::kN; i++) {
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
    const auto seeds = at::cuda::philox::unpack(p.rng_engine_inputs);

    if (kSupportsDropout && p.use_dropout) {
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
      if (p.rng_state) {
        p.rng_state[0] = std::get<0>(seeds);
        p.rng_state[1] = std::get<1>(seeds);
      }
#if DROPOUT_ON_SHARED
      curandStatePhilox4_32_10_t curand_state_init;
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
#endif
    }
#endif

    auto my_warp_id = warp_id();
    auto my_lane_id = lane_id();

    int32_t problem_size_0_m =
        cutlass::fast_min((int32_t)kQueriesPerBlock, p.num_queries);
    int32_t problem_size_0_n = cutlass::fast_min(
        int32_t(kKeysPerBlock), p.num_keys/* - iter_key_start*/);
    int32_t const& problem_size_0_k = p.head_dim;
    int32_t const& problem_size_1_n = p.head_dim_value;
    int32_t const& problem_size_1_k = problem_size_0_n;

    // Construct iterators to A and B operands
    typename MM0::IteratorA iterator_Q(
        typename MM0::IteratorA::Params(
            typename MM0::MmaCore::LayoutA(p.q_strideM)),
        p.query_ptr,
        {problem_size_0_m, problem_size_0_k},
        thread_id(),
        {0, 0});

    typename MM0::IteratorB iterator_K(
        typename MM0::IteratorB::Params(
            typename MM0::MmaCore::LayoutB(p.k_strideM)),
        p.key_ptr /*+ iter_key_start * p.k_strideM*/,
        {problem_size_0_k, problem_size_0_n},
        thread_id(),
        {0, 0});

#if !ENABLE_AIU
    typename MM1::Mma::IteratorB iterator_V(
        typename MM1::IteratorB::Params{MM1::LayoutB(p.v_strideM)},
        p.value_ptr /*+ iter_key_start * p.v_strideM*/,
        {problem_size_1_k, problem_size_1_n},
        thread_id(),
        {0, 0});
#else
  using ElementB = scalar_t;
  using LayoutB = typename MM1::LayoutB;
  using WmmaLayoutB =
      typename cutlass::platform::conditional<cutlass::platform::is_same<LayoutB, cutlass::layout::RowMajor>::value,
                                awmma::row_major, awmma::col_major>::type;

  using WmmaFragABType_ = typename FromCutlassType<scalar_t>::type;
  using MmaShape = typename MM1::InstructionShape;
  using FragBType = awmma::fragment<awmma::matrix_b, MmaShape::kM, MmaShape::kN, MmaShape::kK,
                                    WmmaFragABType_, WmmaLayoutB>;
  using IteratorV = aiu::gemm::threadblock::AiuLoaderB<ElementB, LayoutB, typename MM1::ThreadblockShape,
                                                       typename MM1::WarpShape, MmaShape, FragBType>;
  IteratorV iterator_V(
      typename IteratorV::Params{MM1::LayoutB(p.v_strideM)},
      p.value_ptr /*+ iter_key_start * p.v_strideM*/,
      {problem_size_1_k, problem_size_1_n},
      thread_id(),
      {0, 0});
#endif


    int warp_idx_mn = my_warp_id;
    constexpr int warp_idx_k = 0;

    int warp_idx_m = warp_idx_mn % MM0::Mma::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / MM0::Mma::WarpCount::kM;

#if !ENABLE_AIU
    typename MM0::Mma::SmemIteratorA smem_iterator_Q(shared_storage.mainloop.smem_q.buffer_ref(), thread_id());
    typename MM0::Mma::SmemIteratorB smem_iterator_K(shared_storage.mainloop.smem_k.buffer_ref(), thread_id());

    typename MM0::Mma::Operator::IteratorA warp_tile_iterator_Q(shared_storage.mainloop.smem_q.buffer_ref(), my_lane_id);
    typename MM0::Mma::Operator::IteratorB warp_tile_iterator_K(shared_storage.mainloop.smem_k.buffer_ref(), my_lane_id);

    // Add per-warp offsets in units of warp-level tiles
    warp_tile_iterator_Q.add_tile_offset(
        {warp_idx_m, 0});
    warp_tile_iterator_K.add_tile_offset(
        {0, warp_idx_n});

    typename MM1::Mma::SmemIteratorB1 smem_iterator_V(shared_storage.mainloop.smem_v.buffer_ref(), thread_id());
    typename MM1::Mma::Operator::IteratorB warp_tile_iterator_V(shared_storage.mainloop.smem_v.buffer_ref(), my_lane_id);
    warp_tile_iterator_V.add_tile_offset(
        {0, warp_idx_n});

#else
    auto *smem_iterator_Q = reinterpret_cast<WmmaFragABType_*>(shared_storage.mainloop.smem_q.buffer.data());
    auto *smem_iterator_K = reinterpret_cast<WmmaFragABType_*>(shared_storage.mainloop.smem_k.buffer.data());
    auto *smem_iterator_V = reinterpret_cast<WmmaFragABType_*>(shared_storage.mainloop.smem_v.buffer.data());

    auto *warp_tile_iterator_Q = smem_iterator_Q;
    auto *warp_tile_iterator_K = smem_iterator_K;
    auto *warp_tile_iterator_V = smem_iterator_V;

#endif

#if LAYOUT_CONVERT_ON_SHARED
    typename MM1::WarpIteratorA warp_tile_iterator_A(shared_storage.mainloop.si.accum_ref(), my_lane_id);
    warp_tile_iterator_A.add_tile_offset({warp_idx_m, 0});
#endif

    // copy Q from global to shared
    fmha::copy_g2s(iterator_Q, smem_iterator_Q);

    // copy K from global to shared
    fmha::copy_g2s(iterator_K, smem_iterator_K);

    __syncthreads(); // Need to have shared memory initialized, and `m_prime`

    // Defines the boundary of a stage of cp.async.
    cutlass::arch::cp_async_fence();

    // Iterate through keys
    for (int32_t iter_key_start = 0; iter_key_start < p.num_keys;
         iter_key_start += kKeysPerBlock) {

      problem_size_0_n = cutlass::fast_min(
        int32_t(kKeysPerBlock), p.num_keys - iter_key_start);

#if ENABLE_AIU
      // update dim.y if the next V is small than kKeysPerBlock
      if (p.num_keys - iter_key_start < kKeysPerBlock) {
        iterator_V.dim_.y = p.num_keys - iter_key_start;
      }
#endif
      // copy V from global to shared
      fmha::copy_g2s(iterator_V, smem_iterator_V);
      iterator_V.add_pointer_offset(kKeysPerBlock * p.v_strideM);
      cutlass::arch::cp_async_fence();

      //
      // MATMUL: Q.K_t
      //
      // Computes the block-matrix product of:
      // (a) query[query_start:query_end, :]
      // with
      // (b) key[iter_key_start:iter_key_start + kKeysPerBlock]
      // and stores that into `shared_storage.si`
      //
      typename MM0::Mma::FragmentC accum;
      accum.clear();

      cutlass::arch::cp_async_wait<1>();
      __syncthreads();

      // compute 1st gemm
#if !ENABLE_AIU
      fmha::gemm_on_sAsB<typename MM0::Mma>(accum, warp_tile_iterator_Q, warp_tile_iterator_K);
#else
      fmha::gemm_on_sAsB_aiu<typename MM0::Mma>(accum, iterator_Q, iterator_K, warp_tile_iterator_Q, warp_tile_iterator_K);
#endif

      // load next K
      __syncthreads();
      if ((iter_key_start + kKeysPerBlock) < p.num_keys) {
#if ENABLE_AIU
        // update dim.y if the next K is small than kKeysPerBlock
        if (p.num_keys - (iter_key_start + kKeysPerBlock) < kKeysPerBlock) {
          iterator_K.dim_.y = p.num_keys - (iter_key_start + kKeysPerBlock);
        }
#endif
        iterator_K.add_pointer_offset(kKeysPerBlock * p.k_strideM);
        fmha::copy_g2s(iterator_K, smem_iterator_K);
        cutlass::arch::cp_async_fence();
      }

      typename MM0::Mma::Operator::IteratorC::TensorCoord
          iteratorC_tile_offset = {warp_idx_m, warp_idx_n};

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
            p.attn_bias_ptr + int64_t(query_start) * p.bias_strideM + iter_key_start,
            {problem_size_0_m, problem_size_0_n},
            thread_id());
        cutlass::TensorRef<scalar_t, cutlass::layout::RowMajor> bias_tensor_ref(
            shared_storage.mainloop.bias.data(),
            cutlass::layout::RowMajor(MM0::ThreadblockShape::kN));
        typename MM0::BiasLoader::SmemTileIterator smem_tile_iter(
            bias_tensor_ref, thread_id());
        MM0::BiasLoader::load(bias_iter, smem_tile_iter);

        // Pij += Bij, Pij is in register fragment and Bij is in shared memory
        auto lane_offset = MM0::AccumLambdaIterator::get_lane_offset(
            my_lane_id, my_warp_id, iteratorC_tile_offset);
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

      // softmax
      DISPATCH_BOOL(iter_key_start == 0, kIsFirst, ([&] {
                      DISPATCH_BOOL(
                          p.num_keys - iter_key_start >= kKeysPerBlock,
                          kFullColumns,
                          ([&] {
                            // Update `mi` from accum stored in registers
                            // Also does accum[i] <- exp(accum[i] - mi)
                            ScalingCoefsUpdater::template update<
                                kQueriesPerBlock,
                                kFullColumns,
                                kIsFirst,
                                kKeepOutputInRF,
                                MM0::Mma::WarpCount::kN,
                                kSupportsBias,
                                kNumWarpsPerBlock>(
                                accum_o,
                                accum,
                                mi,
                                m_prime,
                                s_prime_tmp,
                                lane_id(),
                                thread_id(),
                                warp_id(),
                                warp_idx_n,
                                p.num_keys - iter_key_start,
                                iteratorC_tile_offset,
                                kSupportsBias ? 1.0f : p.scale);
                          }));
                    }));

#if !LAYOUT_CONVERT_ON_SHARED
      // convert accum from accum_t to scalar_t
      using Converter = NumericArrayConverterPPU<scalar_t, accum_t, MM0::Mma::FragmentC::kElements>;
      Converter convert_fp32_to_fp16;
      cutlass::Array<scalar_t, MM0::Mma::FragmentC::kElements> accum_scalar;
      accum_scalar = convert_fp32_to_fp16(accum);

      // compute 2nd gemm
      using AccumulatorLayout = cutlass::layout::ColumnMajor;
      using FragmentIteratorA =
          cutlass::gemm::warp::MmaTensorOpFragmentIterator<
              cutlass::MatrixShape<MM1::WarpShape::kM, MM1::InstructionShape::kK>, //warp shape
              cutlass::MatrixShape<MM1::WarpShape::kM, MM1::WarpShape::kN>, //accumulator shape
              MM1::WarpShape::kK, //kBlocksColumn
              scalar_t, scalar_t, AccumulatorLayout, typename MM1::InstructionShape, void>;

      FragmentIteratorA warp_tile_iterator_A(accum_scalar);

#else


      auto p_dropout_in_uint8_t = uint8_t(std::floor((1 - p.dropout_prob) * 255.0));

#if defined(HAS_PYTORCH) && !DROPOUT_ON_SHARED
      // Need col to be multiples of 16, since we're doing dropout with block of:
      //   PPU: 32 x 16, A100: 16 x 32
      static constexpr int DROPOUT_BLOCK_M = 32;
      static constexpr int DROPOUT_BLOCK_N = 16;
      static constexpr int DROPOUT_ROW_STRIDE = 1;
      int block_row_idx = (query_start + warp_idx_m * MM0::WarpShape::kM) / DROPOUT_BLOCK_M;
      int block_col_idx = (iter_key_start + warp_idx_n * MM0::WarpShape::kN) / DROPOUT_BLOCK_N;

      auto seed = std::get<0>(seeds);
      auto offset = std::get<1>(seeds) + p.dropout_batch_head_rng_offset * 32 + lane_id();
      if (kReturnSoftmax) {
        typename MM0::Mma::FragmentC accum_copy;
        #pragma unroll
        for (int i = 0; i < accum_copy.size(); i++) {
          accum_copy[i] = accum[i];
        }
        fmha::apply_dropout<MM0::WarpShape::kM, MM0::WarpShape::kN>(accum_copy, p_dropout_in_uint8_t,
            seed, offset, block_row_idx, block_col_idx, DROPOUT_ROW_STRIDE);

        // store accum_copy to global
        // auto query_start = blockIdx.x * kQueriesPerBlock;
        auto lane_offset = MM0::AccumLambdaIterator::get_lane_offset(
            lane_id(), warp_id(), iteratorC_tile_offset);
        MM0::AccumLambdaIterator::iterateRows(
            lane_offset,
            [&](int accum_m) {},
            [&](int accum_m, int accum_n, int idx) {
              int row = query_start + accum_m;
              int col = iter_key_start + accum_n;
              int s_index = row * p.num_keys_softmax + col;
              // if (thread_id() == 0 && blockIdx.x == 1)
              // printf("kReturnSoftmax, row = %d, col = %d, s_index = %d, p.num_queries = %d, p.num_keys_absolute = %d, accum=%f\n",
              //   row, col, s_index, p.num_queries, p.num_keys_absolute, accum_copy[idx]);
              if (accum_m < p.num_queries && col < p.num_keys_absolute) {
                s_dmask_ptr[s_index] = (scalar_t)accum_copy[idx];
              }
            },
            [&](int accum_m) {});
      }

      if (kSupportsDropout && p.use_dropout) {
        const float dropout_scale = 1.0f / (1.0f - p.dropout_prob);
        fmha::apply_dropout<MM0::WarpShape::kM, MM0::WarpShape::kN, fmha::DROPOUT_MODE_RESET_TO_0>(accum, p_dropout_in_uint8_t,
            seed, offset, block_row_idx, block_col_idx, DROPOUT_ROW_STRIDE,
            dropout_scale);
      }
#endif

      MM0::B2bGemm::accumToSmem(
          shared_storage.mainloop.si, accum, my_lane_id, {warp_idx_m, warp_idx_n});

#if defined(HAS_PYTORCH) && DROPOUT_ON_SHARED
      __syncthreads();
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
        auto si = shared_storage.mainloop.si.accum_ref();
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
                  int64_t(query_start + thread_i) * p.num_keys_absolute +
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
#endif // end of HAS_PYTORCH  and DROPOUT_ON_SHARED

#endif // end of LAYOUT_CONVERT_ON_SHARED

      if ((iter_key_start + kKeysPerBlock) < p.num_keys) {
        cutlass::arch::cp_async_wait<1>();
      } else {
        cutlass::arch::cp_async_wait<0>();
      }
      __syncthreads();

#if !ENABLE_AIU
      fmha::gemm_on_sAsB<typename MM1::Mma>(accum_o, warp_tile_iterator_A, warp_tile_iterator_V);
#else
      fmha::gemm_on_sAsB_aiuB<typename MM1::Mma>(accum_o, warp_tile_iterator_A, iterator_V, warp_tile_iterator_V);
#endif

      __syncthreads();
    } // end iter_key_start loop 

    if (kKeepOutputInRF) {
      constexpr bool kIsFirst = true;
      constexpr bool kIsLast = true;
      using DefaultEpilogue = typename MM1::DefaultEpilogue;
      using DefaultOp = typename MM1::EpilogueOutputOp;
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
              false, // IterationsUnroll
              typename MM1::OutputTileIteratorAccum // source tile
              >;
      auto dest_iter = createOutputIter(0);
      EpilogueOutputOp rescale(s_prime, m_prime);
      Epilogue epilogue(
          shared_storage.epilogue_shared_storage(),
          thread_id(),
          warp_id(),
          lane_id());
      epilogue(rescale, dest_iter, accum_o);
    }

    // 7. Calculate logsumexp
    // To make the backward easier, we pad logsumexp with `inf`
    // this avoids a few bound checks, and is not more expensive during fwd
    //static_assert(kQueriesPerBlock < kNumWarpsPerBlock * kWarpSize, "");
    if (p.logsumexp_ptr && thread_id() < kQueriesPerBlock) {
      constexpr float kLog2e = 1.4426950408889634074;
      auto lse_dim = ceil_div((int32_t)p.num_queries, kAlignLSE) * kAlignLSE;
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

template <typename AK, int MaxThreads = 512>
__launch_bounds__(MaxThreads)
__global__ void attention_kernel_batched_impl(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::attention_kernel(p);
}

template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
    attention_kernel_batched(typename AK::Params params);