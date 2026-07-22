#pragma once

#include "cutlass2/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass2/aligned_buffer.h"
#include "cutlass2/utils.h"

#include "cutlass2/epilogue/thread/linear_combination.h"
#include "cutlass2/epilogue/thread/linear_combination_clamp.h"
#include "cutlass2/gemm/warp/default_mma_tensor_op.h"
#include "cutlass2/gemm/threadblock/default_mma.h"
#include "cutlass2/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "cutlass2/transform/pitch_linear_thread_map.h"
#include "aiu/gemm/threadblock/mma_base.h"
#include "aiu/gemm/tool/cutlass_type_convert.h"
#include "cutlass2/tfloat32.h"

#define HGGC_MMA_FRAGMENT_VALUE(wmmafrag, cutlassfragType)  (*(reinterpret_cast<cutlassfragType *>(&(wmmafrag))))

namespace aiu {
namespace gemm {
namespace threadblock {

template <typename T>
CUTLASS_DEVICE
void float_to_tf32(const T &src, T &dst) {}

template <>
CUTLASS_DEVICE
void float_to_tf32(const float &src, float &dst) {
#if SAIL_TYPE_CONVERT == 1 && (defined(__HGGC_ARCH__) || defined(__CUDA_ARCH__))
  #if defined(__HGGC_ARCH__)
    dst = nvcuda::wmma::__float_to_tf32_rna(src);
  #elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // only cuda-11.0 or later version support th__float_to_tf32is func
    // dst = nvcuda::wmma::__float_to_tf32(src);
    uint32_t& x = reinterpret_cast<uint32_t &>(dst);
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(x) : "f"(src));
  #elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
    uint32_t x = reinterpret_cast<uint32_t const &>(src);
    x += 0x1000u;
    dst = reinterpret_cast<float &>(x);
  #endif
#else
  uint32_t x = reinterpret_cast<uint32_t const &>(src);
  #if SAIL_TYPE_CONVERT == 2
    x += 0x1000u;
  #else
    if (std::isfinite(src)) {
      x += 0x1000u;
    }
  #endif
  #if !(defined(__HGGC_ARCH__) || defined(__CUDA_ARCH__))
    // host code
    x &= 0xffffe000u;
  #endif
  dst = reinterpret_cast<float &>(x);
#endif
}

template<typename Use, int M, int N, int K, typename T, typename Layout>
CUTLASS_DEVICE void frag_float_to_tf32(awmma::fragment<Use, M, N, K, T, Layout> &frag) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < frag.num_elements; i++) {
    float_to_tf32(frag.x[i], frag.x[i]);
  }
}

// convert half to cutlass::half_t
//         bfloat16 to cutlass::bfloat16_t
//         awmma::precision::tf32 to cutlass::bfloat16_t
template <typename Element> class ToCutlassTypeAIU {
public:
  using Element_if_tf32 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, awmma::precision::tf32>::value,
                                                    cutlass::tfloat32_t, Element>::type;
  using Element_if_bf16 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, nv_bfloat16>::value,
                                                    cutlass::bfloat16_t, Element_if_tf32>::type;
  using Element_if_fp16 = typename cutlass::platform::conditional<cutlass::platform::is_same<Element, half>::value,
                                                    cutlass::half_t, Element_if_bf16>::type;
  using type = Element_if_fp16;
};

/// default impl for stage >= 3
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
struct MmaMultistage {
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using PrefetchIterator = PrefetchIterator_;

  using ElementAccumulator = ElementAccumulator_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using Operator = Operator_;
  static cutlass::ComplexTransform const kTransformA = Operator::kTransformA;
  static cutlass::ComplexTransform const kTransformB = Operator::kTransformB;
  static int const kStages = Stages;

  using Shape = ThreadblockShape_;

  using WarpCount = cutlass::gemm::GemmShape<Shape::kM / WarpShape_::kM,
                                             Shape::kN / WarpShape_::kN,
                                             Shape::kK / WarpShape_::kK>;

  /// Fragment of accumulator tile
  using FragmentC = typename Operator_::FragmentC;

  using AccFragType = typename FromCutlassType<ElementAccumulator_>::type;

  using ElementA_ = typename FromCutlassType<typename IteratorA::Element>::type;

  using SharedStorage = typename aiu::gemm::threadblock::MmaBase<ThreadblockShape_,
                                                                 Operator_, Stages>::SharedStorage;
  using LayoutC = cutlass::layout::RowMajor;
  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape_, InstructionShape_,
    typename IteratorA::Element, typename IteratorA::Layout,
    typename IteratorB::Element, typename IteratorB::Layout,
    ElementAccumulator_,
    cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;
  using Policy = cutlass::gemm::threadblock::MmaPolicy<
    WarpMmaTensorOp,
    cutlass::MatrixShape<0, 0>,
    cutlass::MatrixShape<0, 0>,
    WarpCount::kK
  >;

  CUTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ) {
      tsm_astart_ = reinterpret_cast<ElementA_*>(shared_storage.operand_A.data());
      //tsm_bstart_ = reinterpret_cast<ElementA_*>(shared_storage.operand_B.data());
  }

  ElementA_* tsm_astart_;
  //ElementA_* tsm_bstart_;

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
      FragmentC const &src_accum) {
       operator()(gemm_k_iterations, accum, iterator_A, iterator_B, nullptr);
  }

  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA &iterator_A,
      ///< iterator over B operand in global memory
      IteratorB &iterator_B,
      ///< initial value of accumulator
      PrefetchIterator &iterator_Prefetch) {
        operator()(gemm_k_iterations, accum, iterator_A, iterator_B, &iterator_Prefetch);
  }
  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA &iterator_A,
      ///< iterator over B operand in global memory
      IteratorB &iterator_B,
      ///< initial value of accumulator
      PrefetchIterator *iterator_Prefetch) {

    constexpr int CHUNK_LENGTH_K = ThreadblockShape_::kK;
    constexpr int WARP_X_ITER = (WarpShape_::kM / InstructionShape_::kM);
    constexpr int WARP_Y_ITER = (WarpShape_::kN / InstructionShape_::kN);
    constexpr int B_TSM_OFFSET = (ThreadblockShape_::kM * CHUNK_LENGTH_K);
    constexpr int A_B_TSM_OFFSET = (ThreadblockShape_::kM * CHUNK_LENGTH_K + ThreadblockShape_::kN * CHUNK_LENGTH_K);
    constexpr int CHUNK_K = WarpShape_::kK / InstructionShape_::kK;

    ElementA_* tsm_bstart_ = tsm_astart_ + B_TSM_OFFSET;

    using accType = awmma::fragment<awmma::accumulator, InstructionShape_::kM, InstructionShape_::kN,
                    InstructionShape_::kK, AccFragType>;
    auto & fragAcc = *reinterpret_cast<accType (*)[WARP_Y_ITER][WARP_X_ITER]>(accum.data());

    // compuate fragAcc = fragA * fragB
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

      // prefetch for the Stages iteration
      #pragma unroll
      for (int stage = 0; stage < Stages; ++stage) {
        ElementA_* tsm_astart_staged = tsm_astart_ + stage * A_B_TSM_OFFSET;
        iterator_A.LoadToTsm(tsm_astart_staged);

        ElementA_* tsm_bstart_staged = tsm_bstart_ + stage * A_B_TSM_OFFSET;
        iterator_B.LoadToTsm(tsm_bstart_staged);

        __pipeline_commit();
      }

      __pipeline_wait_prior(Stages - 1);
      #if SAIL_PPU_MMA
        __ppu_barrier_sync_nocnt(0, 1);
      #else
        __syncthreads();
      #endif

      if (PrefetchIterator::kPrefetchStrategyType != cutlass::epilogue::PrefetchStrategyType::Kind::kNoPrefetchNeeded) {
        // Stage=4 normally means smaller block-tile, so memory footprint is not tight, then prefetched will mostly not evicted out.
        iterator_Prefetch->prefetch_all_to_llc();
      }

      // tsm -> vreg
      iterator_A.LoadToVreg(tsm_astart_, fragA, 0);
      iterator_B.LoadToVreg(tsm_bstart_, fragB, 0);

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

      for (int kChunk = 0; kChunk < gemm_k_size; kChunk += CHUNK_LENGTH_K) {
        int current_stage = (kChunk / CHUNK_LENGTH_K) % Stages;
        #pragma unroll
        for (int kIter = 0; kIter < CHUNK_K; kIter++) {
          if (true) {
            //always prefetch the fragments from tsm for the next kIter iteration
            int kIterNext = (kIter + 1) % CHUNK_K;

            ElementA_* tsm_astart_staged = tsm_astart_ + current_stage * A_B_TSM_OFFSET;
            iterator_A.LoadToVreg(tsm_astart_staged, fragA, kIterNext);

            ElementA_* tsm_bstart_staged = tsm_bstart_ + current_stage * A_B_TSM_OFFSET;
            iterator_B.LoadToVreg(tsm_bstart_staged, fragB, kIterNext);
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

          if (kIter == CHUNK_K - 2 && kChunk < gemm_k_size - CHUNK_LENGTH_K) {
            // prefetch the CHUNK from vmem to tsm
            // before tsm -> vreg prefetch in the next iteration
            // for mma in the next next iteration
            __pipeline_wait_prior(Stages - 2);
            #if SAIL_PPU_MMA
              __ppu_barrier_sync_nocnt(0, 1);
            #else
              __syncthreads();
            #endif
            if (kChunk < gemm_k_size - 2 * CHUNK_LENGTH_K) {
              ElementA_* tsm_astart_staged = tsm_astart_ + current_stage * A_B_TSM_OFFSET;
              iterator_A.LoadToTsm(tsm_astart_staged);

              ElementA_* tsm_bstart_staged = tsm_bstart_ + current_stage * A_B_TSM_OFFSET;
              iterator_B.LoadToTsm(tsm_bstart_staged);

              __pipeline_commit();
            }

            // update current_stage
            current_stage = (current_stage + 1) % Stages;
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

/// Partial specialization for stage=2
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
    /// Operation perfomed by GEMM
    typename Operator_,
    /// Wmma Fragment type for A and B (for FP32 tensorcore on PPU)
    typename WmmaFragABType>
struct MmaMultistage<IteratorA_, IteratorB_, PrefetchIterator_, ElementAccumulator_, ArchTag_, ThreadblockShape_,
                     WarpShape_, InstructionShape_, 2, Operator_, WmmaFragABType> {
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using PrefetchIterator = PrefetchIterator_;
  using ElementAccumulator = ElementAccumulator_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using Operator = Operator_;
  static cutlass::ComplexTransform const kTransformA = Operator::kTransformA;
  static cutlass::ComplexTransform const kTransformB = Operator::kTransformB;
  static int const kStages = 2;

  using Shape = ThreadblockShape_;

  using WarpCount = cutlass::gemm::GemmShape<Shape::kM / WarpShape_::kM,
                                             Shape::kN / WarpShape_::kN,
                                             Shape::kK / WarpShape_::kK>;

  /// Fragment of accumulator tile
  using FragmentC = typename Operator_::FragmentC;

  using AccFragType = typename FromCutlassType<ElementAccumulator_>::type;

  using ElementA_ = typename FromCutlassType<typename IteratorA::Element>::type;

  using SharedStorage = typename aiu::gemm::threadblock::MmaBase<ThreadblockShape_,
                                                                 Operator_, 2>::SharedStorage;
  using LayoutC = cutlass::layout::RowMajor;
  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape_, InstructionShape_,
    typename IteratorA::Element, typename IteratorA::Layout,
    typename IteratorB::Element, typename IteratorB::Layout,
    ElementAccumulator_,
    cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;
  using Policy = cutlass::gemm::threadblock::MmaPolicy<
    WarpMmaTensorOp,
    cutlass::MatrixShape<0, 0>,
    cutlass::MatrixShape<0, 0>,
    WarpCount::kK
  >;

  CUTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ) {
      tsm_astart_ = reinterpret_cast<ElementA_*>(shared_storage.operand_A.data());
      //tsm_bstart_ = reinterpret_cast<ElementA_*>(shared_storage.operand_B.data());
  }

  ElementA_* tsm_astart_;
  //ElementA_* tsm_bstart_;

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
      FragmentC const &src_accum) {
       operator()(gemm_k_iterations, accum, iterator_A, iterator_B, nullptr);
  }

  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA &iterator_A,
      ///< iterator over B operand in global memory
      IteratorB &iterator_B,
      ///< initial value of accumulator
      PrefetchIterator &iterator_Prefetch) {
        operator()(gemm_k_iterations, accum, iterator_A, iterator_B, &iterator_Prefetch);
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
      PrefetchIterator *iterator_Prefetch) {

    constexpr int CHUNK_LENGTH_K = ThreadblockShape_::kK;
    constexpr int WARP_X_ITER = (WarpShape_::kM / InstructionShape_::kM);
    constexpr int WARP_Y_ITER = (WarpShape_::kN / InstructionShape_::kN);
    constexpr int B_TSM_OFFSET = (ThreadblockShape_::kM * CHUNK_LENGTH_K);
    constexpr int A_B_TSM_OFFSET = (ThreadblockShape_::kM * CHUNK_LENGTH_K + ThreadblockShape_::kN * CHUNK_LENGTH_K);
    constexpr int CHUNK_K = WarpShape_::kK / InstructionShape_::kK;


    ElementA_* tsm_bstart_ = tsm_astart_ + B_TSM_OFFSET;

    using accType = awmma::fragment<awmma::accumulator, InstructionShape_::kM, InstructionShape_::kN,
                    InstructionShape_::kK, AccFragType>;
    auto & fragAcc = *reinterpret_cast<accType (*)[WARP_Y_ITER][WARP_X_ITER]>(accum.data());

    int iter = 0;

    // compuate fragAcc = fragA * fragB
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

      {
        // vmem -> tsm
        iterator_A.LoadToTsm(tsm_astart_);
        iterator_B.LoadToTsm(tsm_bstart_);
        __pipeline_commit();

        if (gemm_k_size > CHUNK_LENGTH_K) {
          // another around of vmem -> tsm, we don't wait for it right away
          iterator_A.LoadToTsm(tsm_astart_ + A_B_TSM_OFFSET);
          iterator_B.LoadToTsm(tsm_bstart_ + A_B_TSM_OFFSET);
          __pipeline_commit();
          __pipeline_wait_prior(1);
        } else {
          __pipeline_wait_prior(0);
        }

        #if SAIL_PPU_MMA
          __ppu_barrier_sync_nocnt(0, 1);
        #else
          __syncthreads();
        #endif

        // tsm -> vreg
        iterator_A.LoadToVreg(tsm_astart_, fragA, 0);
        iterator_B.LoadToVreg(tsm_bstart_, fragB, 0);
      }

      if (PrefetchIterator::kPrefetchStrategyType == cutlass::epilogue::PrefetchStrategyType::Kind::kLinearPrefetchLLCSingle) {
        iterator_Prefetch->prefetch_all_to_llc();
      }

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

      for (int kChunk = 0; kChunk < gemm_k_size; kChunk += CHUNK_LENGTH_K) {

        if (PrefetchIterator::kPrefetchStrategyType == cutlass::epilogue::PrefetchStrategyType::Kind::kStridedPrefetchLLCStream) {
          bool start_prefetch = (kChunk + (iterator_Prefetch->get_iter_num() * CHUNK_LENGTH_K)) >= gemm_k_size;
          if (start_prefetch) {
            iterator_Prefetch->prefetch_to_llc(iter++);
          }
        }

        #pragma unroll
        for (int kIter = 0; kIter < CHUNK_K; kIter++) {

          if (true) {
            //always prefetch the fragments from tsm for the next kIter iteration
            int kIterNext = (kIter + 1) % CHUNK_K;

            ElementA_* pSrc = reinterpret_cast<ElementA_*>(tsm_astart_);
            if (((kChunk + (kIter + 1) * InstructionShape_::kK) / CHUNK_LENGTH_K) & 0x1u) {
              pSrc += A_B_TSM_OFFSET;
            }

            iterator_A.LoadToVreg(pSrc, fragA, kIterNext);
            iterator_B.LoadToVreg(pSrc + B_TSM_OFFSET, fragB, kIterNext);
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

          if (kIter == CHUNK_K - 2) {
            __pipeline_wait_prior(0);
            #if SAIL_PPU_MMA
              __ppu_barrier_sync_nocnt(0, 1);
            #else
              __syncthreads();
            #endif
            if (kChunk < gemm_k_size - 2 * CHUNK_LENGTH_K) {
              ElementA_* pDst = reinterpret_cast<ElementA_*>(tsm_astart_);
              if ((kChunk / CHUNK_LENGTH_K) & 0x1u) {
                pDst += A_B_TSM_OFFSET;
              }
              iterator_A.LoadToTsm(pDst);
              iterator_B.LoadToTsm(pDst + B_TSM_OFFSET);
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
} // namespace aiu