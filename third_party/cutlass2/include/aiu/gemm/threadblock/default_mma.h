#pragma once

#include "cutlass2/cutlass2.h"
#include "cutlass2/numeric_types.h"
#include "cutlass2/arch/arch.h"
#include "cutlass2/arch/wmma.h"

#include "cutlass2/layout/matrix.h"

//#if defined(CUTLASS_ARCH_WMMA_ENABLED)
//#include "cutlass2/gemm/threadblock/default_mma_core_wmma.h"
//#endif //CUTLASS_ARCH_WMMA_ENABLED

#include "aiu/gemm/threadblock/gemm_aiu_loader_a.h"
#include "aiu/gemm/threadblock/gemm_aiu_loader_b.h"
#include "aiu/gemm/threadblock/mma_multistage.h"

////////////////////////////////////////////////////////////////////////////////

namespace aiu {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Epilogue type
    typename Epilogue = void,
    /// Wmma Fragment type for A and B (for FP32 tensorcore on PPU)
    typename WmmaFragABType = ElementA_,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false
    >
struct DefaultMma;

/// Specialization for row-major output with prefetch (OperatorClass TensorOp)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Epilogue type
    typename Epilogue,
    /// Wmma Fragment type for A and B (for FP32 tensorcore on PPU)
    typename WmmaFragABType
    >
struct DefaultMma<ElementA, LayoutA, ElementB, LayoutB,
                  ElementAccumulator, cutlass::layout::RowMajor,
                  cutlass::arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, Epilogue, WmmaFragABType, false> {
  using MmaShape = InstructionShape;
  using WmmaLayoutA =
      typename cutlass::platform::conditional<cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value,
                                awmma::row_major, awmma::col_major>::type;
  using WmmaLayoutB =
      typename cutlass::platform::conditional<cutlass::platform::is_same<LayoutB, cutlass::layout::RowMajor>::value,
                                awmma::row_major, awmma::col_major>::type;

  using WmmaFragABType_ = typename FromCutlassType<WmmaFragABType>::type;

  using FragAType = awmma::fragment<awmma::matrix_a, MmaShape::kM, MmaShape::kN, MmaShape::kK,
                                    WmmaFragABType_, WmmaLayoutA>;
  using FragBType = awmma::fragment<awmma::matrix_b, MmaShape::kM, MmaShape::kN, MmaShape::kK,
                                    WmmaFragABType_, WmmaLayoutB>;

  using IteratorA = aiu::gemm::threadblock::AiuLoaderA<ElementA, LayoutA, ThreadblockShape,
                                                       WarpShape, InstructionShape, FragAType>;

  using IteratorB = aiu::gemm::threadblock::AiuLoaderB<ElementB, LayoutB, ThreadblockShape,
                                                       WarpShape, InstructionShape, FragBType>;

  using IteratorC = typename Epilogue::OutputTileIterator;
  using PrefetchIterator = typename cutlass::epilogue::threadblock::PrefetchTileIterator<typename IteratorC::ThreadMap,
                                                                                         typename IteratorC::Element,
                                                                                         ThreadblockShape,
                                                                                         cutlass::epilogue::GetPrefetchStrategy<typename Epilogue::OutputOp>::value>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB,
      ElementAccumulator, cutlass::layout::RowMajor, Operator, ThreadblockShape::kK / WarpShape::kK
      >::Type;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = aiu::gemm::threadblock::MmaMultistage<
      IteratorA, IteratorB, PrefetchIterator, ElementAccumulator, ArchTag,
      ThreadblockShape, WarpShape, InstructionShape, Stages, MmaTensorOp, WmmaFragABType_>;
};

/// Specialization for row-major output without prefetch (OperatorClass TensorOp)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Wmma Fragment type for A and B (for FP32 tensorcore on PPU)
    typename WmmaFragABType
    >
struct DefaultMma<ElementA, LayoutA, ElementB, LayoutB,
                  ElementAccumulator, cutlass::layout::RowMajor,
                  cutlass::arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, void, WmmaFragABType, false> {
  using MmaShape = InstructionShape;
  using WmmaLayoutA =
      typename cutlass::platform::conditional<cutlass::platform::is_same<LayoutA, cutlass::layout::RowMajor>::value,
                                awmma::row_major, awmma::col_major>::type;
  using WmmaLayoutB =
      typename cutlass::platform::conditional<cutlass::platform::is_same<LayoutB, cutlass::layout::RowMajor>::value,
                                awmma::row_major, awmma::col_major>::type;

  using WmmaFragABType_ = typename FromCutlassType<WmmaFragABType>::type;

  using FragAType = awmma::fragment<awmma::matrix_a, MmaShape::kM, MmaShape::kN, MmaShape::kK,
                                    WmmaFragABType_, WmmaLayoutA>;
  using FragBType = awmma::fragment<awmma::matrix_b, MmaShape::kM, MmaShape::kN, MmaShape::kK,
                                    WmmaFragABType_, WmmaLayoutB>;

  using IteratorA = aiu::gemm::threadblock::AiuLoaderA<ElementA, LayoutA, ThreadblockShape,
                                                       WarpShape, InstructionShape, FragAType>;

  using IteratorB = aiu::gemm::threadblock::AiuLoaderB<ElementB, LayoutB, ThreadblockShape,
                                                       WarpShape, InstructionShape, FragBType>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementAccumulator,
    128 / cutlass::sizeof_bits<ElementAccumulator>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator,
    cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, WarpMmaTensorOp, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount>::Epilogue;

  using IteratorC = typename Epilogue::OutputTileIterator;
  using PrefetchIterator = typename cutlass::epilogue::threadblock::PrefetchTileIterator<typename IteratorC::ThreadMap,
                                                                                         typename IteratorC::Element,
                                                                                         ThreadblockShape,
                                                                                         cutlass::epilogue::GetPrefetchStrategy<typename Epilogue::OutputOp>::value>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB,
      ElementAccumulator, cutlass::layout::RowMajor, Operator, ThreadblockShape::kK / WarpShape::kK
      >::Type;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = aiu::gemm::threadblock::MmaMultistage<
      IteratorA, IteratorB, PrefetchIterator, ElementAccumulator, ArchTag,
      ThreadblockShape, WarpShape, InstructionShape, Stages, MmaTensorOp, WmmaFragABType_>;
};

} // namespace threadblock
} // namespace gemm
} // namespace aiu