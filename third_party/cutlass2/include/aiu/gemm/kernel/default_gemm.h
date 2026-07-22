#pragma once

#include "aiu/gemm/threadblock/default_mma.h"
#include "cutlass2/gemm/kernel/gemm.h"


namespace aiu {
namespace gemm {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Wmma Fragment type for A and B (for FP32 tensorcore on PPU)
    typename WmmaFragABType = typename ToTF32<ElementA_>::type,
    /// Option to enable or disable cutlass epilogue for flexibility
    bool EnableCutlassEpilogue = true>
struct DefaultGemm;

/// Partial specialization for PPU-1.0 Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator_,
    /// Wmma Fragment type for A and B (for FP32 tensorcore on PPU)
    typename WmmaFragABType,
    /// Option to enable or disable cutlass epilogue for flexibility
    bool EnableCutlassEpilogue>
struct DefaultGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC,
                   cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                   ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,
                   ThreadblockSwizzle, Stages, SplitKSerial, Operator_, WmmaFragABType,
                   EnableCutlassEpilogue> {

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  using Operator = Operator_;
  using ArchTag = cutlass::arch::Sm80;

  using WarpMmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape, InstructionShape, ElementA, LayoutA, ElementB, LayoutB, ElementAccumulator,
    cutlass::layout::RowMajor, cutlass::arch::OpMultiplyAdd>::Type;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, WarpMmaTensorOp, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount>::Epilogue;

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename aiu::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, ElementB, LayoutB,
      ElementAccumulator, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape, Stages,
      Operator, Epilogue, WmmaFragABType>::ThreadblockMma;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = cutlass::gemm::kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace aiu