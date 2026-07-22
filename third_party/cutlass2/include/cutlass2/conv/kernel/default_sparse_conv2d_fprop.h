/*
Copyright © 2026 Zhejiang Runhe Cloud Intelligence Technology Co., Ltd.
[TODO]: expand more license details.
*/
/*! \file
    \brief
    Default kernel-level implicit GEMM sparse convolution definitions combine threadblock-scoped
      matrix multiply-add with the appropriate threadblock-scoped epilogue.
*/
#pragma once

#include "cutlass2/cutlass2.h"
#include "cutlass2/conv/kernel/default_conv2d.h"

#include "cutlass2/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h"
#include "cutlass2/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h"
#include "cutlass2/gemm/threadblock/default_mma_core_sparse_sm80.h"
#include "cutlass2/transform/threadblock/predicated_tile_iterator.h"

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for SparseConv2dFprop
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename OperatorClass,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kAnalytic,
  conv::StrideSupport StrideSupport = StrideSupport::kStrided
#ifdef SAIL_CUSTOMIZE_CUTLASS
  /// Sparse is used for compression matrix.
  , gemm::Operand CompressOp_ = gemm::Operand::kA
#endif
> struct DefaultSparseConv2dFprop;

/// Defines a kernel for Sparse Conv2dFprop specialzation for Optimzed IteratorAlgorithm and
/// multistage pipeline for compression A.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag
>
struct DefaultSparseConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized
> {
  using MmaCore = typename cutlass::gemm::threadblock::DefaultSparseMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag
      >;

  static int const kSparse = MmaCore::kSparse;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK / kSparse>,
      ElementA,
      LayoutA,
      ThreadMapA
    >;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      LayoutB,
      ThreadMapB
    >;

  // Define iterators over tiles from the E operand
  using ElementE = typename MmaCore::ElementE;
  using LayoutE = typename MmaCore::GmemLayoutE;
  using ThreadMapE = typename MmaCore::IteratorThreadMapE;
  using AccessTypeE =
      cutlass::Array<ElementE, MmaCore::VectorSizeE / sizeof_bits<ElementE>::value>;
  using IteratorE =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM,
                               ThreadblockShape::kK / kSparse /
                                   MmaCore::kElementsPerElementE>,
          ElementE, LayoutE, 1, ThreadMapE, AccessTypeE>;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::SparseImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    typename MmaCore::SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    typename MmaCore::SmemIteratorB,
    arch::CacheOperation::Global,
    IteratorE,
    typename MmaCore::SmemIteratorE,
    MmaCore::kCacheOpE,
    MmaPolicy,
    Stages
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    1,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionSparse<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

#ifdef SAIL_CUSTOMIZE_CUTLASS
/// Defines a kernel for Sparse Conv2dFprop specialzation for Optimzed IteratorAlgorithm and
/// multistage pipeline for compression B.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag
>
struct DefaultSparseConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  arch::Sm80,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport::kStrided,
  gemm::Operand::kB
> {
  using MmaCore = typename cutlass::gemm::threadblock::DefaultSparseMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag, false, cutlass::arch::CacheOperation::Global, cutlass::arch::CacheOperation::Global, gemm::Operand::kB>;

  static int const kSparse = MmaCore::kSparse;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      LayoutA,
      ThreadMapA
    >;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK / kSparse, ThreadblockShape::kN>,
      ElementB,
      LayoutB,
      ThreadMapB
    >;

  // Define iterators over tiles from the E operand
  using ElementE = typename MmaCore::ElementE;
  using LayoutE = typename MmaCore::GmemLayoutE;
  using ThreadMapE = typename MmaCore::IteratorThreadMapE;
  using AccessTypeE =
      cutlass::Array<ElementE, MmaCore::VectorSizeE / sizeof_bits<ElementE>::value>;
  using IteratorE =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kK / kSparse / MmaCore::kElementsPerElementE, ThreadblockShape::kN>,
          ElementE, LayoutE, 0, ThreadMapE, AccessTypeE>;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::SparseImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    typename MmaCore::SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    typename MmaCore::SmemIteratorB,
    arch::CacheOperation::Global,
    IteratorE,
    typename MmaCore::SmemIteratorE,
    MmaCore::kCacheOpE,
    MmaPolicy,
    Stages,
    bool,
    gemm::Operand::kB
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    1,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolutionSparse<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop,
    Conv2dProblemSize,
    gemm::Operand::kB
  >;
};
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
