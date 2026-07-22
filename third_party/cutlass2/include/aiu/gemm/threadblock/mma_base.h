#pragma once

#include "cutlass2/aligned_buffer.h"
#include "cutlass2/arch/memory.h"
#include "cutlass2/array.h"
#include "cutlass2/cutlass2.h"
#include "cutlass2/gemm/gemm.h"
#include "cutlass2/matrix_shape.h"
#include "cutlass2/numeric_types.h"
////////////////////////////////////////////////////////////////////////////////

namespace aiu {
namespace gemm {
namespace threadblock {

template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Operator,
    /// Number of stages,
    int Stages,
    /// Used for partial specialization
    typename Enable = bool>
class MmaBase {
public:

  /// Tensor reference to the A operand
  using TensorRefA = cutlass::TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

  /// Tensor reference to the B operand
  using TensorRefB = cutlass::TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;

  /// Shared storage object needed by threadblock-scoped GEMM
  class SharedStorage {
   public:
    //
    // Type definitions
    //

    /// Shape of the A matrix operand in shared memory
    using ShapeA = cutlass::MatrixShape<Shape::kM, Shape::kK * Stages>;

    /// Shape of the B matrix operand in shared memory
    using ShapeB = cutlass::MatrixShape<Shape::kK * Stages, Shape::kN>;

   public:
    //
    // Data members
    //

    /// Buffer for A operand
    cutlass::AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

    /// Buffer for B operand
    cutlass::AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

   public:

    //
    // Methods
    //

    /// Returns a layout object for the A matrix
    CUTLASS_DEVICE
    static typename Operator::LayoutA LayoutA() {
      return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }

    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_ref() {
      return TensorRefA{operand_A.data(), LayoutA()};
    }

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_ref() {
      return TensorRefB{operand_B.data(), LayoutB()};
    }
  };

};

} // namespace threadblock
} // namespace gemm
} // namespace aiu