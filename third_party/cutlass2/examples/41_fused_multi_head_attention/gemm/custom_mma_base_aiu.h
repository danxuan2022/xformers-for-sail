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

#include "cutlass2/aligned_buffer.h"
#include "cutlass2/arch/memory.h"
#include "cutlass2/array.h"
#include "cutlass2/cutlass2.h"
#include "cutlass2/gemm/gemm.h"
#include "cutlass2/gemm/threadblock/mma_base.h"
#include "cutlass2/matrix_shape.h"
#include "cutlass2/numeric_types.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Operator,
    /// Number of stages,
    int Stages,
    /// Warp-level tile size (for WarpCount computation)
    typename WarpShape_ = Shape_,
    /// Used for partial specialization
    typename Enable = bool>
class CustomMmaBaseAiu {
 public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Number of stages
  static int const kStages = Stages;

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<
      Shape_::kM / WarpShape_::kM,
      Shape_::kN / WarpShape_::kN,
      Shape_::kK / WarpShape_::kK>;

  /// Shared storage object needed by threadblock-scoped GEMM
  template <typename Element, typename OperandShape, typename OperandLayout>
  struct OperandSharedStorage {
    AlignedBuffer<Element, OperandShape::kCount> buffer;
    using TensorRef = TensorRef<Element, OperandLayout>;

    CUTLASS_DEVICE
    static OperandLayout Layout() {
      return OperandLayout::packed({OperandShape::kRow, OperandShape::kColumn});
    }

    /// Returns a TensorRef to the operand
    CUTLASS_HOST_DEVICE
    TensorRef ref() {
      return TensorRef{buffer.data(), Layout()};
    }
  };

  /// Shape of the A matrix operand in shared memory
  using ShapeA = MatrixShape<Shape::kM, Shape::kK * kStages>;

  /// Shape of the B matrix operand in shared memory
  using ShapeB = MatrixShape<Shape::kK * kStages, Shape::kN>;

  using SharedStorageA = OperandSharedStorage<
      typename Operator::ElementA,
      ShapeA,
      typename Operator::LayoutA>;
  using SharedStorageB = OperandSharedStorage<
      typename Operator::ElementB,
      ShapeB,
      typename Operator::LayoutB>;
  using TensorRefA = typename SharedStorageA::TensorRef;
  using TensorRefB = typename SharedStorageB::TensorRef;

  struct SharedStorage {
    /// Buffer for A operand
    SharedStorageA operand_A;

    /// Buffer for B operand
    SharedStorageB operand_B;
  };
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
