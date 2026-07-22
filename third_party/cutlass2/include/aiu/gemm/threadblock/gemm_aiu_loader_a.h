#pragma once

#include "aiu/gemm/threadblock/aiu_api.h"
#include "cutlass2/layout/layout.h"
#include "cutlass2/tensor_coord.h"
#include "cutlass2/tensor_ref.h"


namespace aiu {
namespace gemm {
namespace threadblock {

// ElementA is match with cutlass::half
// ElementA_ is match with __half

/// default impl for RowMajor
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for internal accumulation
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename MmaShape_,
    typename FragType_>
struct AiuLoaderA :
  public AiuLoaderBase<ElementA, LayoutA_> {

  using Base = AiuLoaderBase<ElementA, LayoutA_>;
  using ElementA_ = typename Base::Element_Internal;
  using AccessType = typename Base::AccessType;

  constexpr static int CUB_W = ThreadblockShape_::kM;
  constexpr static int CUB_Z_MAX = 128 / sizeof(ElementA_);
  constexpr static int CUB_Z = ThreadblockShape_::kK > CUB_Z_MAX ? CUB_Z_MAX : ThreadblockShape_::kK;
  constexpr static int CUB_Z_COUNT = (ThreadblockShape_::kK + CUB_Z - 1) / CUB_Z;

  constexpr static int WARP_PER_BLOCK_X = ThreadblockShape_::kM / WarpShape_::kM;
  constexpr static int WARP_X_ITER = WarpShape_::kM / MmaShape_::kM;
  constexpr static int CHUNK_K = WarpShape_::kK / MmaShape_::kK;

  using FragType = FragType_[WARP_X_ITER];

  CubeCoord dim_;
  CubeCoord offset_;
  CubeCoord tsmcube_;

  int warp_idx_;
  int warpOffset_;

  const ElementA_* vmem_start_;

  CUTLASS_HOST_DEVICE AiuLoaderA(
      /// Precomputed parameters object
      typename Base::Params const &params,
      /// Pointer to start of tensor
      const ElementA* vmem_start,
      /// Extent of tensor
      cutlass::MatrixCoord const &extent,
      /// ID of each participating warp
      int thread_idx,
      /// Initial offset of threadblock
      cutlass::MatrixCoord const &block_offset):
        dim_({1, extent.row(), params.layout_.stride(0)}),
        offset_({0, block_offset.row(), block_offset.column()}),
        tsmcube_({1, CUB_W, CUB_Z}) {
    int aiu_offset = params.layout_.stride(0) - extent.column();
    if (params.layout_.stride(0) == 0) {
      dim_.z = extent.column();
      aiu_offset = 0;
    }
    vmem_start_ = reinterpret_cast<const ElementA_*>(vmem_start) - aiu_offset;
    warp_idx_ = __ppu_read_firstlane(thread_idx >> 5);
    warpOffset_ = (warp_idx_ % WARP_PER_BLOCK_X) * WARP_X_ITER;
    offset_.z += aiu_offset;
  }

  CUTLASS_DEVICE void add_pointer_offset(int64_t vmem_offset) {
    vmem_start_ += vmem_offset;
  }

  CUTLASS_DEVICE void LoadToTsm(ElementA_* tsm_start) {
    if (warp_idx_ == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < CUB_Z_COUNT; ++i) {
        int residual_z = ThreadblockShape_::kK - i * CUB_Z;
        int cub_z = (residual_z > CUB_Z_MAX) ? CUB_Z_MAX : residual_z;
        CubeCoord cub_size {tsmcube_.x, tsmcube_.y, cub_z};
        TsmLoadCube(tsm_start, vmem_start_, dim_, offset_, cub_size);
        IncreaseVmemOffset(cub_z);
        tsm_start += CUB_W * CUB_Z;
      }
    }
  }

  CUTLASS_DEVICE void LoadToVreg(ElementA_* tsm_start,
                            FragType_ frag[WARP_X_ITER][2],
                            int kIter) {
    CUTLASS_PRAGMA_UNROLL
    for (uint xIter = 0; xIter < WARP_X_ITER; xIter++) {
      LoadMatrixSyncSwizzle<CUB_W, MmaShape_::kM>(tsm_start + kIter * CUB_W * MmaShape_::kK,
          frag[xIter][kIter & 0x1], warpOffset_ + xIter, kIter);
    }
  }

  CUTLASS_DEVICE void LoadToVreg(ElementA_* tsm_start,
                            FragType_ frag[WARP_X_ITER],
                            int kIter) {
    CUTLASS_PRAGMA_UNROLL
    for (uint xIter = 0; xIter < WARP_X_ITER; xIter++) {
      LoadMatrixSyncSwizzle<CUB_W, MmaShape_::kM>(tsm_start + kIter * CUB_W * MmaShape_::kK,
          frag[xIter], warpOffset_ + xIter, kIter & 0x3);
    }
  }

  CUTLASS_DEVICE void IncreaseVmemOffset(int cub_z) {
    offset_.z += cub_z;
  }

  CUTLASS_DEVICE void add_tile_offset(cutlass::MatrixCoord const& offset) {
    offset_.y += offset.row() * MmaShape_::kM;
    offset_.z += offset.column() * MmaShape_::kK;
  }
};


/// Partial specialization for ColumnMajor
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Element type for internal accumulation
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename MmaShape_,
    typename FragType_>
struct AiuLoaderA <ElementA, cutlass::layout::ColumnMajor,
                   ThreadblockShape_, WarpShape_, MmaShape_,
                   FragType_> : 
                   public AiuLoaderBase <ElementA, cutlass::layout::ColumnMajor> {

  using Base = AiuLoaderBase<ElementA, cutlass::layout::ColumnMajor>;
  using ElementA_ = typename Base::Element_Internal;
  using AccessType = typename Base::AccessType;

  constexpr static int CUB_W = ThreadblockShape_::kK;
  constexpr static int CUB_Z_MAX = 128 / sizeof(ElementA_);
  constexpr static int CUB_Z = ThreadblockShape_::kM > CUB_Z_MAX ? CUB_Z_MAX : ThreadblockShape_::kM;
  constexpr static int WARP_PER_BLOCK_X = ThreadblockShape_::kM / WarpShape_::kM;
  constexpr static int WARP_X_ITER = WarpShape_::kM / MmaShape_::kM;
  constexpr static int CHUNK_K = WarpShape_::kK / MmaShape_::kK;
  constexpr static int CUB_COUNT = ThreadblockShape_::kM / CUB_Z;
  constexpr static int WARP_NUM = (ThreadblockShape_::kM / WarpShape_::kM) * (ThreadblockShape_::kN / WarpShape_::kN)
                                  * (ThreadblockShape_::kK / WarpShape_::kK);

  using FragType = FragType_[WARP_X_ITER];

  CubeCoord dim_;
  CubeCoord offset_;
  CubeCoord tsmcube_;

  int warp_idx_;
  int warpOffset_;

  const ElementA_* vmem_start_;

  CUTLASS_HOST_DEVICE AiuLoaderA(
      /// Precomputed parameters object
      typename Base::Params const &params,
      /// Pointer to start of tensor
      const ElementA *vmem_start,
      /// Extent of tensor
      cutlass::MatrixCoord const &extent,
      /// ID of each participating warp
      int thread_idx,
      /// Initial offset of threadblock
      cutlass::MatrixCoord const &block_offset)
      : dim_({1, extent.column(), params.layout_.stride(0)}),
        offset_({0, block_offset.column(), block_offset.row()}),
        tsmcube_({1, CUB_W, CUB_Z}) {
    int aiu_offset = params.layout_.stride(0) - extent.row();
    if (params.layout_.stride(0) == 0) {
      dim_.z = extent.row();
      aiu_offset = 0;
    }
    vmem_start_ = reinterpret_cast<const ElementA_*>(vmem_start) - aiu_offset;
    warp_idx_ = __ppu_read_firstlane(thread_idx >> 5);
    warpOffset_ = (warp_idx_ % WARP_PER_BLOCK_X) * WARP_X_ITER;
    offset_.z += aiu_offset;
  }

  CUTLASS_DEVICE void add_pointer_offset(int64_t vmem_offset) {
    vmem_start_ += vmem_offset;
  }

  CUTLASS_DEVICE void LoadToTsm(ElementA_* tsm_start) {
    int cub = warp_idx_;
    CUTLASS_PRAGMA_UNROLL
    for (int loop = 0; loop < CUB_COUNT / WARP_NUM; ++loop) {
      CubeCoord cub_offset = CubeCoord{offset_.x, offset_.y, offset_.z + cub * CUB_Z};
      TsmLoadCube(tsm_start + cub * CUB_W * CUB_Z, vmem_start_, dim_, cub_offset, tsmcube_);
      cub += WARP_NUM;
    }
    if ((CUB_COUNT % WARP_NUM) && cub < CUB_COUNT) {
      CubeCoord cub_offset = CubeCoord{offset_.x, offset_.y, offset_.z + cub * CUB_Z};
      TsmLoadCube(tsm_start + cub * CUB_W * CUB_Z, vmem_start_, dim_, cub_offset, tsmcube_);
    }
    IncreaseVmemOffset();
  }

  CUTLASS_DEVICE void LoadToVreg(ElementA_* tsm_start,
                            FragType_ frag[WARP_X_ITER][2],
                            int kIter) {
    constexpr uint sliceStride = CUB_W * MmaShape_::kM;
    constexpr uint step = sizeof(ElementA_) / sizeof(half);

    CUTLASS_PRAGMA_UNROLL
    for (uint xIter = 0; xIter < WARP_X_ITER; xIter++) {
      LoadMatrixSyncSwizzle<CUB_W, MmaShape_::kK>(tsm_start + (warpOffset_ + xIter) * sliceStride,
          frag[xIter][kIter & 0x1u], kIter, ((warpOffset_ + xIter) * step) & 0x3);
    }
  }

  CUTLASS_DEVICE void LoadToVreg(ElementA_* tsm_start,
                            FragType_ frag[WARP_X_ITER],
                            int kIter) {
    constexpr uint sliceStride = CUB_W * MmaShape_::kM;
    constexpr uint step = sizeof(ElementA_) / sizeof(half);

    CUTLASS_PRAGMA_UNROLL
    for (uint xIter = 0; xIter < WARP_X_ITER; xIter++) {
      LoadMatrixSyncSwizzle<CUB_W, MmaShape_::kK>(tsm_start + (warpOffset_ + xIter) * sliceStride,
          frag[xIter], kIter, ((warpOffset_ + xIter) * step) & 0x3);
    }
  }

  CUTLASS_DEVICE void IncreaseVmemOffset() {
    offset_.y += ThreadblockShape_::kK;
  }

  CUTLASS_DEVICE void add_tile_offset(cutlass::MatrixCoord const& offset) {
    offset_.z += offset.row() * MmaShape_::kM;
    offset_.y += offset.column() * MmaShape_::kK;
  }

};


} // namespace threadblock
} // namespace gemm
} // namespace aiu