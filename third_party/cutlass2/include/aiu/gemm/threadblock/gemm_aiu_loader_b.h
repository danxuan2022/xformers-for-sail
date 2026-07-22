#pragma once

#include "aiu/gemm/threadblock/aiu_api.h"
#include "cutlass2/layout/matrix.h"
#include "cutlass2/matrix_shape.h"
#include "cutlass2/tensor_coord.h"
#include "cutlass2/tensor_ref.h"

namespace aiu {
namespace gemm {
namespace threadblock {

/// default impl for ColumnMajor
template <
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for internal accumulation
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename MmaShape_,
    typename FragType_>
struct AiuLoaderB :
  public AiuLoaderBase<ElementB, LayoutB_> {

  using Base = AiuLoaderBase<ElementB, LayoutB_>;
  using ElementB_ = typename Base::Element_Internal;
  using AccessType = typename Base::AccessType;

  using Shape = cutlass::MatrixShape<ThreadblockShape_::kK, ThreadblockShape_::kN>;

  constexpr static int CUB_W = ThreadblockShape_::kN;
  constexpr static int CUB_Z_MAX = 128 / sizeof(ElementB_);
  constexpr static int CUB_Z = ThreadblockShape_::kK > CUB_Z_MAX ? CUB_Z_MAX : ThreadblockShape_::kK;
  constexpr static int CUB_Z_COUNT = (ThreadblockShape_::kK + CUB_Z - 1) / CUB_Z;
  constexpr static int WARP_PER_BLOCK = (ThreadblockShape_::kM / WarpShape_::kM)
                                      * (ThreadblockShape_::kN / WarpShape_::kN);
  constexpr static int WARP_PER_BLOCK_X = ThreadblockShape_::kM / WarpShape_::kM;
  constexpr static int WARP_Y_ITER = (WarpShape_::kN / MmaShape_::kN);
  constexpr static int CHUNK_K = WarpShape_::kK / MmaShape_::kK;

  using TransFragType_ = typename awmma::fragment<awmma::matrix_b, MmaShape_::kM, MmaShape_::kN, MmaShape_::kK,
                                         ElementB_, typename Base::WmmaLayoutTrans>;
  using TransFragType = TransFragType_[WARP_Y_ITER];
  using FragType = FragType_[WARP_Y_ITER];

  CubeCoord dim_;
  CubeCoord offset_;
  CubeCoord tsmcube_;

  int warp_idx_;
  int warpOffset_;
  const ElementB_* vmem_start_;

  CUTLASS_HOST_DEVICE AiuLoaderB(
      /// Precomputed parameters object
      typename Base::Params const &params,
      /// Pointer to start of tensor
      const ElementB* vmem_start,
      /// Extent of tensor
      cutlass::MatrixCoord const &extent,
      /// ID of each participating warp
      int thread_idx,
      /// Initial offset of threadblock
      cutlass::MatrixCoord const &block_offset):
        dim_({1, extent.column(), params.layout_.stride(0)}),
        offset_({0, block_offset.column(), block_offset.row()}),
        tsmcube_({1, CUB_W, CUB_Z}) {
    int aiu_offset = params.layout_.stride(0) - extent.row();
    if (params.layout_.stride(0) == 0) {
      dim_.z = extent.row();
      aiu_offset = 0;
    }
    vmem_start_ = reinterpret_cast<const ElementB_*>(vmem_start) - aiu_offset;
    warp_idx_ = __ppu_read_firstlane(thread_idx >> 5);
    warpOffset_ = (warp_idx_ / WARP_PER_BLOCK_X) * WARP_Y_ITER;
    offset_.z += aiu_offset;
  }

  CUTLASS_DEVICE void add_pointer_offset(int64_t vmem_offset) {
    vmem_start_ += vmem_offset;
  }

  CUTLASS_DEVICE void LoadToTsm(ElementB_* tsm_start) {
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

  CUTLASS_DEVICE void LoadToVreg(ElementB_* tsm_start,
                            FragType_ frag[WARP_Y_ITER][2],
                            int kIter) {
    // slice_id = kIter
    // sliceStride = CUB_W * MmaShape_::kK
    CUTLASS_PRAGMA_UNROLL
    for (uint yIter = 0; yIter < WARP_Y_ITER; yIter++) {
      LoadMatrixSyncSwizzle<CUB_W, MmaShape_::kM>(tsm_start + kIter * CUB_W * MmaShape_::kK,
          frag[yIter][kIter & 0x1], warpOffset_ + yIter, kIter);
    }
  }

  CUTLASS_DEVICE void LoadToVreg(ElementB_* tsm_start,
                            FragType_ frag[WARP_Y_ITER],
                            int kIter) {
    CUTLASS_PRAGMA_UNROLL
    for (uint yIter = 0; yIter < WARP_Y_ITER; yIter++) {
      LoadMatrixSyncSwizzle<CUB_W, MmaShape_::kM>(tsm_start + kIter * CUB_W * MmaShape_::kK,
          frag[yIter], warpOffset_ + yIter, kIter & 0x3);
    }
  }

  CUTLASS_DEVICE void LoadToVregWithTsmTrans(ElementB_* tsm_start,
                            TransFragType_ frag[WARP_Y_ITER],
                            int kIter) {
    constexpr uint sliceStride = WarpShape_::kK * MmaShape_::kN;
    constexpr uint step = sizeof(ElementB_) / sizeof(half);

    CUTLASS_PRAGMA_UNROLL
    for (uint yIter = 0; yIter < WARP_Y_ITER; yIter++) {
      LoadMatrixSyncSwizzle<CUB_W, MmaShape_::kK>(tsm_start + (warpOffset_ + yIter) * sliceStride,
          frag[yIter], kIter, ((warpOffset_ + yIter) * step) & 0x3);
    }

  }

  CUTLASS_DEVICE void IncreaseVmemOffset(int cub_z) {
    offset_.z += cub_z;
  }

  CUTLASS_DEVICE void add_tile_offset(cutlass::MatrixCoord const& offset) {
    offset_.y += offset.column() * MmaShape_::kN;
    offset_.z += offset.row() * MmaShape_::kK;
  }

};


/// Partial specialization for RowMajor
template <
    /// Element type for B matrix operand
    typename ElementB,
    /// Element type for internal accumulation
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename MmaShape_,
    typename FragType_>
struct AiuLoaderB <ElementB, cutlass::layout::RowMajor,
                   ThreadblockShape_, WarpShape_, MmaShape_,
                   FragType_> :
  public AiuLoaderBase<ElementB, cutlass::layout::RowMajor> {

  using Base = AiuLoaderBase<ElementB, cutlass::layout::RowMajor>;
  using ElementB_ = typename Base::Element_Internal;
  using AccessType = typename Base::AccessType;

  using Shape = cutlass::MatrixShape<ThreadblockShape_::kK, ThreadblockShape_::kN>;

  constexpr static int CUB_W = ThreadblockShape_::kK;
  constexpr static int CUB_Z_MAX = 128 / sizeof(ElementB_);
  constexpr static int CUB_Z = ThreadblockShape_::kN > CUB_Z_MAX ? CUB_Z_MAX : ThreadblockShape_::kN;
  constexpr static int WARP_PER_BLOCK = (ThreadblockShape_::kM / WarpShape_::kM)
                                      * (ThreadblockShape_::kN / WarpShape_::kN);
  constexpr static int WARP_PER_BLOCK_X = ThreadblockShape_::kM / WarpShape_::kM;
  constexpr static int WARP_Y_ITER = (WarpShape_::kN / MmaShape_::kN);
  constexpr static int CHUNK_K = WarpShape_::kK / MmaShape_::kK;
  constexpr static int CUB_COUNT = (ThreadblockShape_::kN + CUB_Z - 1) / CUB_Z;
  constexpr static int WARP_NUM = (ThreadblockShape_::kM / WarpShape_::kM) * (ThreadblockShape_::kN / WarpShape_::kN)
                                  * (ThreadblockShape_::kK / WarpShape_::kK);

  using FragType = FragType_[WARP_Y_ITER];

  using TransFragType_ = awmma::fragment<awmma::matrix_b, MmaShape_::kM, MmaShape_::kN, MmaShape_::kK,
                                         ElementB_, typename Base::WmmaLayoutTrans>;

  using TransFragType = TransFragType_[WARP_Y_ITER];

  CubeCoord dim_;
  CubeCoord offset_;
  CubeCoord tsmcube_;

  int warp_idx_;
  int warpOffset_;
  const ElementB_* vmem_start_;

  CUTLASS_HOST_DEVICE AiuLoaderB(
      /// Precomputed parameters object
      typename Base::Params const &params,
      /// Pointer to start of tensor
      const ElementB* vmem_start,
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
    vmem_start_ = reinterpret_cast<const ElementB_*>(vmem_start) - aiu_offset;
    warp_idx_ = __ppu_read_firstlane(thread_idx >> 5);
    warpOffset_ = (warp_idx_ / WARP_PER_BLOCK_X) * WARP_Y_ITER;
    offset_.z += aiu_offset;
  }

  CUTLASS_DEVICE void add_pointer_offset(int64_t vmem_offset) {
    vmem_start_ += vmem_offset;
  }

  CUTLASS_DEVICE void LoadToTsm(ElementB_* tsm_start) {
    if (warp_idx_ == 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int cub = 0; cub < CUB_COUNT; ++cub) {
        CubeCoord cub_offset = CubeCoord{offset_.x, offset_.y, offset_.z + cub * CUB_Z};
        int residual_z = ThreadblockShape_::kN - cub * CUB_Z;
        int cub_z = (residual_z > CUB_Z_MAX) ? CUB_Z_MAX : residual_z;
        CubeCoord cub_size {tsmcube_.x, tsmcube_.y, cub_z};
        TsmLoadCube(tsm_start + cub * CUB_W * CUB_Z, vmem_start_, dim_, cub_offset, cub_size);
      }
    }
    IncreaseVmemOffset();
  }

  CUTLASS_DEVICE void LoadToVreg(ElementB_* tsm_start,
                            FragType_ frag[WARP_Y_ITER][2],
                            int kIter) {
    constexpr uint sliceStride = CUB_W * MmaShape_::kN;
    constexpr uint step = sizeof(ElementB_) / sizeof(half);

    CUTLASS_PRAGMA_UNROLL
    for (uint yIter = 0; yIter < WARP_Y_ITER; yIter++) {
      LoadMatrixSyncSwizzle<CUB_W, MmaShape_::kK>(tsm_start + (warpOffset_ + yIter) * sliceStride,
          frag[yIter][kIter & 0x1u], kIter, ((warpOffset_ + yIter) * step) & 0x3);
    }
  }

  CUTLASS_DEVICE void LoadToVreg(ElementB_* tsm_start,
                            FragType_ frag[WARP_Y_ITER],
                            int kIter) {
    constexpr uint sliceStride = CUB_W * MmaShape_::kN;
    constexpr uint step = sizeof(ElementB_) / sizeof(half);

    CUTLASS_PRAGMA_UNROLL
    for (uint yIter = 0; yIter < WARP_Y_ITER; yIter++) {
      LoadMatrixSyncSwizzle<CUB_W, MmaShape_::kK>(tsm_start + (warpOffset_ + yIter) * sliceStride,
          frag[yIter], kIter, ((warpOffset_ + yIter) * step) & 0x3);
    }
  }

  CUTLASS_DEVICE void LoadToVregWithTsmTrans(ElementB_* tsm_start,
                            TransFragType_ frag[WARP_Y_ITER],
                              int kIter) {
    for (uint yIter = 0; yIter < WARP_Y_ITER; yIter++) {
      LoadMatrixSyncSwizzle<CUB_W, MmaShape_::kM>(tsm_start + kIter * WarpShape_::kN * MmaShape_::kK,
          frag[yIter], warpOffset_ + yIter, kIter & 0x3);
    }
  }

  CUTLASS_DEVICE void IncreaseVmemOffset() {
    offset_.y += ThreadblockShape_::kK;
  }

  CUTLASS_DEVICE void add_tile_offset(cutlass::MatrixCoord const& offset) {
    offset_.z += offset.column() * MmaShape_::kN;
    offset_.y += offset.row() * MmaShape_::kK;
  }

};


} // namespace threadblock
} // namespace gemm
} // namespace aiu
