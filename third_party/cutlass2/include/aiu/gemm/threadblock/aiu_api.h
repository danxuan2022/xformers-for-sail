#pragma once
#ifdef __HGGCCC__
#include "hggc_aiu.h"
#endif
#include "aiu/gemm/tool/cutlass_type_convert.h"
#include "cutlass2/array.h"
#include "cutlass2/layout/layout.h"
#include "cutlass2/tensor_ref.h"

constexpr int WARP_SIZE = 32;

namespace aiu {
namespace gemm {
namespace threadblock {

struct CubeCoord {
  uint32_t x; // high dimension
  uint32_t y;
  uint32_t z; // low dimension, contiguous

  CUTLASS_HOST_DEVICE CubeCoord(int a, int b, int c) :
    x((uint32_t)a), y((uint32_t)b), z((uint32_t)c) {}
};

template <typename TsmType, typename VmemType>
CUTLASS_DEVICE void TsmLoadCube(TsmType *tsm, const VmemType *A, CubeCoord &dim,
                                CubeCoord &offset, CubeCoord &cube) {
#ifdef __HGGCCC__
  aiu::cube_load_params aiuLdParams(dim.x, dim.y, dim.z, offset.x, offset.y, offset.z, cube.x,
                                    cube.y, cube.z);
  aiu::tsm_load_cube<aiu::zero_padding, aiu::no_transpose, aiu::swizzle, 1, aiu::strided,
                     aiu::tensor_2D>(reinterpret_cast<int *>(tsm), A, aiuLdParams);
#else
  // cuda simulator for tsm_load_cube
#endif
}

template <int TsmTileSize, int MmaTileSize, typename TsmType, typename FragType>
CUTLASS_DEVICE void LoadMatrixSyncSwizzle(TsmType *tsmStart, FragType &frag, int cube_offset_x,
                                          int slice_id) {
  awmma::matrix_load_params tsmLdParams(0, cube_offset_x * MmaTileSize, 1, TsmTileSize, slice_id);
  awmma::load_matrix_sync<awmma::tile_16x1, awmma::swizzle_default>(frag, tsmStart, tsmLdParams);
}

template <
    /// Element type for A matrix operand
    typename Element_,
    /// Layout type for A matrix operand
    typename Layout_>
struct AiuLoaderBase {

  using TensorRef = cutlass::TensorRef<Element_, Layout_>;
  using Layout = Layout_;
  using Element = Element_;

  using Element_Internal = typename FromCutlassType<Element_>::type;
  using AccessType = cutlass::AlignedArray<Element_, 16>;
  using WmmaLayout =
      typename cutlass::platform::conditional<cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value,
                                awmma::row_major, awmma::col_major>::type;
  using WmmaLayoutTrans =
      typename cutlass::platform::conditional<cutlass::platform::is_same<Layout, cutlass::layout::RowMajor>::value,
                                awmma::col_major, awmma::row_major>::type;

  struct Params {

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout) : layout_(layout) {}

    CUTLASS_HOST_DEVICE
    Params() { }
    Layout layout_;
  };

  CUTLASS_HOST_DEVICE AiuLoaderBase () {

  }

  // CubeCoord dim_;
  // CubeCoord offset_;
  // CubeCoord tsmcube_;

  // int warp;
  // int warpOffset;
  // Element* vmem_ptr_;
};


} // namespace threadblock
} // namespace gemm
} // namespace aiu