#pragma once

#include "cutlass2/cutlass2.h"
#include "cutlass2/numeric_types.h"
#include "cutlass2/array.h"
#include "cutlass2/utils.h"
#include "cutlass2/layout/matrix.h"
#include "cutlass2/layout/tensor.h"
#include "cutlass2/matrix_shape.h"
#include "cutlass2/tensor_ref.h"
#include "cutlass2/transform/pitch_linear_thread_map.h"
#include "cutlass2/epilogue/threadblock/output_tile_thread_map.h"
#include "cutlass2/arch/arch.h"
#include "cutlass2/arch/memory.h"
#include "cutlass2/epilogue/threadblock/predicated_tile_iterator_params.h"

////////////////////////////////////////////////////////////////////////////////

//#define ENABLE_DEBUG

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////

namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to prefetch C input from global memory to LLC.
///
///
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  typename ThreadblockShape_,///< Threadblock-scoped matrix multiply-accumulate 
  PrefetchStrategyType::Kind Strategy_
>
class PrefetchTileIterator {
  using ThreadMap = ThreadMap_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  using ThreadblockShape = ThreadblockShape_;

  public:

  static PrefetchStrategyType::Kind const kPrefetchStrategyType = PrefetchStrategyType::Kind::kNoPrefetchNeeded;
  
    CUTLASS_DEVICE
    PrefetchTileIterator(
      Element *pointer,
      TensorCoord extent,
      int thread_idx,
      TensorCoord threadblock_offset = TensorCoord()
    ){}

    CUTLASS_DEVICE void prefetch_all_to_llc() {}
    CUTLASS_DEVICE void prefetch_to_llc(int i) {}
    CUTLASS_DEVICE int get_iter_num() {return 0;}
};

/// Tile iterator used to prefetch C input from global memory to LLC.
///
/// Specialized for kStridedPrefetchLLCStream
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  typename ThreadblockShape_ ///< Threadblock-scoped matrix multiply-accumulate 
>
class PrefetchTileIterator<ThreadMap_, Element_, ThreadblockShape_, PrefetchStrategyType::Kind::kStridedPrefetchLLCStream> {
public:
  using ThreadMap = ThreadMap_;
  using Element = Element_;
  using ThreadblockShape = ThreadblockShape_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static int const kPrefetchDataInBytes = 128;
  static int const kElementsPerPrefetch = kPrefetchDataInBytes / sizeof(Element);

  static int const kThreadNumPerTbTileN = ThreadblockShape::kN / kElementsPerPrefetch;

  static PrefetchStrategyType::Kind const kPrefetchStrategyType = PrefetchStrategyType::Kind::kStridedPrefetchLLCStream;

  static_assert((ThreadblockShape::kN % kElementsPerPrefetch) == 0, "kElementsPerPrefetch must be dividable by ThreadblockShape::kN");

private:

  //
  // Data members
  //

  /// Byte-level pointer for the base address to be prefetched
  uint8_t *c_base_pointer_ = nullptr;

  /// Extent of the matrix tile in rows
  Index extent_row_;
  Index extent_column_;
  Index tb_offset_;
  Index thread_idx_;
  Index totalStrideByTB_;
  Index problem_size_;
  Index strideByTBPerIter_;
  
  //Index ele_num_per_grid_ = ThreadblockShape::kM * gridDim.x * gridDim.y * ThreadblockShape::kN;

  LongIndex bulk_offset_;

  #ifdef ENABLE_DEBUG
  TensorCoord threadblock_offset_;
  #endif

public:
  CUTLASS_DEVICE
  PrefetchTileIterator(
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord()
  )
  {
    extent_row_ = extent.row();
    extent_column_ = extent.column();

    c_base_pointer_ = reinterpret_cast<uint8_t *>(pointer);

    tb_offset_ = LongIndex(threadblock_offset.row()) * LongIndex(extent_column_) + LongIndex(threadblock_offset.column());

    thread_idx_ = thread_idx;

    totalStrideByTB_ = ThreadblockShape::kM * extent_column_;
    problem_size_ = extent_row_ * extent_column_;
    strideByTBPerIter_ = (blockDim.x * blockDim.y * blockDim.z) / kThreadNumPerTbTileN * extent_column_;
    bulk_offset_ =  LongIndex(tb_offset_) + LongIndex(thread_idx_ / kThreadNumPerTbTileN) * LongIndex(extent_column_)
                                                  +  LongIndex(thread_idx_ % kThreadNumPerTbTileN) * kElementsPerPrefetch;

    #ifdef ENABLE_DEBUG
    threadblock_offset_ = threadblock_offset;
    #endif
  }

  CUTLASS_DEVICE int get_iter_num() {return (totalStrideByTB_ + strideByTBPerIter_ - 1) / strideByTBPerIter_;}

  CUTLASS_DEVICE void prefetch_to_llc(int iter) {
    #if defined(__HGGC_ARCH__)
    Index accumulated_ele_num_per_tb = iter * strideByTBPerIter_;
    if ((accumulated_ele_num_per_tb + bulk_offset_) <= (problem_size_ - kElementsPerPrefetch)) {
      __ppu_prefetch_nonebulk_LLC(c_base_pointer_ + (accumulated_ele_num_per_tb + bulk_offset_) * sizeof(Element));

      #ifdef ENABLE_DEBUG
      //if (accumulated_ele_num_per_tb <= totalStrideByTB_ && (accumulated_ele_num_per_tb + bulk_offset_ + kElementsPerPrefetch) >= problem_size_)
      //if (threadIdx.x == 255 && blockIdx.x == 195 && blockIdx.y == 1)
      printf("threadIdx.x=%lu, threadIdx.y=%lu, threadIdx.z=%lu, \
                    blockIdx.x=%lu, blockIdx.y=%lu, blockIdx.z=%lu, \
                    totalStrideByTB_=%lu, strideByTBPerIter_=%lu, \
                    threadblock_offset_.row()=%lu, extent_column_=%lu, threadblock_offset_.column()=%lu, \
                    tb_offset_=%lu, bulk_offset_=%lu, accumulated_ele_num_per_tb=%lu, thread_idx_=%lu\n", \
                    threadIdx.x, threadIdx.y, threadIdx.z, \
                    blockIdx.x, blockIdx.y, blockIdx.z, \
                    totalStrideByTB_, strideByTBPerIter_, \
                    threadblock_offset_.row(), extent_column_, threadblock_offset_.column(), \
                    tb_offset_, bulk_offset_, accumulated_ele_num_per_tb, thread_idx_ \
                  );

      {
        // Do sanity check for each prefetched offset;
        LongIndex offset = accumulated_ele_num_per_tb + bulk_offset_;
        LongIndex right_offset = blockIdx.x * ThreadblockShape::kM * extent_column_;
        right_offset += blockIdx.y * ThreadblockShape::kN;
        right_offset += accumulated_ele_num_per_tb;
        right_offset += LongIndex(thread_idx_ / kThreadNumPerTbTileN) * LongIndex(extent_column_)
                        +  LongIndex(thread_idx_ % kThreadNumPerTbTileN) * kElementsPerPrefetch;
        if(offset != right_offset) {
          printf("threadIdx.x=%lu, threadIdx.y=%lu, threadIdx.z=%lu, \
                    blockIdx.x=%lu, blockIdx.y=%lu, blockIdx.z=%lu, \
                    offset=%lu, right_offset=%lu, \
                    totalStrideByTB_=%lu, strideByTBPerIter_=%lu, \
                    threadblock_offset_.row()=%lu, extent_column_=%lu, threadblock_offset_.column()=%lu, \
                    tb_offset_=%lu, bulk_offset_=%lu, accumulated_ele_num_per_tb=%lu, thread_idx_=%lu\n", \
                    threadIdx.x, threadIdx.y, threadIdx.z, \
                    blockIdx.x, blockIdx.y, blockIdx.z, \
                    offset, right_offset, \
                    totalStrideByTB_, strideByTBPerIter_, \
                    threadblock_offset_.row(), extent_column_, threadblock_offset_.column(), \
                    tb_offset_, bulk_offset_, accumulated_ele_num_per_tb, thread_idx_ \
                  );
        }
      }
      #endif
    }
    #endif
  }

  CUTLASS_DEVICE void prefetch_all_to_llc() {
    #if defined(__HGGC_ARCH__)
    for (int i = 0; i < get_iter_num(); i++) {
      prefetch_to_llc(i);
    }
    #endif
  }
};

/// Tile iterator used to prefetch C input from global memory to LLC.
///
/// Specialized for kStridedPrefetchLLCSingle
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  typename ThreadblockShape_ ///< Threadblock-scoped matrix multiply-accumulate 
>
class PrefetchTileIterator<ThreadMap_, Element_, ThreadblockShape_, PrefetchStrategyType::Kind::kStridedPrefetchLLCSingle> {
public:
  using ThreadMap = ThreadMap_;
  using Element = Element_;
  using ThreadblockShape = ThreadblockShape_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static int const kPrefetchDataInBytes = 128;
  static int const kElementsPerPrefetch = kPrefetchDataInBytes / sizeof(Element);

  static int const kThreadNumPerTbTileN = ThreadblockShape::kN / kElementsPerPrefetch;

  static PrefetchStrategyType::Kind const kPrefetchStrategyType = PrefetchStrategyType::Kind::kStridedPrefetchLLCSingle;

  static_assert((ThreadblockShape::kN % kElementsPerPrefetch) == 0, "kElementsPerPrefetch must be dividable by ThreadblockShape::kN");

private:

  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.

  /// Byte-level pointer for the base address to be prefetched
  uint8_t *c_base_pointer_ = nullptr;

  /// Extent of the matrix tile in rows
  Index extent_row_;
  Index extent_column_;
  Index tb_offset_;
  Index thread_idx_;
  Index totalStrideByTB_;
  Index problem_size_;
  Index strideByTBPerIter_;
  
  //Index ele_num_per_grid_ = ThreadblockShape::kM * gridDim.x * gridDim.y * ThreadblockShape::kN;

  LongIndex bulk_offset_;

  #ifdef ENABLE_DEBUG
  TensorCoord threadblock_offset_;
  #endif

public:
  CUTLASS_DEVICE
  PrefetchTileIterator(
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord()
  )
  {
    extent_row_ = extent.row();
    extent_column_ = extent.column();

    c_base_pointer_ = reinterpret_cast<uint8_t *>(pointer);

    tb_offset_ = LongIndex(threadblock_offset.row()) * LongIndex(extent_column_) + LongIndex(threadblock_offset.column());

    thread_idx_ = thread_idx;

    totalStrideByTB_ = ThreadblockShape::kM * extent_column_;
    problem_size_ = extent_row_ * extent_column_;
    strideByTBPerIter_ = (blockDim.x * blockDim.y * blockDim.z) / kThreadNumPerTbTileN * extent_column_;
    bulk_offset_ =  LongIndex(tb_offset_) + LongIndex(thread_idx_ / kThreadNumPerTbTileN) * LongIndex(extent_column_)
                                                  +  LongIndex(thread_idx_ % kThreadNumPerTbTileN) * kElementsPerPrefetch;
    #ifdef ENABLE_DEBUG
    threadblock_offset_ = threadblock_offset;
    #endif
  }

  CUTLASS_DEVICE int get_iter_num() {return (totalStrideByTB_ + strideByTBPerIter_ - 1) / strideByTBPerIter_;}

  CUTLASS_DEVICE void prefetch_to_llc(int iter) {}

  CUTLASS_DEVICE void prefetch_all_to_llc() {
    #if defined(__HGGC_ARCH__)
    Index accumulated_ele_num_per_tb = 0;
    while (accumulated_ele_num_per_tb < totalStrideByTB_ && (accumulated_ele_num_per_tb + bulk_offset_) <= (problem_size_ - kElementsPerPrefetch)) {
      __ppu_prefetch_nonebulk_LLC(c_base_pointer_ + (accumulated_ele_num_per_tb + bulk_offset_) * sizeof(Element));

      #ifdef ENABLE_DEBUG
      //if (accumulated_ele_num_per_tb <= totalStrideByTB_ && (accumulated_ele_num_per_tb + bulk_offset_ + kElementsPerPrefetch) >= problem_size_)
      //if (threadIdx.x == 255 && blockIdx.x == 195 && blockIdx.y == 1)
      printf("threadIdx.x=%lu, threadIdx.y=%lu, threadIdx.z=%lu, \
                    blockIdx.x=%lu, blockIdx.y=%lu, blockIdx.z=%lu, \
                    totalStrideByTB_=%lu, strideByTBPerIter_=%lu, \
                    threadblock_offset_.row()=%lu, extent_column_=%lu, threadblock_offset_.column()=%lu, \
                    tb_offset_=%lu, bulk_offset_=%lu, accumulated_ele_num_per_tb=%lu, thread_idx_=%lu\n", \
                    threadIdx.x, threadIdx.y, threadIdx.z, \
                    blockIdx.x, blockIdx.y, blockIdx.z, \
                    totalStrideByTB_, strideByTBPerIter_, \
                    threadblock_offset_.row(), extent_column_, threadblock_offset_.column(), \
                    tb_offset_, bulk_offset_, accumulated_ele_num_per_tb, thread_idx_ \
                  );

      {
        // Do sanity check for each prefetched offset;
        LongIndex offset = accumulated_ele_num_per_tb + bulk_offset_;
        LongIndex right_offset = blockIdx.x * ThreadblockShape::kM * extent_column_;
        right_offset += blockIdx.y * ThreadblockShape::kN;
        right_offset += accumulated_ele_num_per_tb;
        right_offset += LongIndex(thread_idx_ / kThreadNumPerTbTileN) * LongIndex(extent_column_)
                        +  LongIndex(thread_idx_ % kThreadNumPerTbTileN) * kElementsPerPrefetch;
        if(offset != right_offset) {
          printf("threadIdx.x=%lu, threadIdx.y=%lu, threadIdx.z=%lu, \
                    blockIdx.x=%lu, blockIdx.y=%lu, blockIdx.z=%lu, \
                    offset=%lu, right_offset=%lu, \
                    totalStrideByTB_=%lu, strideByTBPerIter_=%lu, \
                    threadblock_offset_.row()=%lu, extent_column_=%lu, threadblock_offset_.column()=%lu, \
                    tb_offset_=%lu, bulk_offset_=%lu, accumulated_ele_num_per_tb=%lu, thread_idx_=%lu\n", \
                    threadIdx.x, threadIdx.y, threadIdx.z, \
                    blockIdx.x, blockIdx.y, blockIdx.z, \
                    offset, right_offset, \
                    totalStrideByTB_, strideByTBPerIter_, \
                    threadblock_offset_.row(), extent_column_, threadblock_offset_.column(), \
                    tb_offset_, bulk_offset_, accumulated_ele_num_per_tb, thread_idx_ \
                  );
        }
      }
      #endif

      accumulated_ele_num_per_tb += strideByTBPerIter_;
    }
    #endif
  }
};

/// Tile iterator used to prefetch C input from global memory to LLC.
///
/// Specialized for kLinearPrefetchLLCSingle
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  typename ThreadblockShape_ ///< Threadblock-scoped matrix multiply-accumulate 
>
class PrefetchTileIterator<ThreadMap_, Element_, ThreadblockShape_, PrefetchStrategyType::Kind::kLinearPrefetchLLCSingle> {
public:
  using ThreadMap = ThreadMap_;
  using Element = Element_;
  using ThreadblockShape = ThreadblockShape_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static int const kPrefetchDataInBytes = 128;
  static int const kElementsPerPrefetch = kPrefetchDataInBytes / sizeof(Element);

  static int const kThreadNumPerTbTileN = ThreadblockShape::kN / kElementsPerPrefetch;

  static PrefetchStrategyType::Kind const kPrefetchStrategyType = PrefetchStrategyType::Kind::kLinearPrefetchLLCSingle;

  static_assert((ThreadblockShape::kN % kElementsPerPrefetch) == 0, "kElementsPerPrefetch must be dividable by ThreadblockShape::kN");

  

private:

  //
  // Data members
  //


  /// Byte-level pointer
  uint8_t *c_base_pointer_ = nullptr;

  /// Extent of the matrix tile in rows
  Index extent_row_;
  Index extent_column_;
  Index problem_size_;
  Index ele_num_per_block_;
  Index ele_num_per_grid_;
  Index block_idx_prefetch_;

  LongIndex bulk_offset_;

  int thread_idx_;

public:
  CUTLASS_DEVICE
  PrefetchTileIterator(
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord()
  )
  {
    extent_row_ = extent.row();
    extent_column_ = extent.column();
    problem_size_ = extent_row_ * extent_column_;
    thread_idx_ = thread_idx;
    c_base_pointer_ = reinterpret_cast<uint8_t *>(pointer);
    ele_num_per_block_ = kElementsPerPrefetch * blockDim.x * blockDim.y * blockDim.z;
    ele_num_per_grid_ = ele_num_per_block_ * gridDim.x * gridDim.y * gridDim.z;
    block_idx_prefetch_ = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    bulk_offset_ = block_idx_prefetch_ * ele_num_per_block_ + thread_idx_ * kElementsPerPrefetch;
  }

  CUTLASS_DEVICE int get_iter_num() {return (problem_size_ - kElementsPerPrefetch + ele_num_per_grid_ - 1) / ele_num_per_grid_;}

  CUTLASS_DEVICE void prefetch_to_llc(int iter) {}

  CUTLASS_DEVICE void prefetch_all_to_llc() {
    #if defined(__HGGC_ARCH__)
    while (bulk_offset_ <= (problem_size_ - kElementsPerPrefetch)) {

      #ifdef ENABLE_DEBUG
      //if ( (bulk_offset_ + kElementsPerPrefetch) >= problem_size_ ) {
        printf("threadIdx.x=%lu, threadIdx.y=%lu, threadIdx.z=%lu, \
                blockIdx.x=%lu, blockIdx.y=%lu, blockIdx.z=%lu, \
                ele_num_per_block_=%lu, ele_num_per_grid_=%lu, \
                bulk_offset_=%lu, block_idx_prefetch_=%lu, thread_idx_=%lu\n", \
                threadIdx.x, threadIdx.y, threadIdx.z, \
                blockIdx.x, blockIdx.y, blockIdx.z, \
                ele_num_per_block_, ele_num_per_grid_, \
                bulk_offset_, block_idx_prefetch_, thread_idx_
              );
      //}
      #endif

      __ppu_prefetch_nonebulk_LLC(c_base_pointer_ + (bulk_offset_ * sizeof(Element)));
      bulk_offset_ += ele_num_per_grid_;
    }
    #endif
  }
};

} //threadblock
} //epilogue
} //cutlass