#pragma once
#include "cutlass2/numeric_types.h"
namespace fmha {

#if ENABLE_AIU

// copy from global to shared memory
template< typename Iterator,
          typename SmemIterator>
inline __device__ void copy_g2s(Iterator &iterator, SmemIterator *smem_ptr) {
  iterator.LoadToTsm(smem_ptr);
}

#endif

// copy from global to shared memory
template< typename Iterator,
          typename SmemIterator>
inline __device__ void copy_g2s(Iterator &iterator, SmemIterator &smem_iterator) {
  iterator.set_iteration_index(0);
  smem_iterator.set_iteration_index(0);

  constexpr int AsyncCopyIterationsPerStage =
      Iterator::ThreadMap::Iterations::kCount;

  // Async Copy for operand A
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < AsyncCopyIterationsPerStage; ++j) {
    typename Iterator::AccessType *dst_ptr =
        reinterpret_cast<typename Iterator::AccessType *>(
            smem_iterator.get());

    CUTLASS_PRAGMA_UNROLL
    for (int v = 0; v < Iterator::kAccessesPerVector; ++v) {
      int const kSrcBytes =
          cutlass::sizeof_bits<typename Iterator::Element>::value *
          Iterator::ThreadMap::kElementsPerAccess /
          Iterator::kAccessesPerVector / 8;

      int src_bytes = (iterator.valid() ? kSrcBytes : 0);

      cutlass::arch::cp_async_zfill<kSrcBytes/*, cutlass::arch::CacheOperation::Global*/>(
          dst_ptr + v, iterator.get(), iterator.valid());

      ++iterator;
    }
    ++smem_iterator;
  }
}


#if ENABLE_AIU

#define CONVERT_MMA_FRAGMENT_VALUE(cutlassfragType, wmmafrag)  (*(reinterpret_cast<wmmafrag *>(&(cutlassfragType))))


// gemm on shared A and shared B
template< typename Mma, typename IteratorA, typename IteratorB, typename ElementAB>
inline __device__ void gemm_on_sAsB_aiu(typename Mma::FragmentC & accum,
                                    IteratorA &warp_tile_iterator_A,
                                    IteratorB &warp_tile_iterator_B,
                                    ElementAB smem_ptr_A,
                                    ElementAB smem_ptr_B) {
  typename Mma::Operator warp_mma;

  typename Mma::Operator::FragmentA warp_frag_A;
  typename Mma::Operator::FragmentB warp_frag_B;

  constexpr int kWarpGemmIterations = IteratorA::CHUNK_K;

  // warp_tile_iterator_A.set_kgroup_index(0);
  // warp_tile_iterator_B.set_kgroup_index(0);

  CUTLASS_PRAGMA_UNROLL
  for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {

    // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
    // as the case may be.

    // warp_tile_iterator_A.set_kgroup_index(warp_mma_k % kWarpGemmIterations);
    // warp_tile_iterator_B.set_kgroup_index(warp_mma_k % kWarpGemmIterations);

    typename IteratorA::FragType& fragA = CONVERT_MMA_FRAGMENT_VALUE(warp_frag_A, typename IteratorA::FragType);
    typename IteratorB::FragType& fragB = CONVERT_MMA_FRAGMENT_VALUE(warp_frag_B, typename IteratorB::FragType);

    warp_tile_iterator_A.LoadToVreg(smem_ptr_A, fragA, warp_mma_k);
    warp_tile_iterator_B.LoadToVreg(smem_ptr_B, fragB, warp_mma_k);

    // ++warp_tile_iterator_A;
    // ++warp_tile_iterator_B;

    warp_mma(accum, warp_frag_A, warp_frag_B, accum);
  }
  // reset iterator to start point
  warp_tile_iterator_A.add_tile_offset({0, - kWarpGemmIterations});
  warp_tile_iterator_B.add_tile_offset({- kWarpGemmIterations, 0});
}

// gemm on shared A and shared B
template< typename Mma, typename MmaOperatorIteratorA, typename IteratorB, typename ElementAB>
inline __device__ void gemm_on_sAsB_aiuB(typename Mma::FragmentC & accum,
                                    MmaOperatorIteratorA &warp_tile_iterator_A,
                                    IteratorB &warp_tile_iterator_B,
                                    ElementAB smem_ptr_B) {
  typename Mma::Operator warp_mma;

  typename Mma::Operator::FragmentA warp_frag_A;
  typename Mma::Operator::FragmentB warp_frag_B;

  constexpr int kWarpGemmIterations = IteratorB::CHUNK_K;

  CUTLASS_PRAGMA_UNROLL
  for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {

    // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
    // as the case may be.
    typename IteratorB::FragType& fragB = CONVERT_MMA_FRAGMENT_VALUE(warp_frag_B, typename IteratorB::FragType);

    warp_tile_iterator_A.load(warp_frag_A);
    warp_tile_iterator_B.LoadToVreg(smem_ptr_B, fragB, warp_mma_k);

    ++warp_tile_iterator_A;

    warp_mma(accum, warp_frag_A, warp_frag_B, accum);
  }
  // reset iterator to start point
  warp_tile_iterator_A.add_tile_offset({0, - kWarpGemmIterations});
  warp_tile_iterator_B.add_tile_offset({- kWarpGemmIterations, 0});
}
#endif

// gemm on shared A and shared B
template< typename Mma, typename MmaOperatorIteratorA>
inline __device__ void gemm_on_sAsB(typename Mma::FragmentC & accum,
                                    MmaOperatorIteratorA & warp_tile_iterator_A,
                                    typename Mma::Operator::IteratorB & warp_tile_iterator_B) {
  typename Mma::Operator warp_mma;

  typename Mma::Operator::FragmentA warp_frag_A;
  typename Mma::Operator::FragmentB warp_frag_B;

  constexpr int kWarpGemmIterations =
      (Mma::Policy::Operator::Shape::kK / Mma::Operator::Policy::MmaShape::kK);

  // warp_tile_iterator_A.set_kgroup_index(0);
  // warp_tile_iterator_B.set_kgroup_index(0);

  CUTLASS_PRAGMA_UNROLL
  for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {

    // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
    // as the case may be.

    // warp_tile_iterator_A.set_kgroup_index(warp_mma_k % kWarpGemmIterations);
    // warp_tile_iterator_B.set_kgroup_index(warp_mma_k % kWarpGemmIterations);

    warp_tile_iterator_A.load(warp_frag_A);
    warp_tile_iterator_B.load(warp_frag_B);

    ++warp_tile_iterator_A;
    ++warp_tile_iterator_B;

    warp_mma(accum, warp_frag_A, warp_frag_B, accum);
  }
  // reset iterator to start point
  warp_tile_iterator_A.add_tile_offset({0, - kWarpGemmIterations});
  warp_tile_iterator_B.add_tile_offset({- kWarpGemmIterations, 0});
}

// gemm on register A and shared B
template< typename Mma,
          typename FragmentIteratorA>
inline __device__ void gemm_on_rAsB(typename Mma::FragmentC & accum,
                                    FragmentIteratorA & warp_tile_iterator_A,
                                    typename Mma::Operator::IteratorB & warp_tile_iterator_B) {
  typename Mma::Operator warp_mma;

  typename Mma::Operator::FragmentA warp_frag_A;
  typename Mma::Operator::FragmentB warp_frag_B;

  constexpr int kWarpGemmIterations =
      (Mma::Policy::Operator::Shape::kK / Mma::Operator::Policy::MmaShape::kK);

  warp_tile_iterator_B.set_kgroup_index(0);

  CUTLASS_PRAGMA_UNROLL
  for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {

    // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
    // as the case may be.


    warp_tile_iterator_A.load(warp_frag_A);
    warp_tile_iterator_B.load(warp_frag_B);

    ++warp_tile_iterator_A;
    ++warp_tile_iterator_B;

    warp_mma(accum, warp_frag_A, warp_frag_B, accum);
  }
  warp_tile_iterator_B.add_tile_offset({- kWarpGemmIterations, 0});
}

namespace flash {

struct ull2 {
    unsigned long long x;
    unsigned long long y;
};

inline __device__ uint2 mulhilo32(const unsigned int a, const unsigned int b) {
    uint2 *res;
    unsigned long long tmp;
    asm ("mul.wide.u32 %0, %1, %2;\n\t"
          : "=l"(tmp)
          : "r"(a), "r"(b));
    res = (uint2*)(&tmp);
    return *res;
}

inline __device__ uint4 philox_single_round(const uint4 ctr, const uint2 key) {
    constexpr unsigned long kPhiloxSA = 0xD2511F53;
    constexpr unsigned long kPhiloxSB = 0xCD9E8D57;
    uint2 res0 = mulhilo32(kPhiloxSA, ctr.x);
    uint2 res1 = mulhilo32(kPhiloxSB, ctr.z);
    uint4 ret = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
    return ret;
}

inline __device__ uint4 philox(unsigned long long seed,
                               unsigned long long subsequence,
                               unsigned long long offset) {
    constexpr unsigned long kPhilox10A = 0x9E3779B9;
    constexpr unsigned long kPhilox10B = 0xBB67AE85;
    uint2 key = reinterpret_cast<uint2&>(seed);
    uint4 counter;
    ull2 *tmp = reinterpret_cast<ull2*>(&counter);
    tmp->x = offset;
    tmp->y = subsequence;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        counter = philox_single_round(counter, key);
        key.x += (kPhilox10A);
        key.y += (kPhilox10B);
    }
    uint4 output = philox_single_round(counter, key);
    return output;
}

} // namespace flash

enum DropoutMode {
  DROPOUT_MODE_KEEP_ABS = 0,
  DROPOUT_MODE_RESET_TO_0 = 1,
  DROPOUT_MODE_SET_BOOLEAN = 2
};

template< int WarpShapeM,
          int WarpShapeN,
          DropoutMode mode = DROPOUT_MODE_KEEP_ABS,
          typename Fragment>
inline __device__ void apply_dropout (Fragment& frag, uint8_t p_dropout_in_uint8_t,
                                      unsigned long long seed, unsigned long long offset,
                                      int block_row_start, int block_col_start,
                                      int block_row_stride, float dropout_scale = 0) {
  using T = typename Fragment::Element;
  auto encode_dropout = [&](bool keep, T val) {
    if (mode == DROPOUT_MODE_KEEP_ABS) {
      return keep ? (T)val : (T)-val;
    } else if (mode == DROPOUT_MODE_RESET_TO_0) {
      return keep ? (T)(dropout_scale * val) :  T(0);
    } else if (mode == DROPOUT_MODE_SET_BOOLEAN) {
      return (T)keep ? T(dropout_scale) : T(0);
    }
  };

  constexpr int M_SIZE = WarpShapeM / 32;
  constexpr int N_SIZE = WarpShapeN / 16;
  static_assert(M_SIZE > 0, "WarpShapeM should be multiple of 32");
  static_assert(N_SIZE > 0, "WarpShapeN should be multiple of 16");

  constexpr int M_STRIDE = 16;
  constexpr int N_STRIDE = M_SIZE * 16;

  #pragma unroll
  for (int m = 0; m < M_SIZE; ++m, block_row_start += block_row_stride) {
    uint2 rowcol = make_uint2(block_row_start, block_col_start);

    #pragma unroll
    for (int n = 0; n < N_SIZE; ++n, ++rowcol.y) {
      // if (threadIdx.x == 0 && threadIdx.y == 0 && (blockIdx.x == 0 || blockIdx.x == 1) && blockIdx.y == 0 && blockIdx.z == 0) {
      //   printf("warpid = %d, m = %d, n = %d, row = %d, col = %d, offset = %d\n",
      //     threadIdx.x / 32, m, n, int(rowcol.x), int(rowcol.y), offset);
      // }
      uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long&>(rowcol), offset);

      uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);

      const int base = m * M_STRIDE + n * N_STRIDE;

      #pragma unroll
      for (int i = 0; i < 16; i++) {
        frag[base + i] = encode_dropout(rnd_8[i] <= p_dropout_in_uint8_t, frag[base + i]);
        // if (threadIdx.x < 4 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //   printf("tid = %d, vreg idx = %d, rnd_8 = %d, p_dropout_in_uint8_t = %d, value = %f\n",
        //       threadIdx.x, base + i, rnd_8[i], p_dropout_in_uint8_t, (float)frag[base + i]);
        // }
      }
    }
  }
}

} // namespace fmha