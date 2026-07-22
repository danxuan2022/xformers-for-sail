#pragma once

#include "cutlass2/array.h"

static CUTLASS_DEVICE int8_t get_lane_id() {
  return threadIdx.x;
}
static CUTLASS_DEVICE int8_t get_warp_id() {
  return threadIdx.y;
}

template <typename T, typename S, int N> struct NumericArrayConverterPPU {
  using ElementCompute = T;

  bool is_source_needed() { return false; }
};

template <> struct NumericArrayConverterPPU<cutlass::half_t, float, 4> {
  using result_type = cutlass::Array<cutlass::half_t, 4>;
  using source_type = cutlass::Array<float, 4>;
  using ElementCompute = cutlass::half_t;

  CUTLASS_DEVICE
  result_type convert(source_type const &src_const) {
    result_type result;
    source_type src;

    // copy src to src_copy
    src = src_const;

#if SAIL_PPU_MMA
    // reorder data layout in Acc fragment
    // first exchange T1(R0)<->T0(R1), T3(R0)<->T2(R1)
    // second exchange T1(R0)<->T2(R0), T1(R1)<->T2(R1)
    int lane_id = get_lane_id();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i += 2) {
      float data = lane_id & 0x1 ? src[i] : src[i + 1];
      float res = __shfl_xor_sync(0xFFFFFFFF, data, 1, 4);
      if (lane_id & 0x1) {
        src[i] = res;
      } else {
        src[i + 1] = res;
      }
      double* tmp = reinterpret_cast<double*>(src.data() + i);
      double d = __shfl_xor_sync(0x66666666, tmp[0], 3, 4);
      if (lane_id % 4 == 1 || lane_id % 4 == 2) {
        tmp[0] = d;
      }
    }
#endif

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < src.size(); i++) {
      result[i] = (cutlass::half_t)src[i];
    }

    return result;
  }

  CUTLASS_DEVICE
  bool is_source_needed() { return false; }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) { return convert(s); }
};

template <int N> struct NumericArrayConverterPPU<cutlass::half_t, float, N> {
  static constexpr int VEC_WIDTH = 4;
  static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

  using result_type = cutlass::Array<cutlass::half_t, N>;
  using source_type = cutlass::Array<float, N>;
  using ElementCompute = cutlass::half_t;

  CUTLASS_DEVICE
  static result_type convert(source_type const &source) {
    using scalar_result_type = typename result_type::Element;
    using scalar_source_type = typename source_type::Element;
    NumericArrayConverterPPU<scalar_result_type, scalar_source_type, VEC_WIDTH> convert_vector_;

    result_type result;
    using vec_result = cutlass::Array<scalar_result_type, VEC_WIDTH>;
    using vec_source = cutlass::Array<scalar_source_type, VEC_WIDTH>;

    vec_result *result_ptr = reinterpret_cast<vec_result *>(&result);
    vec_source const *source_ptr = reinterpret_cast<vec_source const *>(&source);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / VEC_WIDTH; ++i) {
      result_ptr[i] = convert_vector_.convert(source_ptr[i]);
    }

    return result;
  }

  CUTLASS_DEVICE
  bool is_source_needed() { return false; }

  CUTLASS_DEVICE
  result_type operator()(source_type const &s) { return convert(s); }
};
