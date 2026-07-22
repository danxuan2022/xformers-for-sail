/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "cutlass2/arch/mma.h"

////////////////////////////////////////////////////////////////////////////////
// Some helper functions
////////////////////////////////////////////////////////////////////////////////
#undef DISPATCH_TYPES
#define DISPATCH_TYPES(tensor, func)                                           \
  {                                                                            \
    if (query.scalar_type() == at::ScalarType::Float) {                        \
      using scalar_t = float;                                                  \
      func();                                                                  \
    } else if (query.scalar_type() == at::ScalarType::Half) {                  \
      using scalar_t = cutlass::half_t;                                        \
      func();                                                                  \
    } else if (query.scalar_type() == at::ScalarType::BFloat16) {              \
      using scalar_t = cutlass::bfloat16_t;                                    \
      func();                                                                  \
    } else {                                                                   \
      XFORMERS_CHECK(false, "Only fp32, half & bf16 supported at the moment"); \
    }                                                                          \
  }

#undef DISPATCH_BOOL
#define DISPATCH_BOOL(BOOL_V, BOOL_NAME, F) \
  {                                         \
    if (BOOL_V) {                           \
      constexpr bool BOOL_NAME = true;      \
      F();                                  \
    } else {                                \
      constexpr bool BOOL_NAME = false;     \
      F();                                  \
    }                                       \
  }

#undef DISPATCH_ARCHTAG
#define DISPATCH_ARCHTAG(CC, func)                                        \
  {                                                                       \
    if (CC >= 80) {                                                       \
      using ArchTag = cutlass::arch::Sm80;                                \
      func();                                                             \
    } else {                                                              \
      XFORMERS_CHECK(                                                     \
          false,                                                          \
          "Your device is too old. PPU require compute capability >= 80");\
    }                                                                     \
  }

#undef CHECK_NOSPARSE_CONTIGUOUS_CUDA
#define CHECK_NOSPARSE_CONTIGUOUS_CUDA(TENSOR)                            \
  XFORMERS_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  XFORMERS_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  XFORMERS_CHECK(TENSOR.is_contiguous());

#undef CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA
#define CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(TENSOR)                        \
  XFORMERS_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  XFORMERS_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  XFORMERS_CHECK(                                                         \
      TENSOR.stride(-1) == 1, #TENSOR ": last dimension must be contiguous");

#undef CHECK_ALIGNED_PTR
#ifdef TORCH_CHECK
#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT) \
  XFORMERS_CHECK(                         \
      uint64_t(PTR) % ALIGNMENT == 0, #PTR " is not correctly aligned")
#define XFORMERS_CHECK TORCH_CHECK
#elif defined(__CUDACC_RTC__)
#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT)  \
  if (!(uint64_t(PTR) % ALIGNMENT == 0)) { \
    return false;                          \
  }
#define XFORMERS_CHECK(COND, ERR) \
  if (!(COND)) {                  \
    return false;                 \
  }
#else
#include <iostream>
#define CHECK_ALIGNED_PTR(PTR, ALIGNMENT)            \
  if (!(uint64_t(PTR) % ALIGNMENT == 0)) {           \
    std::cerr << #PTR " is not correctly aligned\n"; \
    return false;                                    \
  }
#define XFORMERS_CHECK(COND, ERR)                       \
  if (!(COND)) {                                        \
    std::cerr << "'" #COND "' failed: " << ERR << "\n"; \
    return false;                                       \
  }
#endif

#undef ASSIGN_CHECK_OVERFLOW
#define ASSIGN_CHECK_OVERFLOW(A, B)                                    \
  {                                                                    \
    A = B;                                                             \
    XFORMERS_CHECK(                                                    \
        B < std::numeric_limits<decltype(A)>::max(), #B " overflows"); \
  }
