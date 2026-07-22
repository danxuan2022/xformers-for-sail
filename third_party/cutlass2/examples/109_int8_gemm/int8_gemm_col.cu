/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include "accutlass.h"

#include "cutlass2/cutlass2.h"
#if USE_AIU
#include "aiu/gemm/device/aiugemm.h"
#endif
#include "cutlass2/gemm/device/gemm.h"
#include "cutlass2/util/host_tensor.h"
#include "cutlass2/util/reference/device/gemm.h"
#include "cutlass2/util/reference/host/tensor_compare.h"
#include "cutlass2/util/reference/host/tensor_copy.h"
#include "cutlass2/util/reference/host/tensor_fill.h"
#include "cutlass2/util/tensor_view_io.h"
#include "cutlass2/util/host_reorder.h"
#include "helper.h"

  /// Helper to initialize a tensor view
template <typename Element, typename Layout>
bool initialize_tensor(
    cutlass::TensorView<Element, Layout> view, 
    cutlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == cutlass::Distribution::Uniform) {

      cutlass::reference::host::TensorFillRandomUniform(
        view, seed, 127, -128, 0);
    } 
    else if (dist_kind == cutlass::Distribution::Identity) {

      cutlass::reference::host::TensorFillIdentity(view);
    } 
    else if (dist_kind == cutlass::Distribution::Sequential) {

      cutlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    } 
    else {
      // TODO: Implement the rest
      return false;
    }

    return true;
}

template <typename LayoutA, typename LayoutB, typename OpClass, typename Ctype, int AlignmentC = 4, int AlignmentAB = 1>
int int8_gemm_col(const int problem_m, const int problem_n, const int problem_k, float alpha_f, float beta_f){

  std::cout<<"M: "<<problem_m
          <<" N: "<<problem_n
          <<" K: "<<problem_k
          <<" alpha: "<<alpha_f
          <<" beta: "<<beta_f
          <<std::endl;

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(problem_m, problem_n, problem_k);

  using ElementOutput = Ctype;
  using ElementAccumulator = int32_t;
  using ElementComputeEpilogue = typename cutlass::platform::conditional<cutlass::platform::is_same<Ctype, int32_t>::value, int32_t, float>::type;
  #ifdef USE_AIU
  using GemmTensorOp = aiu::gemm::device::Gemm<
  #else
  using GemmTensorOp = cutlass::gemm::device::Gemm<
  #endif
    int8_t,
    LayoutA,
    int8_t,
    LayoutB,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>, //  pass
    #ifdef __HGGCCC__
    cutlass::gemm::GemmShape<16, 16, 32>,
    #else
    cutlass::gemm::GemmShape<16, 8, 32>,
    #endif
    cutlass::epilogue::thread::LinearCombinationClamp<
        ElementOutput,
        4,
        ElementAccumulator,
        ElementComputeEpilogue>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    AlignmentC,
    AlignmentC
  >;

  using GemmSimt = cutlass::gemm::device::Gemm<
    int8_t,
    LayoutA,
    int8_t,
    LayoutB,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<1, 1, 4>,
    cutlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      1,
      ElementAccumulator,
      ElementComputeEpilogue
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    AlignmentAB,
    AlignmentAB
  >;

  using Gemm = typename cutlass::platform::conditional<cutlass::platform::is_same<OpClass, cutlass::arch::OpClassSimt>::value, GemmSimt, GemmTensorOp>::type;

  ElementComputeEpilogue alpha = ElementComputeEpilogue(alpha_f); 
  ElementComputeEpilogue beta = ElementComputeEpilogue(beta_f);

  cutlass::HostTensor<
    typename Gemm::ElementA, 
    typename Gemm::LayoutA> tensor_A(problem_size.mk());

  cutlass::HostTensor<
    typename Gemm::ElementB, 
    typename Gemm::LayoutB> tensor_B(problem_size.kn());

  cutlass::HostTensor<
    typename Gemm::ElementC, 
    typename Gemm::LayoutC> tensor_C(problem_size.mn());

  cutlass::HostTensor<
    typename Gemm::ElementC, 
    typename Gemm::LayoutC> tensor_D(problem_size.mn());

  cutlass::HostTensor<
    typename Gemm::ElementC, 
    typename Gemm::LayoutC> reference_D(problem_size.mn(), false);

  cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind init_C = cutlass::Distribution::Uniform;
  uint64_t seed = 2080;

  initialize_tensor(tensor_A.host_view(), init_A, seed + 2019);
  initialize_tensor(tensor_B.host_view(), init_B, seed + 2018);
  initialize_tensor(tensor_C.host_view(), init_C, seed + 2017);

  cutlass::reference::host::TensorCopy(
    reference_D.host_view(), 
    tensor_C.host_view());

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();

  // Run kernel 
  typename Gemm::Arguments arguments{
    problem_size,
    tensor_A.device_ref(),
    tensor_B.device_ref(),
    tensor_C.device_ref(),
    tensor_D.device_ref(),
    {alpha, beta}
  };
  Gemm gemm_op;
  cutlass::Status status = gemm_op.initialize(arguments);
  CUTLASS_CHECK(status);
  status = gemm_op();
  CUTLASS_CHECK(status);

  // Run host
  using RefHostGemm = cutlass::reference::host::Gemm<
      typename Gemm::ElementA, typename Gemm::LayoutA,
      typename Gemm::ElementB, typename Gemm::LayoutB,
      typename Gemm::ElementC, typename Gemm::LayoutC,
      ElementComputeEpilogue, ElementAccumulator,
      typename Gemm::Operator>;
  
  RefHostGemm reference_gemm;
  reference_gemm(
    problem_size,
    alpha, 
    tensor_A.host_ref(), 
    tensor_B.host_ref(), 
    beta, 
    reference_D.host_ref(), 
    ElementAccumulator(0)
  );

  // Compare results
  tensor_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(
    reference_D.host_view(), 
    tensor_D.host_view());

#ifdef DEBUG
  std::cout<<"======> Gemm::GemmKernel"<<std::endl;
  std::cout<<get_type_name<typename Gemm::GemmKernel>()<<std::endl;

  if (true) {
    std::stringstream fname;
    fname << "error_Gemm_device_" 
      << problem_size.m() << "x"
      << problem_size.n() << "x"
      << problem_size.k() << "_"
      << Gemm::ThreadblockShape::kM << "x"  
      << Gemm::ThreadblockShape::kN << "x"  
      << Gemm::ThreadblockShape::kK << "_"
      << Gemm::WarpShape::kM << "x"  
      << Gemm::WarpShape::kN << "x"  
      << Gemm::WarpShape::kK << ".txt";

    std::ofstream file(fname.str());
    file
      << "problem: " << problem_size 
      << ", alpha: " << alpha << ", beta: " << beta << "\n\n";
    file 
      << "\n A =\n" << tensor_A.host_view()
      << "\n B =\n" << tensor_B.host_view()
      << "\n C =\n" << tensor_C.host_view()
      << "\n Reference =\n" << reference_D.host_view()
      << "\n Computed =\n" << tensor_D.host_view();
  }
#endif

  std::cout << (passed ? "Passed" : "Failed") << std::endl;
  return (passed ? 0  : -1);
}

int main(int argc, const char *argv[]) {
  if (argc !=1 && argc != 6 && argc != 7) {
    printf("arguments missing, needs m, n, k, ctype(int32=0, int8=1, fp32=2, fp16=3), trans(tn, tt, nn, nt), beta\n");
    return -1;
  }

  int problem_m = 64;
  int problem_n = 64;
  int problem_k = 64;
  int ctype = 0;
  std::string trans = "tn";
  bool is_16_byte_align = true;
  bool is_4_byte_align = true;
  float alpha = 1;
  float beta = 0;
  if (argc >= 6) {
    problem_m = std::stoi(std::string(argv[1]));
    problem_n = std::stoi(std::string(argv[2]));
    problem_k = std::stoi(std::string(argv[3]));
    ctype = std::stoi(std::string(argv[4]));
    trans = std::string(argv[5]);
    if (problem_n % 16 != 0 || problem_k % 16 != 0) {
      is_16_byte_align = false;
    }
    if (trans == "tt" && (problem_m % 4 != 0 || problem_k % 4 != 0)){
      is_4_byte_align = false;
      printf("Set aligmentAB to 1 ! \n");
    } else if (trans == "nt" && (problem_m % 4 != 0 || problem_n % 4 != 0)){
      is_4_byte_align = false;
      printf("Set aligmentAB to 1 ! \n");
    } else if (trans == "nn" && (problem_k % 4 != 0 || problem_n % 4 != 0)){
      is_4_byte_align = false;
      printf("Set aligmentAB to 1 ! \n");
    }
    if (argc == 7) {
      beta = std::stof(std::string(argv[6]));
    }
  }
  if (trans != "tn" && problem_n % 4 != 0) {
    printf("Warning: ldc is not 4-byte aligned for simt !\n");
  }

  if (ctype == 0) {
    using Ctype=int32_t;
    if (trans == "tn") {
      if (is_16_byte_align) {
        return int8_gemm_col<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::arch::OpClassTensorOp, Ctype, 16>(
          problem_m, problem_n, problem_k, alpha, beta);
      } else {
        return int8_gemm_col<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::arch::OpClassTensorOp, Ctype, 4>(
          problem_m, problem_n, problem_k, alpha, beta);
      }
    } else if (trans == "tt") {
      if (is_4_byte_align) {
        return int8_gemm_col<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::arch::OpClassSimt, Ctype, 16, 4>(
          problem_m, problem_n, problem_k, alpha, beta);
      } else {
        return int8_gemm_col<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::arch::OpClassSimt, Ctype, 16, 1>(
          problem_m, problem_n, problem_k, alpha, beta);
      }
    } else if (trans == "nt") {
      if (is_4_byte_align) {
        return int8_gemm_col<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::arch::OpClassSimt, Ctype, 16, 4>(
          problem_m, problem_n, problem_k, alpha, beta);
      } else {
        return int8_gemm_col<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::arch::OpClassSimt, Ctype, 16, 1>(
          problem_m, problem_n, problem_k, alpha, beta);
      }
    } else if (trans == "nn") {
      if (is_4_byte_align) {
        return int8_gemm_col<cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::arch::OpClassSimt, Ctype, 16, 4>(
          problem_m, problem_n, problem_k, alpha, beta);
      } else {
        return int8_gemm_col<cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::arch::OpClassSimt, Ctype, 16, 1>(
          problem_m, problem_n, problem_k, alpha, beta);
      }
    } else {
      printf("Invalid trans, only support nt & nn & tt & tn !\n");
      return -1;
    }
  } else if (ctype == 1) {
    using Ctype=int8_t;
    alpha = float(0.001);
    if (trans == "tn") {
      if (is_16_byte_align) {
        return int8_gemm_col<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::arch::OpClassTensorOp, Ctype, 16>(
          problem_m, problem_n, problem_k, alpha, beta);
      } else {
        return int8_gemm_col<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::arch::OpClassTensorOp, Ctype, 4>(
          problem_m, problem_n, problem_k, alpha, beta);
      }
    } else if (trans == "tt") {
      if (is_4_byte_align) {
        return int8_gemm_col<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::arch::OpClassSimt, Ctype, 16, 4>(
          problem_m, problem_n, problem_k, alpha, beta);
      } else {
        return int8_gemm_col<cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::arch::OpClassSimt, Ctype, 16, 1>(
          problem_m, problem_n, problem_k, alpha, beta);
      }
    } else if (trans == "nt") {
      if (is_4_byte_align) {
        return int8_gemm_col<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::arch::OpClassSimt, Ctype, 16, 4>(
          problem_m, problem_n, problem_k, alpha, beta);
      } else {
        return int8_gemm_col<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::arch::OpClassSimt, Ctype, 16, 1>(
          problem_m, problem_n, problem_k, alpha, beta);
      }
    } else if (trans == "nn") {
      if (is_4_byte_align) {
        return int8_gemm_col<cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::arch::OpClassSimt, Ctype, 16, 4>(
          problem_m, problem_n, problem_k, alpha, beta);
      } else {
        return int8_gemm_col<cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::arch::OpClassSimt, Ctype, 16, 1>(
          problem_m, problem_n, problem_k, alpha, beta);
      }
    } else {
      printf("Invalid trans, only support nt & nn & tt & tn !\n");
      return -1;
    }
  } else {
    printf("Unsupported Ctype, only support 0 for int32_t, 1 for int8_t \n");
    return -1;
  }
  return 0;
}

