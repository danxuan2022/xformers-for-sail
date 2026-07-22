/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
Sparse GEMM kernels needs to takes an additional E matrix which stores the meta data.  The format of
meta data is different for every data types.   CUTLASS templates can automatically infer it based on
input A and B.  Check code below.

Moreover, matrix E needs to be preprocessed so that it can use ldmatrix to load into the registers
efficiently.
*/

#include <iostream>

// PTG cutlass propriatary configs
#include "accutlass.h"

#include "cutlass2/cutlass2.h"
#include "cutlass2/gemm/device/gemm.h"
#include "cutlass2/conv/kernel/default_sparse_conv2d_fprop.h"
#include "cutlass2/conv/device/implicit_gemm_convolution.h"

#include "cutlass2/util/host_tensor.h"
#include "cutlass2/util/host_reorder.h"
#include "cutlass2/util/host_uncompress.h"
#include "cutlass2/util/tensor_view_io.h"
#include "cutlass2/util/reference/host/convolution.h"
#include "cutlass2/util/reference/host/tensor_compare.h"
#include "cutlass2/util/reference/host/tensor_copy.h"
#include "cutlass2/util/reference/host/tensor_fill.h"

#include "cutlass2/util/command_line.h"
#include "helper.h"


// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                 // <- data type of accumulator
using ElementComputeEpilogue = float;             // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;            // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;            // <- data type of elements in input matrix B
using ElementOutput = float;                      // <- data type of elements in output matrix D

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 64>;  // <- threadblock tile M = 128, N = 128, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;  // <- warp tile M = 64, N = 64, K = 64
// This code section describes the size of MMA op

#ifdef __HGGCCC__
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 16, 32>; 
#else
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;     // <- MMA Op tile M = 16, N = 8, K = 16
#endif

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// Number of pipelines you want to use
constexpr int NumStages = 3;

// This code section describe iterator algorithm selected is Analytic or Optimized
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized;

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultSparseConv2dFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ShapeMMAThreadBlock,
  ShapeMMAWarp,
  ShapeMMAOp,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd,
  IteratorAlgorithm
>::Kernel;

// Data type and layout of meta data matrix E can be inferred from template Gemm.
using ElementInputE = typename Conv2dFpropKernel::ElementE;
using LayoutInputE = cutlass::layout::RowMajor;
using ReorderedLayoutInputE = typename Conv2dFpropKernel::LayoutE;

// Blow property is defined in include/cutlass/arch/sp_mma_sm80.h
// 50% Sparsity on Ampere
constexpr int kSparse = Conv2dFpropKernel::kSparse;
// How many elements of A are covered per ElementE
constexpr int kElementsPerElementE = Conv2dFpropKernel::kElementsPerElementE;
// The size of individual meta data
constexpr int kMetaSizeInBits = Conv2dFpropKernel::kMetaSizeInBits;

using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  cutlass::Tensor4DCoord input_size;
  cutlass::Tensor4DCoord filter_size;
  cutlass::Tensor4DCoord padding;
  cutlass::MatrixCoord conv_stride;
  cutlass::MatrixCoord dilation;
  bool reference_check;
  bool measure_performance;
  int iterations;
  bool save_workspace;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32,32, 128),
    filter_size(128, 1, 1, 128),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    reference_check(true),
    measure_performance(true),
    iterations(1),
    save_workspace(false),
    alpha(1),
    beta(0),
    benchmark(false) { }

  // Verify the problem size is compatible with the CUTLASS Convolution implementation.
  bool valid() {

    //
    // CUTLASS attempts to load 128b vectors of cutlass::half_t (F16) elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 8 elements.
    //
    int const kAlignment = 8;

    if ((input_size.c() % kAlignment) ||
      (filter_size.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding.h() != filter_size.h() / 2) ||
      (padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
    cutlass::Tensor4DCoord input_size,
    cutlass::Tensor4DCoord filter_size) {

    this->input_size = input_size;
    this->filter_size = filter_size;

    padding.n() = filter_size.h() / 2;
    padding.h() = filter_size.h() / 2;
    padding.w() = filter_size.w() / 2;
    padding.c() = filter_size.w() / 2;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("ref-check")) {
      reference_check = true;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = true;
    }

    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    filter_size.c() = input_size.c();

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);

    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    if (filter_size.h() == 3 && filter_size.w() == 3) {
      padding = {1, 1, 1, 1};
    }
    else {
      filter_size.h() = 1;
      filter_size.w() = 1;
      padding = {0, 0, 0, 0};
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "16_ampere_tensorop_conv2dfprop example\n\n"
      << "  This example uses Ampere's Tensor Core operators on F16 data types to compute\n"
      << "  forward convolution on tensors of layout NHWC.\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --n <int>            Input tensor extent N\n"
      << "  --h <int>            Input tensor extent H\n"
      << "  --w <int>            Input tensor extent W\n"
      << "  --c <int>            Input tensor extent C\n"
      << "  --k <int>            Filter extent K\n"
      << "  --r <int>            Filter extent R\n"
      << "  --s <int>            Filter extent S\n\n"
      << "  --alpha <float>      Epilogue scalar alpha\n"
      << "  --beta <float>       Epilogue scalar beta\n\n"
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations <int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag <string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/16_ampere_tensorop_conv2dfprop/16_ampere_tensorop_conv2dfprop  --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1\n\n"
      << "$ ./examples/16_ampere_tensorop_conv2dfprop/16_ampere_tensorop_conv2dfprop  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check\n\n";

    return out;
  }

  /// Computes the output tensor size (NPQK)
  cutlass::Tensor4DCoord output_size() const {
    printf("input:%d, filter:%d, stride:%d, padding[%d, %d]\n", input_size.h(), filter_size.h(), conv_stride.row(), padding.n(), padding.h());

    return cutlass::Tensor4DCoord(
      input_size.n(),
      (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
      (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
      filter_size.n());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());

    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cutlass::Status reference_check;
  cudaError_t error;

  Result():
    runtime_ms(0),
    gflops(0),
    status(cutlass::Status::kSuccess),
    reference_check(cutlass::Status::kInvalid),
    error(cudaSuccess) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,H,W,C,K,R,S,Runtime,GFLOPs";

    return out;
  }

  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    out
      << "conv_" << idx << ","
      << options.input_size.n() << ","
      << options.input_size.h() << ","
      << options.input_size.w() << ","
      << options.input_size.c() << ","
      << options.filter_size.n() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << runtime_ms << ","
      << gflops;

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Runs one benchmark
Result profile_convolution(Options const &options) {

  Result result;

  //
  // Allocate host-device tensors using the CUTLASS Utilities.
  //
  // Create matrix A with dimensions NQP x (RSC / 2) by NHWxC/2
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      cutlass::make_Coord(options.input_size.n(), options.input_size.h(), options.input_size.w(), options.input_size.c() / kSparse));
  // <- Create uncompressed matrix A with dimensions NQP x RCS for reference computing
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a_uncompressed(options.input_size);
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.filter_size);
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.output_size());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_c(options.output_size());

  int gemm_m = options.input_size.n() * options.output_size().h() * options.output_size().w();
  int gemm_n = options.filter_size.n();
  int gemm_k = options.filter_size.h() * options.filter_size.w() * options.input_size.c();
  printf("gemm_m:%d, gemm_n:%d, gemm_k:%d, tensor_e:%d\n", gemm_m, gemm_n, gemm_k, gemm_k / kSparse / kElementsPerElementE);

  printf("tensor_a_uncompressed[%d, %d, %d, %d]\n", tensor_a_uncompressed.extent().n(), tensor_a_uncompressed.extent().h(), tensor_a_uncompressed.extent().w(), tensor_a_uncompressed.extent().c());
  // Create matrix E with dimensions NPQ x (RSC / 2 / kElementsPerElementE). This one is used by reference computing.
  cutlass::HostTensor<ElementInputE, LayoutInputE> tensor_e(
        cutlass::make_Coord(gemm_m, gemm_k / kSparse / kElementsPerElementE));

  // Same size as the above.  The above one needs to be reordered and stored in this one.
  cutlass::HostTensor<ElementInputE, ReorderedLayoutInputE> tensor_e_reordered(
      cutlass::make_Coord(gemm_m, gemm_k / kSparse / kElementsPerElementE));

  //
  // Initialize tensors
  //

  // Fill tensor A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(7),
      ElementInputA(-8),
      0);

  // Fill tensor B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(7),
      ElementInputB(-8),
      0);

  // Fill matrix E on host with uniform-distribution random meta data
  cutlass::reference::host::TensorFillRandomSparseMeta(
    tensor_e.host_view(),
    1,
    kMetaSizeInBits);

  // Fill tensor C on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_c.host_view());

  // Fill tensor C for reference on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_c.host_view());

  cutlass::reference::host::TensorFill(
      tensor_a_uncompressed.host_view());

  // Reorder the meta data matrix so that we can use ldmatrix to load them to tensor core
  // instructions.
  cutlass::reorder_meta(tensor_e_reordered.host_ref(), tensor_e.host_ref(),
                         {gemm_m, gemm_n,
                         gemm_k / kSparse / kElementsPerElementE});

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_e_reordered.sync_device();
  tensor_ref_c.sync_device();

  //
  // Define arguments for CUTLASS Convolution
  //

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Construct Conv2dProblemSize with user defined output size
  cutlass::conv::Conv2dProblemSize problem_size(
      options.input_size,
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      options.output_size(),
      mode,
      split_k_slices
  );

  // Construct ImplicitGemm::Argument structure with conv2d
  // problem size, data pointers, and epilogue values
  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    tensor_c.device_ref(),
    tensor_c.device_ref(),
    tensor_e_reordered.device_ref(),
    {options.alpha, options.beta},
  };

  //
  // Initialize CUTLASS Convolution
  //

  ImplicitGemm implicit_gemm_op;

  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  result.status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(result.status);

  result.status = implicit_gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(result.status);

  //
  // Launch initialized CUTLASS kernel
  //
  result.status = implicit_gemm_op();
  CUTLASS_CHECK(result.status);

  //
  // Optional reference check
  //

  if (options.reference_check) {
    std::cout << "Verification on host...\n";

    // uncompress tensor_a based on meta data tensor_e. We need it for reference computing.
    // cutlass::uncompress(tensor_a_uncompressed.host_ref(), tensor_a.host_ref(),
    //                     tensor_e.host_ref(), problem_size.m(), problem_size.k());
    // uncompress tensor_a based on meta data tensor_e. We need it for reference computing by transform form matrix to conv tensor.
    cutlass::uncompress_col2img(tensor_a_uncompressed.host_ref(), tensor_a.host_ref(),
                        tensor_e.host_ref(), options.input_size, options.output_size(), options.filter_size, options.padding, options.conv_stride);
    std::cout << "finish uncompress on host...\n";

    // Compute with reference implementation
    cutlass::reference::host::Conv2dFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      cutlass::NumericConverter<ElementOutput, ElementComputeEpilogue>
    >(
      problem_size,
      tensor_a_uncompressed.host_ref(),
      tensor_b.host_ref(),
      tensor_c.host_ref(),
      tensor_ref_c.host_ref(),
      options.alpha,
      options.beta
    );

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    tensor_c.sync_host();
    bool passed = cutlass::reference::host::TensorEquals(
      tensor_c.host_view(),
      tensor_ref_c.host_view());

    if (!passed) {
      result.reference_check = cutlass::Status::kErrorInternal;
      std::cout << "ERROR - results miscompared.\n";
    }
    else {
      result.reference_check = cutlass::Status::kSuccess;
      std::cout << "Passed.\n";
    }
  }
  else {
    result.reference_check = cutlass::Status::kInvalid;
  }

  if (1) {

    // std::stringstream ss;

    // ss << "18_ampere_workspace_conv2dfprop_"
    //   << options.input_size.n() << "x" << options.input_size.h() << "x" << options.input_size.w() << "x" << options.input_size.c()
    //   << "_"
    //   << options.filter_size.n() << "x" << options.filter_size.h() << "x" << options.filter_size.w() << "x" << options.filter_size.c()
    //   << ".dat";

    // std::ofstream output_workspace(ss.str());

    // output_workspace
    //   << "Input = \n" << tensor_a.host_view() << "\n\n"
    //   << "Input_unpcompress = \n" << tensor_a_uncompressed.host_view() << "\n\n"
    //   << "Meta_data = \n" << tensor_e.host_view() << "\n\n"
    //   << "Meta_data_recorder = \n" << tensor_e_reordered.host_view() << "\n\n"
    //   << "Filters = \n" << tensor_b.host_view() << "\n\n";

    // if (options.reference_check) {
    //   output_workspace << "Reference = \n" << tensor_ref_c.host_view() << "\n\n";
    // }

    // output_workspace << "Computed = \n" << tensor_c.host_view() << std::endl;

    // std::cout << "Results written to '" << ss.str() << "'." << std::endl;
    std::ofstream trace("ppu.csv");
    trace << "real_A = \n" << tensor_a.host_view() << "\n\n";
    trace << "uncompressed_A = \n" << tensor_a_uncompressed.host_view() << "\n\n";
    // trace << "real_B = \n" << tensor_b.host_view() << "\n\n";
    trace << "meta = \n" << tensor_e.host_view() << "\n\n";
    trace << "meta_recorder = \n" << tensor_e_reordered.host_view() << "\n\n";
    trace << "compute = \n" << tensor_c.host_view() << "\n\n";
    trace << "Reference = \n" << tensor_ref_c.host_view() << "\n\n" << std::endl;
  }

  //
  // Performance measurement
  //
  if (options.measure_performance) {

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return result;
      }
    }

    // Record an event at the start of a series of convolution operations.
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Launch a sequence of implicit GEMM operations on the device
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      result.status = implicit_gemm_op();
      CUTLASS_CHECK(result.status);
    }

    // Record an event when the convolutions have been launched.
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Print average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    // Cleanup
    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }
  }

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync are first available in CUDA 11.0.
  //
  // CUTLASS must be compiled with CUDA 11 Toolkit to run Conv2dFprop examples.
  if (!(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 0))) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

  if (!(props.major > 8 || (props.major == 8 && props.minor >= 0))) {
    std::cerr << "Ampere Tensor Ops must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    return 0;
  }

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.benchmark) {
    // Benchmark several layers

    int batch_sizes[] = {1};
    struct Benchmark {
      int h, w, c, k, r, s;
    } layers[] = {
      {8,  8,   64, 8, 3, 3},
      // {56,  56,   64,    64, 3, 3},
      // {56,  56,  256,    64, 1, 1},
      // {56,  56,  256,   512, 1, 1},
      // {56,  56,  256,   128, 1, 1},
      // {28,  28,  128,   128, 3, 3},
      // {28,  28,  128,   512, 1, 1},
      // {28,  28,  512,   128, 1, 1},
      // {28,  28,  512,  1024, 1, 1},
      // {28,  28,  512,   256, 1, 1},
      // {14,  14,  256,   256, 3, 3},
      // {14,  14,  256,  1024, 1, 1},
      // {14,  14,  1024,  256, 1, 1},
      // {14,  14,  1024, 2048, 1, 1},
      // {14,  14,  1024,  512, 1, 1},
      // {7,    7,   512,  512, 3, 3},
    };

    Result::print_header(std::cout, options) << std::endl;

    int idx = 1;

    for (auto const &layer : layers) {
      for (auto N : batch_sizes) {

        options.update({N, layer.h, layer.w, layer.c}, {layer.k, layer.r, layer.s, layer.c});

        Result result = profile_convolution(options);
        result.print(std::cout, idx, options) << std::endl;
      }

      ++idx;
    }
  }
  else {

    // Execute one problem size
    if (!options.valid()) {
      std::cerr << "Invalid problem." << std::endl;
      return -1;
    }

    Result result = profile_convolution(options);
    Result::print_header(std::cout, options) << std::endl;
    result.print(std::cout, 1, options) << std::endl;
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
