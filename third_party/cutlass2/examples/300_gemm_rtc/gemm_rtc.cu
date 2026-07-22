#define CUTLASS_ENABLE_RTC 1

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// Helper methods to check for errors
#include "helper.h"

#include "accutlass.h"
//
// CUTLASS includes needed for single-precision GEMM kernel
//

#if CUTLASS_ENABLE_RTC
const std::string user_pre_headers = std::string("#include \"examples/common/accutlass.h\"\n");
#include "ppurtc.h"
#endif

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass2/gemm/device/gemm.h"
///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, bool ColMajor = true>
class Memory {
public:
  Memory(int row, int col, int stride, bool host_only = false)
      : m_row(row), m_col(col), m_stride(stride), m_hptr(NULL), m_dptr(NULL) {
    m_size = (ColMajor ? col : row) * stride;
    m_hptr = new T[m_size];
    if (!host_only) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_dptr), m_size * sizeof(T)));
      assert(((unsigned long long)m_dptr) % 128 == 0);
    }
  }

  ~Memory() {
    if (m_hptr) {
      delete [] m_hptr;
    }
    if (m_dptr) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_dptr)));
    }
  }

  void initRandom() {
    for (int i = 0; i < m_size; ++i) {
      float tmp = static_cast<float>(std::rand()) * 3.0f / static_cast<float>(RAND_MAX);
      m_hptr[i] = (T)(tmp);
    }
    if (m_dptr) {
      copy(cudaMemcpyHostToDevice);
    }
  }

  void clear() {
    memset(m_hptr, 0, m_size * sizeof(T));
    if (m_dptr) {
      CUDA_CHECK(cudaMemset(m_dptr, 0, m_size * sizeof(T)));
    }
  }

  void copy(cudaMemcpyKind direction) {
    if (m_dptr) {
      switch (direction) {
        case cudaMemcpyHostToDevice:
          CUDA_CHECK(cudaMemcpy(m_dptr, m_hptr, m_size * sizeof(T), direction));
          break;
        case cudaMemcpyDeviceToHost:
          CUDA_CHECK(cudaMemcpy(m_hptr, m_dptr, m_size * sizeof(T), direction));
          break;
        default:
          printf("Unsupport copy!\n");
      }
    }
  }

  T* getDevicePtr() const { return m_dptr; }
  T* getHostPtr()   const { return m_hptr; }
  int row() const { return m_row; }
  int col() const { return m_col; }
  int stride() const { return m_stride; }

  T& at(int m, int n) {
    assert(m < m_row && n < m_col);
    return ColMajor ? m_hptr[m + n * m_row] : m_hptr[m * m_col + n];
  }

private:
  int        m_row;
  int        m_col;
  int        m_stride;
  size_t     m_size;
  T*         m_hptr;
  T*         m_dptr;
};

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
bool CutlassSgemmNN(Memory<float> &a, Memory<float> &b, Memory<float> &c, float alpha, float beta) {
  assert(a.row() == c.row() && b.col() == c.col() && a.col() == b.row());

  int M = a.row();
  int N = b.col();
  int K = a.col();
  int lda = a.stride();
  int ldb = b.stride();
  int ldc = c.stride();

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {a.getDevicePtr(), lda},    // Tensor-ref for source matrix A
                              {b.getDevicePtr(), ldb},    // Tensor-ref for source matrix B
                              {c.getDevicePtr(), ldc},    // Tensor-ref for source matrix C
                              {c.getDevicePtr(), ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //
  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //
  return status == cutlass::Status::kSuccess ? true : false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Reference GEMM computation.
template<typename T>
void ReferenceGemm(Memory<T> &a, Memory<T> &b, Memory<T> &c, T alpha, T beta) {
  assert(a.row() == c.row() && b.col() == c.col() && a.col() == b.row());
  for (int m = 0; m < c.row(); ++m) {
    for (int n = 0; n < c.col(); ++n) {
      T acc = 0;
      for (int k = 0; k < a.col(); ++k) {
        acc += a.at(m, k) * b.at(k, n);
      }
      c.at(m, n) = alpha * acc + beta;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
bool TestCutlassGemm(int M, int N, int K, float alpha, float beta) {
  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //
  Memory<float> matA(M, K, lda);
  Memory<float> matB(K, N, ldb);
  Memory<float> matC(M, N, ldc);
  Memory<float> refC(M, N, ldc, true);

  matA.initRandom();
  matB.initRandom();
  matC.clear();
  refC.clear();

  //
  // Launch CUTLASS GEMM.
  //

  if (!CutlassSgemmNN(matA, matB, matC, alpha, beta)) {
    std::cerr << "CUTLASS GEMM kernel failed." << std::endl;
    return false;
  }
  CUDA_CHECK(cudaStreamSynchronize(0));
  matC.copy(cudaMemcpyDeviceToHost);

  //
  // Verify.
  //
  // Launch reference GEMM
  ReferenceGemm(matA, matB, refC, alpha, beta);

  //
  // Test for bit equivalence of results.
  //
  bool pass = true;
  const float threshold = static_cast<float>(2e-4);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      if (abs (matC.at(m, n) - refC.at(m, n)) > threshold) {
        printf("DIFF at (%d, %d): ref=%f, gpu=%f\n", m, n, refC.at(m, n), matC.at(m, n));
        pass = false;
        break;
      }
    }
    if (!pass) {
      break;
    }
  }
  return pass;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 128, 128, 128 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  //
  // Run the CUTLASS GEMM test.
  //

  bool pass = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  if (pass) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return pass ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
