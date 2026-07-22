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
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Basic include for CUTLASS.
*/

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <sstream>
#include <fstream>

#ifndef checkCudaDrvErrors
#define checkCudaDrvErrors(expr)                                                          \
  do {                                                                                    \
    CUresult __ret = expr;                                                                \
    if (__ret != CUDA_SUCCESS) {                                                          \
      const char *perror = nullptr;                                                       \
      cuGetErrorString(__ret, &perror);                                                   \
      fprintf(stderr, "CUDA Driver error at %s:%d code=%d(%s) \"%s\" \n",                 \
              __FILE__, __LINE__, static_cast<int>(__ret), perror, #expr);                \
      throw std::runtime_error("CUDA driver failed.");                                    \
    }                                                                                     \
  } while (0)
#endif

#ifndef checkCudaErrors
#define checkCudaErrors(expr)                                                             \
  do {                                                                                    \
    expr;                                                                                 \
    cudaError_t __err = cudaGetLastError();                                               \
    if (__err != cudaSuccess) {                                                           \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", __FILE__, __LINE__,    \
              static_cast<int>(__err), cudaGetErrorString(__err), #expr);                 \
      throw std::runtime_error("CUDA runtime failed.");                                   \
    }                                                                                     \
  } while (0)
#endif

static CUmodule upload(const char *filename) {
  CUmodule module;
  std::ostringstream fatbin;
  std::ifstream fileModule(filename, std::ios::binary);
  fatbin << fileModule.rdbuf();
  checkCudaDrvErrors(cuModuleLoadDataEx(&module, fatbin.str().c_str(), 0, 0, 0));
  // checkCudaDrvErrors(cuModuleLoad(&module, filename));
  return module;
}

static void showFuncAttrib(CUfunction kernel) {
  int attr = 0;
  checkCudaDrvErrors(cuFuncGetAttribute(&attr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel));
  std::cout << "[rtc] CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = " << attr << std::endl;
  checkCudaDrvErrors(cuFuncGetAttribute(&attr, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES    , kernel));
  std::cout << "[rtc] CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES     = " << attr << std::endl;
  checkCudaDrvErrors(cuFuncGetAttribute(&attr, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES     , kernel));
  std::cout << "[rtc] CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES      = " << attr << std::endl;
  checkCudaDrvErrors(cuFuncGetAttribute(&attr, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES     , kernel));
  std::cout << "[rtc] CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES      = " << attr << std::endl;
  checkCudaDrvErrors(cuFuncGetAttribute(&attr, CU_FUNC_ATTRIBUTE_NUM_REGS             , kernel));
  std::cout << "[rtc] CU_FUNC_ATTRIBUTE_NUM_REGS              = " << attr << std::endl;
  checkCudaDrvErrors(cuFuncGetAttribute(&attr, CU_FUNC_ATTRIBUTE_PTX_VERSION          , kernel));
  std::cout << "[rtc] CU_FUNC_ATTRIBUTE_PTX_VERSION           = " << attr << std::endl;
  checkCudaDrvErrors(cuFuncGetAttribute(&attr, CU_FUNC_ATTRIBUTE_BINARY_VERSION       , kernel));
  std::cout << "[rtc] CU_FUNC_ATTRIBUTE_BINARY_VERSION        = " << attr << std::endl;
}

#if CUTLASS_ENABLE_RTC

#include <vector>
#include <nvrtc.h>
#include "nvrtc/assert.h"
#include "nvrtc/stdint.h"
#include "nvrtc/device_kernel.h"

#ifndef checkNVrtcErrors
#define checkNVrtcErrors(expr)                                                            \
  do {                                                                                    \
    nvrtcResult __ret = expr;                                                             \
    if (__ret != NVRTC_SUCCESS) {                                                         \
      fprintf(stderr, "NVRTC error at %s:%d code=%d(%s) \"%s\" \n", __FILE__, __LINE__,   \
              static_cast<int>(__ret), nvrtcGetErrorString(__ret), #expr);                \
      throw std::runtime_error("NVRTC failed.");                                          \
    }                                                                                     \
  } while (0)
#endif

#if NVRTC_GET_TYPE_NAME
inline nvrtcResult __nvrtcGetTypeName(const std::type_info &tinfo, std::string *result) {
  const char *name = tinfo.name();
  int status;
  char *undecorated_name = abi::__cxa_demangle(name, 0, 0, &status);
  // std::cout << "[rtc] mangled_name:" << std::endl << name << std::endl;
  if (status == 0) {
    *result = undecorated_name;
    free(undecorated_name);
    return NVRTC_SUCCESS;
  }

  return NVRTC_ERROR_INTERNAL_ERROR;
}

template <typename T>
nvrtcResult __nvrtcGetTypeName(std::string *result) {
  nvrtcResult res = __nvrtcGetTypeName(typeid(__nvrtcGetTypeName_helper_t<T>), result);
  if (res != NVRTC_SUCCESS)
    return res;

  std::string repr = *result;
  std::size_t idx = repr.find("__nvrtcGetTypeName_helper_t");
  idx = (idx != std::string::npos) ? repr.find("<", idx) : idx;
  std::size_t last_idx = repr.find_last_of('>');
  if (idx == std::string::npos || last_idx == std::string::npos) {
    return NVRTC_ERROR_INTERNAL_ERROR;
  }
  ++idx;
  *result = repr.substr(idx, last_idx - idx);
  return NVRTC_SUCCESS;
}
#endif // #if NVRTC_GET_TYPE_NAME

static CUfunction rtCompile(const std::string &prog_src, const std::string &kernel_instantiation) {
    char const *stdHeaders[] = {
      cutlass::nvrtc::assert_h,
      cutlass::nvrtc::stdint_h,
    };

    char const *stdHeaderNames[] = {
      "assert.h",
      "stdint.h",
    };

    const char *ppu_home = std::getenv("PPU_HOME");
    if (!ppu_home) {
      std::cerr << "[ERROR] Cannot find enviorment: PPU_HOME" << std::endl;
      exit(1);
    }
#if 0
    const char *cuda_path = std::getenv("CUDA_PATH");
    if (!cuda_path) {
      std::cerr << "[WARNING] Cannot find enviorment: CUDA_PATH, use default one." << std::endl;
    }
#else
    const char *cuda_path = "/usr/local/cuda";
#endif

    const std::string opt_cuda_inc    = "--include-path=" + std::string(cuda_path) + "/include";
    const std::string opt_cuda_std    = "--include-path=" + std::string(cuda_path) + "/include/cuda/std";
    const std::string opt_cutlass     = "--include-path=" + std::string(ppu_home) + "/cutlass";
    const std::string opt_cutlass_inc = "--include-path=" + std::string(ppu_home) + "/cutlass/include";

    std::vector<const char*> options = {
      "--gpu-architecture=compute_80",
      "--std=c++11",
      "--generate-line-info",
      //"--device-as-default-execution-space",
      opt_cuda_inc.c_str(),
      opt_cuda_std.c_str(),
      opt_cutlass.c_str(),
      opt_cutlass_inc.c_str(),
    };

    nvrtcProgram program;
    checkNVrtcErrors(nvrtcCreateProgram(&program, prog_src.c_str(), "kernel.cu", 2, stdHeaders, stdHeaderNames));
    // std::cout << "[rtc] program source:" << std::endl << prog_src << std::endl;

    checkNVrtcErrors(nvrtcAddNameExpression(program, kernel_instantiation.c_str()));
    if (nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data()) != NVRTC_SUCCESS) {
      size_t log_size;
      checkNVrtcErrors(nvrtcGetProgramLogSize(program, &log_size));
      std::vector<char> log(log_size);
      checkNVrtcErrors(nvrtcGetProgramLog(program, log.data()));
      std::cerr << "Compile Failed:" << std::endl << log.data() << std::endl;
    }
    std::cout << "[rtc] compile success." << std::endl;

    // Query the size of the genereated PTX/CUBIN so that we can allocate storage and retrieve it afterwards
    size_t ptx_size;
    checkNVrtcErrors(nvrtcGetPTXSize(program, &ptx_size));
    std::vector<char> ptx(ptx_size);
    checkNVrtcErrors(nvrtcGetPTX(program, ptx.data()));
    std::string sptx = std::string(ptx.begin(), ptx.end());
    //std::cout << "[rtc] compiled ptx:" << std::endl << sptx << std::endl;

    // size_t cubin_size;
    // checkNVrtcErrors(nvrtcGetCUBINSize(program, &cubin_size));
    // std::vector<char> cubin(cubin_size);
    // if (cubin_size > 0) {
    //   checkNVrtcErrors(nvrtcGetCUBIN(program, cubin.data()));
    //   std::string scubin = std::string(cubin.begin(), cubin.end());
    //   // std::cout << "[rtc] compiled cubin:" << std::endl << scubin << std::endl;
    // }

    char const *kernel_lowered;
    checkNVrtcErrors(nvrtcGetLoweredName(program, kernel_instantiation.c_str(), &kernel_lowered));
    // std::cout << "[rtc] lowered name:" << std::endl << kernel_lowered << std::endl;

    CUmodule module;
    CUfunction kernel;
    checkCudaDrvErrors(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
    checkCudaDrvErrors(cuModuleGetFunction(&kernel, module, kernel_lowered));
    // showFuncAttrib(kernel);

    // we do not need the nvrtc program anymore
    checkNVrtcErrors(nvrtcDestroyProgram(&program));

    return kernel;
}

#endif // #if CUTLASS_ENABLE_RTC
////////////////////////////////////////////////////////////////////////////////////////////////////

