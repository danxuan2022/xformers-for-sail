#pragma once

#include "cuda_runtime.h"

#include <string>
#include <memory>
#include <cstdlib>
#include <cstdio>
#include <cxxabi.h>

template<typename T>
std::string get_type_name()
{
    int status;
    std::string tname = typeid(T).name();
    char *demangled_name = abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
    if(status == 0) {
        tname = demangled_name;
        std::free(demangled_name);
    } else {
      printf("demangled failed, status=%d\n", status);
      std::string cmd = std::string("c++filt -r -t ") + tname;
      std::array<char, 40960> buffer;
      std::string result;

      std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
      if (!pipe) {
        throw std::runtime_error("popen() failed!");
      }
      while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
          result += buffer.data();
      }
      return result;
    }
    return tname;
}

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }
