## compiling with ppu_sdk

```
# compiling with ppu_sdk
export PPU_SDK=YOUR_SDK_DIR
cd 00_basic_gemm
## compiling with nvcc
../common/build_and_run.sh basic_gemm ppu
## compiling with clang
../common/build_and_run.sh basic_gemm ppu clang
```

## compiling with gpu
```
export CUDA_ROOT=CUDA_SDK_DIR
cd 00_basic_gemm
../common/build_and_run.sh basic_gemm gpu
```

## Failed cases
compiling with GPU, failed cases
```
06_splitK_gemm
10_planar_complex
11_planar_complex_array
13_two_tensor_op_fusion
15_ampere_sparse_tensorop_gemm
206_conv_add_biasadd_scale_biasadd_relu_newapi
```

compiling with PPU_SDK, failed cases
```
09_turing_tensorop_conv2dfprop, b4 type not supported
10_planar_complex, Error same with GPU
11_planar_complex_array, Error same with GPU
13_two_tensor_op_fusion, Error same with GPU
15_ampere_sparse_tensorop_gemm, Error same with GPU
```

compiling with PPU_SDK success, running failed
```
18_ampere_sparse_tensorop_conv2dfprop, ERROR - results miscompared.
103_gemm_add_biasadd_mul_biasadd_leakyrelu, compare with ref failed, same with GPU
104_gemm_add_scale_biasadd_leakyrelu, compare with ref failed, GPU success
105_gemm_add_rsqrt_mul_biasadd_leakyrelu, compare with ref failed, GPU success
205_conv_abs_scale_biasadd_relu, compare with ref failed, same with GPU
206_conv_add_biasadd_scale_biasadd_relu_newapi, compare with ref failed, GPU has compiling issue
208_conv_scale_biasadd_relu_newapi, compare with ref failed
```

