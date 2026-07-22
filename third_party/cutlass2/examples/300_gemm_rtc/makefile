CUDA_ROOT=${CUDA_PATH}/lib
CUTLASS_ROOT=${PWD}/../..

NVCC_FLAGS = -std=c++11 -g \
	-DNVRTC_GET_TYPE_NAME=1 -DCUTLASS_ENABLE_CUBLAS=1 -DCUTLASS_ENABLE_CUDNN=1 \
	-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_DEBUG_TRACE_LEVEL=0 \
	-I${CUTLASS_ROOT}/include -I${CUTLASS_ROOT}/examples/common -I${CUTLASS_ROOT}/tools/util/include \
	-isystem=${CUDA_PATH}/include -Xcompiler=-fPIE -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -x cu \
	-L${CUDA_ROOT}/stubs -L${CUDA_ROOT} -lcudadevrt -lcuda -lcudart -lnvrtc -lrt -lpthread -ldl
#NVCC_FLAGS += -keep-dir=tmp -keep
GENCODES = -gencode=arch=compute_80,code=[sm_80,sm_86]

TARGET = case

${TARGET} : gemm_rtc.cu
	# mkdir -p tmp
	nvcc ${NVCC_FLAGS} ${GENCODES} -o ${TARGET} ./gemm_rtc.cu

.PHONY: clean
clean:
	rm -rf ${TARGET} tmp