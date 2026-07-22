#pragma once

// macros for cutlass changes required by ali customized logic, must be defined before cutlass includes
#ifdef __HGGCCC__
#ifndef ACOMPUTE_VERSION
#define ACOMPUTE_VERSION 10000
#endif
#define SAIL_PPU_MMA                 1
// [TODO] workaround code temporarily, will double check later.
#define SAIL_TMP_WORKAROUND          1
#else
#define SAIL_PPU_MMA                 0
#define SAIL_TMP_WORKAROUND          0
#endif

#define SAIL_ENABLE_CUTLASS_WMMA     0
#define SAIL_ENABLE_CUTLASS_SIMT     0
#define SAIL_SIMULATE_CUTLASS_MMA    0
#define SAIL_CUSTOMIZE_CUTLASS       1
#define SAIL_DGRAD_STRIDE_OPT        1
#define SAIL_WGRAD_ITER_OPT          1
#define SAIL_VALID_COPY_ONLY         0  // avoid invalid copy in pipeline/multistage, have some problem when main loop < stage
#define SAIL_EPILOGUE_OPT            2  // >=1: skipped identity output op; >=2: bring type convert before writing to shm
#define SAIL_TYPE_CONVERT            1  // 0: use c-code RTN; 1: use build-in intrinsic rna; 2: use build-in intrinsic, 3: use c-code RTN w/o check inf.
#define SAIL_REDUCE_SPLITK_OPT       1
#define SAIL_FUSE_OP_EXT             1
#define SAIL_FUSE_ALIGN_CUDNN        1  // align to cudnn's fusion flow: conv_result(FP32 -> FP16) + Add(FP16) + BiasAdd(FP16) + ReLu
