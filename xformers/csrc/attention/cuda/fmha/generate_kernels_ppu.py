# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Generates combination of kernels - implementations and registry

# Kernels are ordered (see `sort_index`), and when dispatching,
# we select the first kernel in the list that supports the inputs

import collections
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypeVar


DTYPES = {
    "f32": "float",
    "f16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [80, 89]  # Sm80 kernels support up to Sm100

KERNEL_IMPL_TEMPLATE = """__global__ void __launch_bounds__(
    {CPP_CLASS}::kNumThreads,
    {CPP_CLASS}::kMinBlocksPerSm)
{NAME}(typename {CPP_CLASS}::Params p) {{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == {SM}0
  if (!p.advance_to_block()) {{
    return;
  }}
  {CPP_CLASS}::attention_kernel(p);
  return;
#elif __CUDA_ARCH__ == {SM_MAX}0
  return;
#else
  printf(
        "FATAL: kernel `{NAME}` is for sm{SM}/sm{SM_MAX}, but was built for sm%d\\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif  // __CUDA_ARCH__
}}
"""


@dataclass(order=True)
class FwdKernelFA2:
    sort_index: tuple[int, ...] = field(init=False, repr=False)
    version: str
    aligned: bool
    dtype: str
    sm_range: tuple[int, int]
    head_dim: int
    q_block: int
    k_block: int
    q_warp: int
    k_warp: int
    supports_dropout: bool = True
    supports_bias: bool = True
    dispatch_cond: Optional[str] = None

    def __post_init__(self) -> None:
        # Set kernel selection priority
        # The lowest value that matches inputs
        # will be selected
        self.sort_index = (
            # First select aligned kernel
            0 if self.aligned else 1,
            # Prefer kernels without dropout/bias if available
            1 if self.supports_dropout else 0,
            1 if self.supports_bias else 0,
        )

    @property
    def _aligned_suffix(self) -> str:
        return "aligned" if self.aligned else "notaligned"

    @property
    def name(self) -> str:
        # acc = "rf" if self.max_k <= self.k else "gmem"
        dropout = "_dropout" if self.supports_dropout else ""
        return f"fmha_cutlassF_{self.dtype}_{self._aligned_suffix}_k{self.head_dim}x{self.q_block}x{self.k_block}{dropout}_sm{self.sm_range[0]}"

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                DTYPES[self.dtype],
                f"cutlass::arch::Sm{self.sm_range[0]}",
                "true" if self.aligned else "false",
                str(self.head_dim),
                str(self.q_block),
                str(self.k_block),
                str(self.q_warp),
                str(self.k_warp),
                "true" if self.supports_dropout else "false",
                "true" if self.supports_bias else "false",
            ]
        )
        return f"AttentionKernelFA2<{template_args}>"

    @property
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        return f"{self.dtype}_{self._aligned_suffix}"

    @property
    def cpp_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[-1],
        )

    @classmethod
    def get_all(cls) -> list["FwdKernelFA2"]:
        kernels: list[FwdKernelFA2] = []
        for aligned, dtype, (
            sm,
            sm_max,
        ), supports_dropout, supports_bias in itertools.product(
            [True, False],  # aligned
            # DTYPES.keys(),  # dtype
            ["bf16", "f16"],
            zip(SM, SM[1:]),
            [False, True],  # supports_dropout
            [True],  # supports_bias
        ):
            # Remove some kernels we don't use
            if dtype == "bf16" and sm < 80:
                continue
            if not aligned and sm >= 80:
                continue
            for head_dim, q_block, k_block, q_warp, k_warp in [
                (32, 128, 64, 32, 64),
                (48, 128, 32, 32, 32),
                (64, 128, 64, 32, 64),
                (80, 128, 64, 32, 64),
                (96, 128, 64, 32, 32),
                (128, 128, 64, 32, 64),
                (160, 256, 128, 32, 64),
                (256, 256, 32, 32, 32),
            ]:
                kernels.append(
                    cls(
                        version="fa2",
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        head_dim=head_dim,
                        q_block=q_block,
                        k_block=k_block,
                        q_warp=q_warp,
                        k_warp=k_warp,
                        supports_dropout=supports_dropout,
                        supports_bias=supports_bias,
                    )
                )
        return kernels


@dataclass(order=True)
class FwdKernel:
    sort_index: tuple[int, ...] = field(init=False, repr=False)
    version: str
    aligned: bool
    dtype: str
    sm_range: tuple[int, int]
    q: int
    k: int
    maxk: int
    single_value_iteration: bool
    supports_dropout: bool = True
    supports_bias: bool = True
    dispatch_cond: Optional[str] = None

    def __post_init__(self) -> None:
        # Set kernel selection priority
        # The lowest value that matches inputs
        # will be selected
        self.sort_index = (
            # First select aligned kernel
            0 if self.aligned else 1,
            # Then keep output in RF
            # self.max_k,
            self.k,
            0 if self.single_value_iteration else 1,
            # Prefer kernels without dropout/bias if available
            1 if self.supports_dropout else 0,
            1 if self.supports_bias else 0,
        )
        print(f"name = {self.name} sort_index = {self.sort_index}")

    @property
    def _aligned_suffix(self) -> str:
        return "aligned" if self.aligned else "notaligned"

    @property
    def name(self) -> str:
        if self.single_value_iteration:
            self.max_k = 64 if self.k == 64 else 128
        else:
            self.max_k = 65536
        acc = "rf" if self.max_k <= self.k else "gmem"
        # return f"fmha_cutlassF_{self.dtype}_{self._aligned_suffix}_k{self.head_dim}x{self.q_block}x{self.k_block}_sm{self.sm_range[0]}"
        return f"fmha_cutlassF_{self.dtype}_{self._aligned_suffix}_{self.q}x{self.k}_k{self.maxk}_{acc}_sm{self.sm_range[0]}"

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                DTYPES[self.dtype],
                f"cutlass::arch::Sm{self.sm_range[0]}",
                "true" if self.aligned else "false",
                str(self.q),
                str(self.k),
                "true" if self.single_value_iteration else "false",
                "true" if self.supports_dropout else "false",
                "true" if self.supports_bias else "false",
                str(self.maxk),
            ]
        )
        return f"AttentionKernel<{template_args}>"

    @property
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        return f"{self.dtype}_{self._aligned_suffix}"

    @property
    def cpp_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[-1],
        )

    @classmethod
    def get_all(cls) -> list["FwdKernel"]:
        kernels: list[FwdKernel] = []
        for aligned, dtype, (sm, sm_max) in itertools.product(
            [True, False],  # aligned
            DTYPES.keys(),  # dtype
            # ['f32'],
            zip(SM, SM[1:]),  # (sm, sm_max)
        ):
            # Remove some kernels we don't use
            # if dtype == "bf16" and sm < 80:
            #     continue
            if not aligned and sm >= 80:
                continue
            for q, k, maxk, single_value_iteration, supports_dropout, supports_bias in [
                (64, 64, 64, True, True, True),
                (32, 128, 128, True, True, True),
                (32, 128, 65536, False, True, True),
            ]:
                kernels.append(
                    cls(
                        version="fa1",
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        q=q,
                        k=k,
                        maxk=maxk,
                        single_value_iteration=single_value_iteration,
                        supports_dropout=supports_dropout,
                        supports_bias=supports_bias,
                    )
                )
        return kernels


@dataclass(order=True)
class BwdKernelFA2:
    sort_index: tuple[int, ...] = field(init=False, repr=False)
    version: str
    sm_range: tuple[int, int]
    dtype: str
    aligned: bool
    apply_dropout: bool
    preload_mmas: bool  # set true
    block_i: int
    block_j: int
    max_k: int
    warp_num: int
    warp_Dp: int
    warp_Kv: int
    warp_Dq: int
    dispatch_cond: Optional[str] = None
    keys_queries_aligned_to_blocksizes: bool = False
    kenable_split_keys: bool = False
    kenable_v2: bool = False

    def __post_init__(self) -> None:
        # Set kernel selection priority
        # The lowest value that matches inputs
        # will be selected
        self.sort_index = (
            # First select aligned kernel
            0 if self.aligned else 1,
            # Take a kernel without dropout if possible
            1 if self.apply_dropout else 0,
            # Then take the smallest maxK
            self.max_k,
            # .. and the highest block_i
            -self.block_i,
            # and finally avoid bounds-checks if possible
            0 if self.keys_queries_aligned_to_blocksizes else 1,
        )

    @property
    def _aligned_suffix(self) -> str:
        return "aligned" if self.aligned else "notaligned"

    @property
    def name(self) -> str:
        dropout_suffix = "_dropout" if self.apply_dropout else ""
        seqlen_aligned_suffix = (
            "_seqaligned" if self.keys_queries_aligned_to_blocksizes else ""
        )
        return (
            f"fmha_cutlassB_{self.dtype}_{self._aligned_suffix}"
            f"_{self.block_i}x{self.block_j}_k{self.max_k}{dropout_suffix}{seqlen_aligned_suffix}_sm{self.sm_range[0]}"
        )

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                f"cutlass::arch::Sm{self.sm_range[0]}",
                DTYPES[self.dtype],
                "true" if self.aligned else "false",
                "true" if self.apply_dropout else "false",
                # "true" if self.preload_mmas else "false",
                "true",
                str(self.block_i),
                str(self.block_j),
                str(self.max_k),
                "true" if self.keys_queries_aligned_to_blocksizes else "false",
                "true" if self.kenable_split_keys else "false",
                "true" if self.kenable_v2 else "false",
                str(self.warp_num),
                str(self.warp_Dp),
                str(self.warp_Kv),
                str(self.warp_Dq),
            ]
        )
        # if self.keys_queries_aligned_to_blocksizes:
        #     template_args += ", true"
        return f"AttentionBackwardKernel<{template_args}>"

    @property
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        dropout_suffix = "_dropout" if self.apply_dropout else ""
        return f"{self.dtype}_{self._aligned_suffix}_k{self.max_k}{dropout_suffix}"

    @property
    def cpp_impl(self) -> str:
        # import pdb;pdb.set_trace()
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[-1],
        )

    @classmethod
    def get_all(cls) -> list["BwdKernelFA2"]:
        kernels: list[BwdKernel] = []
        for (
            aligned,
            dtype,
            (sm, sm_max),
            apply_dropout,
            preload_mmas,
            keys_queries_aligned_to_blocksizes,
            kenable_split_keys,
            kenable_v2,
        ) in itertools.product(
            [True],
            # DTYPES.keys(),
            ["bf16", "f16"],
            zip(SM, SM[1:]),
            [True, False],  # apply_dropout true/false
            [True],  # preload_mmas True
            [True, False],  # keys_queries_aligned_to_blocksizes true/false
            [True],  # kenable_split_keys all true
            [True],  # kenable_v2 all true
            # [32, 64, 128, 2**16],
        ):
            for block_i, block_j, max_k, warp_num, warp_Dp, warp_Kv, warp_Dq in [
                (128, 128, 32, 8, 4, 4, 4),
                (128, 128, 64, 8, 2, 4, 4),
                (64, 128, 96, 8, 2, 4, 4),
                (64, 128, 128, 8, 2, 4, 2),
                (64, 64, 160, 8, 2, 4, 4),
                (64, 64, 192, 8, 2, 2, 2),
                (64, 64, 224, 8, 2, 4, 4),
                (64, 64, 256, 8, 2, 2, 2),
            ]:
                if dtype == "bf16" and sm < 80:
                    continue
                if not aligned and sm >= 80:
                    continue
                is_half = dtype in ["bf16", "f16"]

                kernels.append(
                    cls(
                        version="fa2",
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        apply_dropout=apply_dropout,
                        preload_mmas=preload_mmas,
                        block_i=block_i,
                        block_j=block_j,
                        max_k=max_k,
                        keys_queries_aligned_to_blocksizes=keys_queries_aligned_to_blocksizes,
                        kenable_split_keys=kenable_split_keys,
                        kenable_v2=kenable_v2,
                        warp_num=warp_num,
                        warp_Dp=warp_Dp,
                        warp_Kv=warp_Kv,
                        warp_Dq=warp_Dq,
                    )
                )
        return kernels


@dataclass(order=True)
class BwdKernel:
    sort_index: tuple[int, ...] = field(init=False, repr=False)
    version: str
    sm_range: tuple[int, int]
    dtype: str
    aligned: bool
    apply_dropout: bool
    preload_mmas: bool  # set true
    block_i: int
    block_j: int
    max_k: int
    dispatch_cond: Optional[str] = None
    keys_queries_aligned_to_blocksizes: bool = False
    kenable_split_keys: bool = False
    kenable_v2: bool = False

    def __post_init__(self) -> None:
        # Set kernel selection priority
        # The lowest value that matches inputs
        # will be selected
        self.sort_index = (
            # First select aligned kernel
            0 if self.aligned else 1,
            # Take a kernel without dropout if possible
            1 if self.apply_dropout else 0,
            # Then take the smallest maxK
            self.max_k,
            # .. and the highest block_i
            -self.block_i,
            # and finally avoid bounds-checks if possible
            0 if self.keys_queries_aligned_to_blocksizes else 1,
        )

    @property
    def _aligned_suffix(self) -> str:
        return "aligned" if self.aligned else "notaligned"

    @property
    def name(self) -> str:
        dropout_suffix = "_dropout" if self.apply_dropout else ""
        seqlen_aligned_suffix = (
            "_seqaligned" if self.keys_queries_aligned_to_blocksizes else ""
        )
        return (
            f"fmha_cutlassB_{self.dtype}_{self._aligned_suffix}"
            f"_{self.block_i}x{self.block_j}_k{self.max_k}{dropout_suffix}{seqlen_aligned_suffix}_sm{self.sm_range[0]}"
        )

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                f"cutlass::arch::Sm{self.sm_range[0]}",
                DTYPES[self.dtype],
                "true" if self.aligned else "false",
                "true" if self.apply_dropout else "false",
                "true" if self.preload_mmas else "false",
                str(self.block_i),
                str(self.block_j),
                str(self.max_k),
                "true" if self.keys_queries_aligned_to_blocksizes else "false",
            ]
        )
        # if self.keys_queries_aligned_to_blocksizes:
        #     template_args += ", true"
        return f"AttentionBackwardKernel<{template_args}>"

    @property
    def impl_group(self) -> str:
        # Maps to file which will contain the implementation
        dropout_suffix = "_dropout" if self.apply_dropout else ""
        return f"{self.dtype}_{self._aligned_suffix}_k{self.max_k}{dropout_suffix}"

    @property
    def cpp_impl(self) -> str:
        # import pdb;pdb.set_trace()
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name,
            SM=self.sm_range[0],
            SM_MAX=self.sm_range[-1],
        )

    @classmethod
    def get_all(cls) -> list["BwdKernel"]:
        kernels: list[BwdKernel] = []
        for (
            aligned,
            dtype,
            (sm, sm_max),
            apply_dropout,
            preload_mmas,
            keys_queries_aligned_to_blocksizes,
        ) in itertools.product(
            [True],  # aligned: true in fa1 bwd
            DTYPES.keys(),  # dtype: f32/f16/bf16
            zip(SM, SM[1:]),
            [True, False],  # apply_dropout true/false
            [False],  # preload_mmas all false for bf16/f16 headdim>256 and f32
            [
                False
            ],  # keys_queries_aligned_to_blocksizes all false for bf16/f16 headdim>256 and f32
            # [True],         # kenable_split_keys all true
            # [True]          # kenable_v2 all true
            # [32, 64, 128, 2**16],
        ):
            if dtype in ["f32"]:
                for block_i, block_j, max_k in [
                    (128, 128, 32),
                    (128, 128, 64),
                    (128, 128, 128),
                    (64, 128, 128),
                    (128, 128, 256),
                    (64, 128, 256),
                    (128, 128, 65536),
                    (64, 128, 65536),
                ]:
                    # preload_mmas and keys_queries_aligned_to_blocksizes: All false for bf16/f16 headdim>256 and f32
                    kernels.append(
                        cls(
                            version="fa1",
                            aligned=aligned,
                            dtype=dtype,
                            sm_range=(sm, sm_max),
                            apply_dropout=apply_dropout,
                            preload_mmas=preload_mmas,
                            block_i=block_i,
                            block_j=block_j,
                            max_k=max_k,
                            keys_queries_aligned_to_blocksizes=keys_queries_aligned_to_blocksizes,
                        )
                    )
            # preload_mmas and keys_queries_aligned_to_blocksizes: All false for bf16/f16 headdim>256 and f32
            elif dtype in ["f16", "bf16"]:
                # import pdb;pdb.set_trace()
                for block_i, block_j, max_k in [(128, 128, 65536), (64, 128, 65536)]:
                    kernels.append(
                        cls(
                            version="fa1",
                            aligned=aligned,
                            dtype=dtype,
                            sm_range=(sm, sm_max),
                            apply_dropout=apply_dropout,
                            preload_mmas=preload_mmas,
                            block_i=block_i,
                            block_j=block_j,
                            max_k=max_k,
                            keys_queries_aligned_to_blocksizes=keys_queries_aligned_to_blocksizes,
                        )
                    )
            else:
                # import pdb;pdb.set_trace()
                print("dtype not supported!")
        # import pdb;pdb.set_trace()
        return kernels


T = TypeVar("T", FwdKernel, FwdKernelFA2, BwdKernel, BwdKernelFA2)


def write_decl_impl_with_fa1(
    kernels: list[T],
    family_name: str,
    impl_file: str,
    impl_file_2: str,
    disable_def: str,
) -> None:
    cpp_file_header = """/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
"""
    if len(kernels) == 2:
        kernels_0 = kernels[0]
        kernels_1 = kernels[1]
        kernels_0.sort()
        kernels_1.sort()
        kernels = kernels_0 + kernels_1
    else:
        kernels.sort()

    implfile_to_kernels: dict[str, list[T]] = collections.defaultdict(list)
    cat_to_kernels: dict[tuple[str, int, int], list[T]] = collections.defaultdict(list)
    cat_to_kernels_with_version: dict[tuple[str, int, int], list[T]] = (
        collections.defaultdict(list)
    )

    # for cutlassF/B.h
    dispatch_all = ""
    dispatch_all_fa1 = ""
    dispatch_all_fa2 = ""
    declarations = cpp_file_header + "#pragma once\n"
    declarations += """
#if defined(__HGGCCC__)
#ifndef ENABLE_AIU
#define ENABLE_AIU 1
#endif
#else
#define ENABLE_AIU 0
#endif
"""
    declarations += f"#ifndef {disable_def}\n"
    declarations += f"""#include "extension/fmha/{impl_file}"\n"""
    declarations += f"""#include "extension/fmha/{impl_file_2}"\n"""

    # Declaration of kernel functions
    for k in kernels:
        implfile_to_kernels[k.impl_group].append(k)
        # add k.version to seperate fa1/fa2 kernels
        cat_to_kernels[(k.dtype, k.sm_range[0], k.sm_range[1])].append(k)
        cat_to_kernels_with_version[
            (k.version, k.dtype, k.sm_range[0], k.sm_range[1])
        ].append(k)
    for (
        cat_version,
        cat_dt,
        cat_sm,
        cat_sm_max,
    ), kernels in cat_to_kernels_with_version.items():
        declarations += f"// ======== {cat_dt} / sm{cat_sm} / {cat_version} ======== \n"
        # ---- __global__ void __launch_bounds__( part
        declarations += "\n".join(
            k.cpp_impl.split("{")[0].rstrip() + ";" for k in kernels
        )
        # ---- template <typename T> void dispatch_cutlassF_f32_sm80(T cb, int cc) {  part
        dispatch_category_fn = (
            f"dispatch_{family_name}_{cat_dt}_sm{cat_sm}_{cat_version}"
        )
        declarations += (
            f"\n\ntemplate <typename T> void {dispatch_category_fn}(T cb, int cc) {{\n"
        )
        # ---- cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, true, true, true>(), fmha_cutlassF_f32_aligned_64x64_rf_sm80); part
        for k in kernels:
            _call = f"cb({k.cpp_class}(), {k.name});\n"
            if k.dispatch_cond is not None:
                _call = f"if ({k.dispatch_cond}) {_call}"
            declarations += f"    {_call}"
            # print(f"kernel = {k}")
        declarations += "}\n\n"
    #     dispatch_all += f"""
    # if (std::is_same<DT, {DTYPES[cat_dt]}>::value && {cat_sm} <= cc && cc < {cat_sm_max}) {{
    #     {dispatch_category_fn}(cb, cc);
    # }}"""

    for (
        cat_version,
        cat_dt,
        cat_sm,
        cat_sm_max,
    ), kernels in cat_to_kernels_with_version.items():
        # print(cat_version, cat_dt, cat_sm, cat_sm_max)
        dispatch_category_fn = (
            f"dispatch_{family_name}_{cat_dt}_sm{cat_sm}_{cat_version}"
        )
        if cat_version == "fa1":
            dispatch_all_fa1 += f"""
        if (std::is_same<DT, {DTYPES[cat_dt]}>::value && {cat_sm} <= cc && cc < {cat_sm_max}) {{
        {dispatch_category_fn}(cb, cc);
    }}"""
        if cat_version == "fa2":
            dispatch_all_fa2 += f"""
        if (std::is_same<DT, {DTYPES[cat_dt]}>::value && {cat_sm} <= cc && cc < {cat_sm_max}) {{
        {dispatch_category_fn}(cb, cc);
    }}"""
        # import pdb;pdb.set_trace()

    declarations += f"""
template <typename DT, typename T>
void dispatch_{family_name}_fa1(T cb, int cc = 0) {{
{dispatch_all_fa1}
}}
"""
    declarations += f"""
template <typename DT, typename T>
void dispatch_{family_name}_fa2(T cb, int cc = 0) {{
{dispatch_all_fa2}
}}
"""

    #     for (cat_version, cat_dt, cat_sm, cat_sm_max), kernels in cat_to_kernels_with_version.items():
    #         dispatch_category_fn = f"dispatch_{family_name}_{cat_dt}_sm{cat_sm}_{cat_version}"
    # # add head_dim for choose fa1/fa2
    # # dispatch_cutlassB_fa1/fa2
    #         declarations += f"""
    #     template <typename DT, typename T>
    #     void dispatch_{family_name}_{cat_version}(T cb, int cc = 0, int head_dim = 0) {{
    #     {dispatch_all}
    #     }}
    #     """

    # Add MaxkTrait()
    declarations += """
// kNumWarpsPerBlock = kQueriesPerBlock * kKeysPerBlock / (32 * 32);
template <int kNumWarpsPerBlock, int kKeysPerBlock, bool kSingleValueIteration>
struct MaxkTrait {
    static constexpr int kMaxK = 0;
};

template <>
struct MaxkTrait<4, 64, true> {
    static constexpr int kMaxK = 64;
};
template <>
struct MaxkTrait<4, 128, true> {
    static constexpr int kMaxK = 128;
};

template <>
struct MaxkTrait<4, 128, false> {
    static constexpr int kMaxK = 65536;
};
"""
    declarations += f"#endif // {disable_def}\n"

    # write cutlassF/B.h
    autogen_dir = Path(__file__).parent / "autogen_ppu"

    (autogen_dir / f"{family_name}.h").write_text(declarations)

    # for impl/*.cu
    for f, f_kernels in implfile_to_kernels.items():
        impl_cu = cpp_file_header
        impl_cu += """
#if defined(__HGGCCC__)
#ifndef ENABLE_AIU
#define ENABLE_AIU 1
#endif
#else
#define ENABLE_AIU 0
#endif
"""
        impl_cu += f"#ifndef {disable_def}\n"
        impl_cu += """#include "accutlass.h"\n"""
        impl_cu += f"""#include "extension/fmha/{impl_file}"\n"""
        impl_cu += f"""#include "extension/fmha/{impl_file_2}"\n"""
        for k in f_kernels:
            impl_cu += k.cpp_impl
        impl_cu += f"#endif // {disable_def}\n"

        # write impl/*.cu
        (autogen_dir / "impl" / f"{family_name}_{f}.cu").write_text(impl_cu)


def write_decl_impl(
    kernels: list[T], family_name: str, impl_file: str, disable_def: str
) -> None:
    cpp_file_header = """/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// This file is auto-generated. See "generate_kernels.py"
"""

    kernels.sort()

    implfile_to_kernels: dict[str, list[T]] = collections.defaultdict(list)
    cat_to_kernels: dict[tuple[str, int, int], list[T]] = collections.defaultdict(list)

    dispatch_all = ""
    declarations = cpp_file_header + "#pragma once\n"
    declarations += """
#if defined(__HGGCCC__)
#ifndef ENABLE_AIU
#define ENABLE_AIU 1
#endif
#else
#define ENABLE_AIU 0
#endif
"""
    declarations += f"#ifndef {disable_def}\n"
    declarations += f"""#include "extension/fmha/{impl_file}"\n"""

    # Declaration of kernel functions
    for k in kernels:
        implfile_to_kernels[k.impl_group].append(k)
        cat_to_kernels[(k.dtype, k.sm_range[0], k.sm_range[1])].append(k)

    for (cat_dt, cat_sm, cat_sm_max), kernels in cat_to_kernels.items():
        declarations += f"// ======== {cat_dt} / sm{cat_sm} ========\n"
        declarations += "\n".join(
            k.cpp_impl.split("{")[0].rstrip() + ";" for k in kernels
        )
        # import pdb;pdb.set_trace()
        dispatch_category_fn = f"dispatch_{family_name}_{cat_dt}_sm{cat_sm}"
        declarations += (
            f"\n\ntemplate <typename T> void {dispatch_category_fn}(T cb, int cc) {{\n"
        )
        for k in kernels:
            _call = f"cb({k.cpp_class}(), {k.name});\n"
            if k.dispatch_cond is not None:
                _call = f"if ({k.dispatch_cond}) {_call}"
            declarations += f"    {_call}"
        declarations += "}\n\n"
        dispatch_all += f"""
    if (std::is_same<DT, {DTYPES[cat_dt]}>::value && {cat_sm} <= cc && cc < {cat_sm_max}) {{
        {dispatch_category_fn}(cb, cc);
    }}"""

    declarations += f"""
template <typename DT, typename T>
void dispatch_{family_name}(T cb, int cc = 0) {{
{dispatch_all}
}}
"""
    declarations += f"#endif // {disable_def}\n"

    autogen_dir = Path(__file__).parent / "autogen_ppu"
    # import pdb;pdb.set_trace()

    (autogen_dir / f"{family_name}.h").write_text(declarations)

    for f, f_kernels in implfile_to_kernels.items():
        # import pdb;pdb.set_trace()
        impl_cu = cpp_file_header
        impl_cu += """
#if defined(__HGGCCC__)
#ifndef ENABLE_AIU
#define ENABLE_AIU 1
#endif
#else
#define ENABLE_AIU 0
#endif
"""
        impl_cu += f"#ifndef {disable_def}\n"
        impl_cu += """#include "accutlass.h"\n"""
        impl_cu += f"""#include "extension/fmha/{impl_file}"\n"""
        for k in f_kernels:
            # import pdb;pdb.set_trace()
            impl_cu += k.cpp_impl
        impl_cu += f"#endif // {disable_def}\n"
        # import pdb;pdb.set_trace()
        (autogen_dir / "impl" / f"{family_name}_{f}.cu").write_text(impl_cu)


write_decl_impl_with_fa1(
    [FwdKernelFA2.get_all(), FwdKernel.get_all()],
    "cutlassF",
    impl_file="kernel_forward.h",
    impl_file_2="kernel_forward_fa2.h",
    disable_def="XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD",
)
write_decl_impl_with_fa1(
    [BwdKernelFA2.get_all(), BwdKernel.get_all()],
    "cutlassB",
    impl_file="kernel_backward.h",
    impl_file_2="kernel_backward_fa2.h",
    disable_def="XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD",
)
