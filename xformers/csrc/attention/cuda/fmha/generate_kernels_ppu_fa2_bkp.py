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
    # "f32": "float",
    "f16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [80, 100]  # Sm80 kernels support up to Sm100

KERNEL_IMPL_TEMPLATE = """__global__ void __launch_bounds__(
    {CPP_CLASS}::kNumThreads,
    {CPP_CLASS}::kMinBlocksPerSm)
{NAME}(typename {CPP_CLASS}::Params p) {{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= {SM}0
#if __CUDA_ARCH__ < {SM_MAX}0
  if (!p.advance_to_block()) {{
    return;
  }}
  {CPP_CLASS}::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: kernel `{NAME}` is for sm{SM}-sm{SM_MAX}, but was built for sm%d\\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}}
"""


@dataclass(order=True)
class FwdKernel:
    sort_index: tuple[int, ...] = field(init=False, repr=False)
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
        return f"fmha_cutlassF_{self.dtype}_{self._aligned_suffix}_k{self.head_dim}x{self.q_block}x{self.k_block}_sm{self.sm_range[0]}"

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
            SM_MAX=self.sm_range[1],
        )

    @classmethod
    def get_all(cls) -> list["FwdKernel"]:
        kernels: list[FwdKernel] = []
        for (
            aligned,
            dtype,
            (sm, sm_max),
        ) in itertools.product([True, False], DTYPES.keys(), zip(SM, SM[1:])):
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
                        aligned=aligned,
                        dtype=dtype,
                        sm_range=(sm, sm_max),
                        head_dim=head_dim,
                        q_block=q_block,
                        k_block=k_block,
                        q_warp=q_warp,
                        k_warp=k_warp,
                        supports_dropout=False,
                        supports_bias=False,
                    )
                )
        return kernels


@dataclass(order=True)
class BwdKernel:
    sort_index: tuple[int, ...] = field(init=False, repr=False)
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
            SM_MAX=self.sm_range[1],
        )

    @classmethod
    def get_all(cls) -> list["BwdKernel"]:
        kernels: list[BwdKernel] = []
        for (
            aligned,
            dtype,
            (sm, sm_max),
            apply_dropout,
            keys_queries_aligned_to_blocksizes,
            kenable_split_keys,
            kenable_v2,
        ) in itertools.product(
            [True, False],
            DTYPES.keys(),
            zip(SM, SM[1:]),
            [False],  # off drop out for test
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
                (64, 64, 160, 8, 4, 4, 4),
                (64, 64, 192, 8, 4, 2, 2),
                (64, 64, 224, 8, 4, 4, 4),
                (64, 64, 256, 8, 4, 2, 2),
            ]:
                preload_mmas = True
                if dtype == "bf16" and sm < 80:
                    continue
                if not aligned and sm >= 80:
                    continue
                is_half = dtype in ["bf16", "f16"]

                # bi_values = [64]
                # Some architectures have more shmem and can use 128
                # We still need fallback to 64 for GPUs with less shmem
                # (Sm75, Sm86 ...)
                # if sm >= 80 or (sm >= 70 and is_half):
                #     if max_k > 64:
                #         bi_values.append(128)
                # for bi in bi_values:
                #     output_in_rf = is_half and max_k <= bi
                #     # preload_mmas = is_half and sm >= 80 and output_in_rf
                #     preload_mmas=True
                # bj = 128 if (preload_mmas and max_k > 64) else 64
                # import pdb;pdb.set_trace()
                kernels.append(
                    cls(
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
                # # A few specialized kernels that are faster
                # if apply_dropout or max_k > 128 or not is_half or not aligned:
                #     continue
                # if sm not in [70, 80]:
                #     continue
                # kernels.append(
                #     cls(
                #         aligned=aligned,
                #         dtype=dtype,
                #         sm_range=(sm, sm_max),
                #         apply_dropout=apply_dropout,
                #         preload_mmas=preload_mmas,
                #         block_i=bi,
                #         block_j=bj,
                #         max_k=max_k,
                #         keys_queries_aligned_to_blocksizes=True,
                #     )
                # )
            # Add some specialized kernels for stable diffusion BW (K=80)
            # This is the only kernel that can keep the outputs on RF on
            # Sm86/Sm89, so it's much faster than the 64x64 one
            # for dtype in ["f16", "bf16"]:
            #     kernels.append(
            #         cls(
            #             aligned=True,
            #             dtype=dtype,
            #             sm_range=(80, SM[SM.index(80) + 1]),
            #             apply_dropout=False,
            #             # preload_mmas=True,
            #             block_i=128,
            #             block_j=64,
            #             max_k=96,
            #             # Sm80 has a faster kernel for this case
            #             dispatch_cond="cc == 86 || cc == 89",
            #         )
            #     )
        return kernels


T = TypeVar("T", FwdKernel, BwdKernel)


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


write_decl_impl(
    FwdKernel.get_all(),
    "cutlassF",
    impl_file="kernel_forward_fa2.h",
    disable_def="XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD",
)
write_decl_impl(
    BwdKernel.get_all(),
    "cutlassB",
    impl_file="kernel_backward_fa2.h",
    disable_def="XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD",
)
