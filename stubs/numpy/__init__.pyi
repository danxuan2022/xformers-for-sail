# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    Any,
    Container,
    Generic,
    Iterable,
    Optional,
    overload,
    Sized,
    SupportsAbs,
    SupportsBytes,
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    TypeVar,
)
from typing_extensions import Literal

from pyre_extensions import TypeVarTuple, Unpack

DType = TypeVar("DType")
NewDType = TypeVar("NewDType")
Ts = TypeVarTuple("Ts")
Ts2 = TypeVarTuple("Ts2")

N = TypeVar("N", bound=int)
A1 = TypeVar("A1")
A2 = TypeVar("A2")

class _ArrayOrScalarCommon(
    Generic[DType, Unpack[Ts]],
    SupportsInt,
    SupportsFloat,
    SupportsComplex,
    SupportsBytes,
    SupportsAbs[Any],
): ...
class float: ...

class ndarray(_ArrayOrScalarCommon[DType, Unpack[Ts]], Iterable, Sized, Container):
    def __init__(
        self,
        shape: tuple[Unpack[Ts]],
        dtype: type[DType] = ...,
        buffer=...,
        offset: Optional[int] = ...,
        strides: tuple[int, ...] = ...,
        order: Optional[str] = ...,
    ) -> None: ...
    @overload
    def __getitem__(
        self: ndarray[DType, A1, A2], key: Literal[0]
    ) -> ndarray[DType, A2]: ...
    @overload
    def __getitem__(
        self: ndarray[DType, A1, A2], key: Literal[1]
    ) -> ndarray[DType, A1]: ...
    def __setitem__(self, key, value): ...
    @property
    def shape(self) -> tuple[Unpack[Ts]]: ...
    @overload
    def reshape(self, shape: tuple[Unpack[Ts2]]) -> ndarray[DType, Unpack[Ts2]]: ...
    @overload
    def reshape(self, *shape: Unpack[Ts2]) -> ndarray[DType, Unpack[Ts2]]: ...
    def __add__(self, other) -> ndarray[DType, Unpack[Ts]]: ...
    def __div__(self, other) -> ndarray[DType, Unpack[Ts]]: ...
    def __truediv__(self, other) -> ndarray[DType, Unpack[Ts]]: ...
    # ===== BEGIN `astype` =====
    @overload
    def astype(self, dtype: type[NewDType]) -> ndarray[NewDType, Unpack[Ts]]: ...
    @overload
    def astype(self, dtype: Literal["int64"]) -> ndarray[int64, Unpack[Ts]]: ...
    @overload
    def astype(self, dtype: Literal["float32"]) -> ndarray[float32, Unpack[Ts]]: ...
    @overload
    def astype(self, dtype: Literal["float64"]) -> ndarray[float64, Unpack[Ts]]: ...
    # ===== END `astype` =====

# ===== BEGIN `empty` =====
# `shape` as tuple, dtype="int64"
@overload
def empty(
    shape: tuple[Unpack[Ts]], dtype: Literal["int64"]
) -> ndarray[int64, Unpack[Ts]]: ...

# `shape` as tuple, dtype as e.g. np.float32
@overload
def empty(
    shape: tuple[Unpack[Ts]], dtype: type[DType]
) -> ndarray[DType, Unpack[Ts]]: ...

# `shape` as integer, dtype as e.g. np.float32
@overload
def empty(shape: N, dtype: type[DType]) -> ndarray[DType, N]: ...

# ===== END `empty` =====
def array(
    object: object,
    dtype: type[DType] = ...,
    copy: bool = ...,
    subok: bool = ...,
    ndmin: int = ...,
) -> ndarray[DType, Unpack[tuple[Any, ...]]]: ...
def sin(x: ndarray[DType, Unpack[Ts]]) -> ndarray[DType, Unpack[Ts]]: ...

class int64:
    def __init__(self, value=...): ...

class float32:
    def __init__(self, value=...): ...

class float64:
    def __init__(self, value=...): ...

loadtxt: Any
asarray: Any
zeros: Any
