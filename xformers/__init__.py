# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch

from . import _cpp_lib
from .checkpoint import (  # noqa: E402, F401
    checkpoint,
    get_optimal_checkpoint_policy,
    list_operators,
    selective_checkpoint_wrapper,
)


try:
    from .version import __version__  # noqa: F401
except ImportError:
    __version__ = "0.0.0"


logger = logging.getLogger("xformers")

_has_cpp_library: bool = _cpp_lib._cpp_library_load_exception is None

_is_opensource: bool = True


def compute_once(func):
    value = None

    def func_wrapper():
        nonlocal value
        if value is None:
            value = func()
        return value

    return func_wrapper


@compute_once
def _is_triton_available():
    # Triton is disabled for this build
    return False


@compute_once
def get_python_lib():
    return torch.library.Library("xformers_python", "DEF")


# end of file
