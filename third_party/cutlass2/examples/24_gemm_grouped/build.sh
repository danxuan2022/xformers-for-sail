#!/bin/bash

../common/build_and_run.sh gemm_grouped ppu clang 2>&1|tee build.log
../common/build_and_run.sh aiugemm_grouped ppu clang 2>&1|tee build.log
