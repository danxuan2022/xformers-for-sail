#!/bin/bash
bash ../common/build_and_run.sh ampere_fprop_mainloop_fusion_fp32_ppu ppu clang
bash ../common/build_and_run.sh ampere_fprop_mainloop_fusion_ppu ppu clang
