#!/bin/bash

set -e
hd=${1:-"128"}
type=${2:-"fp16"}
cause=${3:-"0"}
#python fmha_backward_test.py ./fused_multi_head_attention_backward.ppu 1 64 64 1 $hd $hd $type $cause
#python fmha_backward_test.py ./fused_multi_head_attention_backward.ppu 8 601 601 8 $hd $hd $type $cause


python fmha_backward_test.py ./fused_multi_head_attention_backward_fa2.ppu 1 128 128 1 $hd $hd $type $cause
#python fmha_backward_test.py ./fused_multi_head_attention_backward_fa2.ppu 12 512 512 32 $hd $hd $type $cause
#python fmha_backward_test.py ./fused_multi_head_attention_backward_fa2.ppu 16 2048 2048 5 $hd $hd $type $cause

