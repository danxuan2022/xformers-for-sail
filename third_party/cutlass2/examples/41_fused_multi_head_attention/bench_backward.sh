#!/bin/bash

head_dim=(16 32 40 64 80 96 128 160 192 224 256)

rm -rf run_backward.log

for hd in "${head_dim[@]}"
do
    sh run_bwd.sh $hd 2>&1 | tee -a run_backward.log
    sh run_bwd.sh $hd bf16 1 2>&1 | tee -a run_backward.log
done

