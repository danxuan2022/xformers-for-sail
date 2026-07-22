
# please replace hggc to cuda for include
# cd ${cutlass_home_dir} && bash tools/replace_cudart.sh

../common/build_and_run.sh fused_multihead_attention_fixed_seqlen ppu

../common/build_and_run.sh fused_multihead_attention_fixed_seqlen_fa2 ppu

../common/build_and_run.sh fused_multi_head_attention_backward ppu
../common/build_and_run.sh fused_multi_head_attention_backward_fa2 ppu
