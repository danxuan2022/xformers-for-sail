#!/bin/bash

set -e

export TraceEventPath=$(pwd)
export HGGC_PROFILE_MODE=4

#################################### legacy basic test ####################################
# D = 64
./fused_multihead_attention_fixed_seqlen.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=50 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=62 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=64 --iterations=1 --causal=false

./fused_multihead_attention_fixed_seqlen.ppu --head_number=1 --batch_size=4 --head_size=64 --seq_length=64 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen.ppu --head_number=1 --batch_size=4 --head_size=64 --seq_length=128 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=135 --iterations=1 --causal=false

./fused_multihead_attention_fixed_seqlen.ppu --head_number=8 --batch_size=4 --head_size=64 --seq_length=128 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen.ppu --head_number=8 --batch_size=1 --head_size=64 --seq_length=64 --iterations=1 --causal=false

# D = 128
./fused_multihead_attention_fixed_seqlen.ppu --head_number=1 --batch_size=1 --head_size=128 --seq_length=128 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen.ppu --head_number=8 --batch_size=4 --head_size=128 --seq_length=512 --iterations=1 --causal=false

# D = 256
./fused_multihead_attention_fixed_seqlen.ppu --head_number=1 --batch_size=1 --head_size=256 --seq_length=256 --iterations=1 --causal=false

# causal = true
./fused_multihead_attention_fixed_seqlen.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=64 --iterations=1 --causal=true

#################################### FA2 basic test ####################################
# D = 64
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=50 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=62 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=64 --iterations=1 --causal=false

./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=4 --head_size=64 --seq_length=64 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=4 --head_size=64 --seq_length=128 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=135 --iterations=1 --causal=false

./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=8 --batch_size=4 --head_size=64 --seq_length=128 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=8 --batch_size=1 --head_size=64 --seq_length=64 --iterations=1 --causal=false

# D = 128
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=128 --seq_length=128 --iterations=1 --causal=false
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=8 --batch_size=4 --head_size=128 --seq_length=512 --iterations=1 --causal=false

# D = 256
# ./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=256 --seq_length=256 --iterations=1 --causal=false

# causal = true
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=64 --iterations=1 --causal=true

#################################### FA2 D = 40 ####################################
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=40 --seq_length=64 --iterations=1 --causal=false

./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=40 --seq_length=4096 --iterations=1 --causal=false

# perf case
./fused_multihead_attention_fixed_seqlen.ppu --head_number=8 --batch_size=2 --head_size=40 --seq_length=9216 --seq_length_kv=77 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=8 --batch_size=2 --head_size=40 --seq_length=9216 --seq_length_kv=77 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
# before=850w, after=570w

#################################### FA2 D = 64 ####################################

./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=64 --iterations=1 --causal=false

./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=64 --seq_length=4096 --iterations=1 --causal=false

#./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=4 --head_size=64 --seq_length=4096 --iterations=1 --causal=false

# perf case
./fused_multihead_attention_fixed_seqlen.ppu --head_number=8 --batch_size=16 --head_size=64 --seq_length=4096 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=8 --batch_size=16 --head_size=64 --seq_length=4096 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
# before=850w, after=570w

#################################### FA2 D = 80 ####################################
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=80 --seq_length=128 --seq_length_kv=32 --iterations=1 --causal=false

./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=80 --seq_length=128 --seq_length_kv=4096 --iterations=1 --causal=false

./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=80 --seq_length=1024 --seq_length_kv=77 --iterations=1 --causal=false

# perf case
./fused_multihead_attention_fixed_seqlen.ppu --head_number=8 --batch_size=2 --head_size=80 --seq_length=1024 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=8 --batch_size=2 --head_size=80 --seq_length=1024 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv

./fused_multihead_attention_fixed_seqlen.ppu --head_number=8 --batch_size=2 --head_size=80 --seq_length=1024 --seq_length_kv=77 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=8 --batch_size=2 --head_size=80 --seq_length=1024 --seq_length_kv=77 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv

#################################### FA2 D = 96 ####################################
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=96 --seq_length=128 --seq_length_kv=32 --iterations=1 --causal=false

./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=96 --seq_length=128 --seq_length_kv=4096 --iterations=1 --causal=false

# perf case
./fused_multihead_attention_fixed_seqlen.ppu --head_number=8 --batch_size=4 --head_size=96 --seq_length=2048 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=8 --batch_size=4 --head_size=96 --seq_length=2048 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
# before=96w, after=60w

#################################### FA2 D = 128 ####################################
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=128 --seq_length=128 --iterations=1 --causal=false

./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=128 --seq_length=128 --seq_length_kv=4096 --iterations=1 --causal=false

# perf case
./fused_multihead_attention_fixed_seqlen.ppu --head_number=12 --batch_size=1 --head_size=128 --seq_length=2048 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=12 --batch_size=1 --head_size=128 --seq_length=2048 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
# before=43w, after=27w

#################################### FA2 D = 160 ####################################
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=160 --seq_length=128 --seq_length_kv=32 --iterations=1 --causal=false

./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=1 --batch_size=1 --head_size=160 --seq_length=128 --seq_length_kv=4096 --iterations=1 --causal=false

# perf case
./fused_multihead_attention_fixed_seqlen.ppu --head_number=16 --batch_size=1 --head_size=160 --seq_length=2048 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
./fused_multihead_attention_fixed_seqlen_fa2.ppu --head_number=16 --batch_size=1 --head_size=160 --seq_length=2048 --iterations=5 --causal=false --reference-check=false
cat trace_event_compute.csv
# before=105w, after=44w
