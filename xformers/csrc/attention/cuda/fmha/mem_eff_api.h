#pragma once
#include <cmath>
#include <mutex>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <iostream> 
#include <algorithm>


/*
  There are 2 modes for using this function.
  (Mode BMHK) With all the heads having the same seqlen
  (Mode 1MHK) `batch=1` with all tokens across batches concatenated
*/

using namespace at;

std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt>
efficient_attention_forward_cutlass(
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    const std::optional<at::Tensor>& bias, // [b, num_heads, seqlen, seqlen]
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const std::optional<at::Tensor>& seqstart_q,
    // (Mode 1MHK only) [b+1]: cu_seqlen_k[b] contains the
    // position of the first key token for batch $b
    const std::optional<at::Tensor>& seqstart_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const std::optional<int64_t> max_seqlen_q_,
    const std::optional<int64_t> max_seqlen_k_,
    double dropout_p, // attention matrix dropout probability
    int64_t custom_mask_type,
    bool compute_logsumexp,
    std::optional<double> scale,
    const std::optional<at::Tensor>& seqlen_k,
    const std::optional<int64_t> window_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mem_efficient_attention_backward_cutlass(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& kernel_bias, // additive attention bias
    const at::Tensor& out,
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const std::optional<at::Tensor>& cu_seqlens_q_dummy,
    // (Mode 1MHK only) [b+1]: cu_seqlens_k[b] contains the
    // position of the first key token for batch $b
    const std::optional<at::Tensor>& cu_seqlens_k_dummy,
    // (Mode 1MHK only) Maximum sequence length across batches
    int64_t max_seqlen_q,
    // (Mode 1MHK only) Maximum sequence length across batches
    int64_t max_seqlen_k,
    const at::Tensor& logsumexp,
    double dropout_p, // dropout probability
    const at::Tensor&
        philox_seed, // seed using for generating random numbers for dropout
    const at::Tensor& philox_offset, // offset into random number sequence
    int64_t custom_mask_type,
    const bool bias_requires_grad,
    const std::optional<double> scale,
    std::optional<int64_t> num_splits_key,
    const std::optional<int64_t> window_size);