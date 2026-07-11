// SPDX-License-Identifier: Apache-2.0

#include "xpu/sycl/decode/fp8_kernel.hpp"
#include "xpu/sycl/decode/gdn_decode_kernel.hpp"
#include "xpu/sycl/decode/nvfp4_kernel.hpp"
#include "xpu/sycl/decode/nvfp4_moe_kernel.hpp"

#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace {

vllm::xpu::decode::ActDType activation_dtype(const torch::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case torch::kFloat32:
      return vllm::xpu::decode::ActDType::f32;
    case torch::kFloat16:
      return vllm::xpu::decode::ActDType::f16;
    case torch::kBFloat16:
      return vllm::xpu::decode::ActDType::bf16;
    default:
      TORCH_CHECK(
          false,
          "Decode kernels support float32, float16, and bfloat16 activations; "
          "got ",
          tensor.scalar_type());
  }
  return vllm::xpu::decode::ActDType::f32;
}

sycl::queue& current_queue(const torch::Tensor& tensor) {
  return c10::xpu::getCurrentXPUStream(tensor.device().index()).queue();
}

void check_xpu_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_xpu(), name, " must be on XPU");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(
    const torch::Tensor& reference,
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(
      tensor.device() == reference.device(),
      name,
      " must be on ",
      reference.device(),
      "; got ",
      tensor.device());
}

}  // namespace

torch::Tensor nvfp4_gemm(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& block_scales,
    double global_scale) {
  check_xpu_contiguous(x, "x");
  check_xpu_contiguous(weight, "weight");
  check_xpu_contiguous(block_scales, "block_scales");
  check_same_device(x, weight, "weight");
  check_same_device(x, block_scales, "block_scales");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "x must have a non-empty K dimension");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K / 2]");
  TORCH_CHECK(
      block_scales.dim() == 2,
      "block_scales must have shape [N, K / 16]");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "weight must be uint8");
  TORCH_CHECK(
      block_scales.scalar_type() == torch::kFloat8_e4m3fn,
      "block_scales must be float8_e4m3fn");

  const int64_t k = x.size(-1);
  const int64_t m = x.numel() / k;
  const int64_t n = weight.size(0);
  TORCH_CHECK(k % 32 == 0, "K must be a multiple of 32");
  TORCH_CHECK(weight.size(1) == k / 2, "weight K / 2 dimension mismatch");
  TORCH_CHECK(
      block_scales.size(0) == n && block_scales.size(1) == k / 16,
      "block_scales must have shape [N, K / 16]");

  auto output_sizes = x.sizes().vec();
  output_sizes.back() = n;
  auto output = torch::empty(output_sizes, x.options());
  const auto dtype = activation_dtype(x);
  if (m == 0 || n == 0) {
    return output;
  }

  const int64_t element_size = x.element_size();
  auto* x_ptr = static_cast<char*>(x.data_ptr());
  auto* output_ptr = static_cast<char*>(output.data_ptr());
  const float scale = static_cast<float>(global_scale);
  auto& queue = current_queue(x);
  constexpr int64_t kRowsPerLaunch = vllm::xpu::decode::detail::kMaxM;
  for (int64_t row = 0; row < m; row += kRowsPerLaunch) {
    const int64_t rows = std::min(kRowsPerLaunch, m - row);
    const auto* input = x_ptr + row * k * element_size;
    auto* result = output_ptr + row * n * element_size;
    if (rows == 1) {
      vllm::xpu::decode::nvfp4_gemv_launch(
          queue,
          weight.data_ptr(),
          block_scales.data_ptr(),
          scale,
          input,
          result,
          static_cast<std::size_t>(n),
          static_cast<std::size_t>(k),
          dtype);
    } else {
      vllm::xpu::decode::nvfp4_gemm_mtiled_launch(
          queue,
          weight.data_ptr(),
          block_scales.data_ptr(),
          scale,
          input,
          result,
          static_cast<std::size_t>(rows),
          static_cast<std::size_t>(n),
          static_cast<std::size_t>(k),
          dtype);
    }
  }
  return output;
}

torch::Tensor fp8_gemv_w8a16(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& scale) {
  check_xpu_contiguous(x, "x");
  check_xpu_contiguous(weight, "weight");
  check_xpu_contiguous(scale, "scale");
  check_same_device(x, weight, "weight");
  check_same_device(x, scale, "scale");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "x must have a non-empty K dimension");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
  TORCH_CHECK(
      weight.scalar_type() == torch::kFloat8_e4m3fn ||
          weight.scalar_type() == torch::kFloat8_e5m2,
      "weight must be float8_e4m3fn or float8_e5m2");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");

  const int64_t k = x.size(-1);
  const int64_t m = x.numel() / k;
  const int64_t n = weight.size(0);
  const bool per_channel = scale.numel() == n;
  const int kind = weight.scalar_type() == torch::kFloat8_e5m2 ? 1 : 0;
  TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16");
  TORCH_CHECK(weight.size(1) == k, "weight K dimension mismatch");
  TORCH_CHECK(
      scale.numel() == 1 || per_channel,
      "scale must contain one value or N per-channel values");

  auto output_sizes = x.sizes().vec();
  output_sizes.back() = n;
  auto output = torch::empty(output_sizes, x.options());
  const auto dtype = activation_dtype(x);
  if (m == 0 || n == 0) {
    return output;
  }
  vllm::xpu::decode::fp8_gemv_launch(
      current_queue(x),
      dtype,
      kind,
      x.data_ptr(),
      weight.data_ptr(),
      scale.data_ptr(),
      per_channel,
      output.data_ptr(),
      static_cast<std::size_t>(m),
      static_cast<std::size_t>(n),
      static_cast<std::size_t>(k));
  return output;
}

namespace {

struct Nvfp4MoeShape {
  int64_t m;
  int64_t topk;
  int64_t experts;
  int64_t hidden_size;
  int64_t intermediate_size;
};

Nvfp4MoeShape check_nvfp4_moe_inputs(
    const torch::Tensor& hidden,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w13_global_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& w2_global_scale) {
  check_xpu_contiguous(hidden, "hidden");
  check_xpu_contiguous(topk_ids, "topk_ids");
  check_xpu_contiguous(topk_weights, "topk_weights");
  check_xpu_contiguous(w13, "w13");
  check_xpu_contiguous(w13_scale, "w13_scale");
  check_xpu_contiguous(w13_global_scale, "w13_global_scale");
  check_xpu_contiguous(w2, "w2");
  check_xpu_contiguous(w2_scale, "w2_scale");
  check_xpu_contiguous(w2_global_scale, "w2_global_scale");
  for (const auto* tensor : {
           &topk_ids,
           &topk_weights,
           &w13,
           &w13_scale,
           &w13_global_scale,
           &w2,
           &w2_scale,
           &w2_global_scale}) {
    check_same_device(hidden, *tensor, "MoE input");
  }

  TORCH_CHECK(hidden.dim() == 2, "hidden must have shape [M, K]");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must have shape [M, topk]");
  TORCH_CHECK(
      topk_weights.sizes() == topk_ids.sizes(),
      "topk_weights must match topk_ids shape");
  TORCH_CHECK(w13.dim() == 3, "w13 must have shape [E, 2I, K / 2]");
  TORCH_CHECK(w2.dim() == 3, "w2 must have shape [E, K, I / 2]");
  TORCH_CHECK(w13.scalar_type() == torch::kUInt8, "w13 must be uint8");
  TORCH_CHECK(w2.scalar_type() == torch::kUInt8, "w2 must be uint8");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32, "topk_ids must be int32");
  TORCH_CHECK(
      topk_weights.scalar_type() == torch::kFloat32,
      "topk_weights must be float32");
  TORCH_CHECK(
      w13_scale.scalar_type() == torch::kFloat8_e4m3fn &&
          w2_scale.scalar_type() == torch::kFloat8_e4m3fn,
      "w13_scale and w2_scale must be float8_e4m3fn");
  TORCH_CHECK(
      w13_global_scale.scalar_type() == torch::kFloat32 &&
          w2_global_scale.scalar_type() == torch::kFloat32,
      "global scales must be float32");

  const int64_t m = hidden.size(0);
  const int64_t k = hidden.size(1);
  const int64_t topk = topk_ids.size(1);
  const int64_t experts = w13.size(0);
  const int64_t intermediate = w13.size(1) / 2;
  TORCH_CHECK(topk_ids.size(0) == m, "topk inputs must contain M rows");
  TORCH_CHECK(k > 0 && intermediate > 0, "K and I must be non-zero");
  TORCH_CHECK(
      experts > 0 || topk == 0,
      "at least one expert is required when topk is non-zero");
  TORCH_CHECK(k % 32 == 0 && intermediate % 32 == 0, "K and I must be multiples of 32");
  TORCH_CHECK(w13.size(1) % 2 == 0, "w13 output dimension must be even");
  TORCH_CHECK(w13.size(2) == k / 2, "w13 K / 2 dimension mismatch");
  TORCH_CHECK(
      w13_scale.sizes() == torch::IntArrayRef({experts, 2 * intermediate, k / 16}),
      "w13_scale must have shape [E, 2I, K / 16]");
  TORCH_CHECK(
      w2.sizes() == torch::IntArrayRef({experts, k, intermediate / 2}),
      "w2 must have shape [E, K, I / 2]");
  TORCH_CHECK(
      w2_scale.sizes() == torch::IntArrayRef({experts, k, intermediate / 16}),
      "w2_scale must have shape [E, K, I / 16]");
  TORCH_CHECK(
      w13_global_scale.numel() == experts &&
          w2_global_scale.numel() == experts,
      "global scales must contain one value per expert");
  return {m, topk, experts, k, intermediate};
}

}  // namespace

torch::Tensor nvfp4_moe(
    const torch::Tensor& hidden,
    const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights,
    const torch::Tensor& w13,
    const torch::Tensor& w13_scale,
    const torch::Tensor& w13_global_scale,
    const torch::Tensor& w2,
    const torch::Tensor& w2_scale,
    const torch::Tensor& w2_global_scale,
    bool multiply_router_weight) {
  const auto shape = check_nvfp4_moe_inputs(
      hidden,
      topk_ids,
      topk_weights,
      w13,
      w13_scale,
      w13_global_scale,
      w2,
      w2_scale,
      w2_global_scale);
  const auto dtype = activation_dtype(hidden);
  auto output = torch::zeros(
      {shape.m, shape.hidden_size}, hidden.options().dtype(torch::kFloat32));
  if (shape.m == 0 || shape.topk == 0) {
    return output;
  }

  auto& queue = current_queue(hidden);
  const auto local_mem_size =
      queue.get_device().get_info<sycl::info::device::local_mem_size>();
  const auto intermediate_size =
      static_cast<std::size_t>(shape.intermediate_size);
  const auto fused_local_bytes = 3 * intermediate_size * sizeof(float);
  const auto split_local_bytes = intermediate_size * sizeof(float);
  const bool fused_supported = fused_local_bytes <= local_mem_size;
  const bool split_supported = split_local_bytes <= local_mem_size;

  // Measured on Arc Pro B60: split wins at decode M<=4, while fused wins once
  // M*topk provides enough work-groups. Split is also the capacity fallback
  // when the fused kernel exceeds local memory.
  const bool select_split = shape.m <= 4 || !fused_supported;

  if (!select_split) {
    TORCH_CHECK(
        fused_supported,
        "NVFP4 fused MoE intermediate size requires ",
        fused_local_bytes,
        " bytes of local memory, but the device provides ",
        local_mem_size);
    vllm::xpu::decode::nvfp4_moe_launch(
        queue,
        dtype,
        hidden.data_ptr(),
        topk_ids.data_ptr(),
        topk_weights.data_ptr(),
        w13.data_ptr(),
        w13_scale.data_ptr(),
        w13_global_scale.data_ptr(),
        w2.data_ptr(),
        w2_scale.data_ptr(),
        w2_global_scale.data_ptr(),
        output.data_ptr(),
        static_cast<std::size_t>(shape.m),
        static_cast<std::size_t>(shape.topk),
        static_cast<std::size_t>(shape.experts),
        static_cast<std::size_t>(shape.hidden_size),
        intermediate_size,
        multiply_router_weight);
    return output;
  }

  TORCH_CHECK(
      split_supported,
      "NVFP4 MoE intermediate size requires ",
      split_local_bytes,
      " bytes of local memory, but the device provides ",
      local_mem_size);
  auto intermediate = torch::empty(
      {shape.m * shape.topk, 2 * shape.intermediate_size},
      hidden.options().dtype(torch::kFloat32));
  vllm::xpu::decode::nvfp4_moe_split_launch(
      queue,
      dtype,
      hidden.data_ptr(),
      topk_ids.data_ptr(),
      topk_weights.data_ptr(),
      w13.data_ptr(),
      w13_scale.data_ptr(),
      w13_global_scale.data_ptr(),
      w2.data_ptr(),
      w2_scale.data_ptr(),
      w2_global_scale.data_ptr(),
      intermediate.data_ptr(),
      output.data_ptr(),
      static_cast<std::size_t>(shape.m),
      static_cast<std::size_t>(shape.topk),
      static_cast<std::size_t>(shape.experts),
      static_cast<std::size_t>(shape.hidden_size),
      intermediate_size,
      multiply_router_weight);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> qwen_gdn_decode(
    const torch::Tensor& projected_qkvz,
    const torch::Tensor& projected_ba,
    torch::Tensor& conv_state,
    torch::Tensor& ssm_state,
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& a_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& state_indices) {
  check_xpu_contiguous(projected_qkvz, "projected_qkvz");
  check_xpu_contiguous(projected_ba, "projected_ba");
  check_xpu_contiguous(conv_state, "conv_state");
  check_xpu_contiguous(ssm_state, "ssm_state");
  check_xpu_contiguous(conv_weight, "conv_weight");
  check_xpu_contiguous(conv_bias, "conv_bias");
  check_xpu_contiguous(a_log, "a_log");
  check_xpu_contiguous(dt_bias, "dt_bias");
  check_xpu_contiguous(state_indices, "state_indices");
  check_same_device(projected_qkvz, projected_ba, "projected_ba");
  check_same_device(projected_qkvz, conv_state, "conv_state");
  check_same_device(projected_qkvz, ssm_state, "ssm_state");
  check_same_device(projected_qkvz, conv_weight, "conv_weight");
  check_same_device(projected_qkvz, conv_bias, "conv_bias");
  check_same_device(projected_qkvz, a_log, "a_log");
  check_same_device(projected_qkvz, dt_bias, "dt_bias");
  check_same_device(projected_qkvz, state_indices, "state_indices");

  TORCH_CHECK(
      projected_qkvz.dim() == 2 && projected_qkvz.size(1) == 12288,
      "projected_qkvz must have shape [B, 12288]");
  const int64_t batch = projected_qkvz.size(0);
  TORCH_CHECK(
      projected_ba.dim() == 2 && projected_ba.size(0) == batch &&
          projected_ba.size(1) == 64,
      "projected_ba must have shape [B, 64]");
  TORCH_CHECK(
      conv_weight.sizes() == torch::IntArrayRef({8192, 4}),
      "conv_weight must have shape [8192, 4]");
  TORCH_CHECK(conv_bias.numel() == 8192, "conv_bias must contain 8192 values");
  TORCH_CHECK(
      a_log.numel() == 32 && dt_bias.numel() == 32,
      "a_log and dt_bias must contain 32 values");
  TORCH_CHECK(
      state_indices.dim() == 1 && state_indices.size(0) == batch,
      "state_indices must have shape [B]");
  TORCH_CHECK(state_indices.scalar_type() == torch::kInt32, "state_indices must be int32");
  TORCH_CHECK(a_log.scalar_type() == torch::kFloat32, "a_log must be float32");
  TORCH_CHECK(
      projected_ba.scalar_type() == projected_qkvz.scalar_type() &&
          conv_state.scalar_type() == projected_qkvz.scalar_type() &&
          conv_weight.scalar_type() == projected_qkvz.scalar_type() &&
          conv_bias.scalar_type() == projected_qkvz.scalar_type(),
      "projected_ba, conv_state, conv_weight, and conv_bias must match "
      "projected_qkvz dtype");
  TORCH_CHECK(
      ssm_state.dim() == 4 && ssm_state.size(1) == 32 &&
          ssm_state.size(2) == 128 && ssm_state.size(3) == 128,
      "ssm_state must have shape [slots, 32, 128, 128]");

  TORCH_CHECK(conv_state.dim() == 3, "conv_state must be three-dimensional");
  const bool conv_dim_first =
      conv_state.size(1) == 8192 && conv_state.size(2) == 3;
  TORCH_CHECK(
      conv_dim_first ||
          (conv_state.size(1) == 3 && conv_state.size(2) == 8192),
      "conv_state must have shape [slots, 8192, 3] or [slots, 3, 8192]");
  TORCH_CHECK(
      conv_state.size(0) == ssm_state.size(0),
      "conv_state and ssm_state slot counts must match");

  const auto projected_dtype = activation_dtype(projected_qkvz);
  const auto state_dtype = activation_dtype(ssm_state);
  const auto dt_dtype = activation_dtype(dt_bias);

  auto mixed_qkv = torch::empty({batch, 8192}, projected_qkvz.options());
  auto z = torch::empty({batch, 32, 128}, projected_qkvz.options());
  auto output = torch::empty({batch, 32, 128}, projected_qkvz.options());
  if (batch == 0) {
    return {output, z};
  }
  TORCH_CHECK(conv_state.size(0) > 0, "state tensors must contain at least one slot");
  auto& queue = current_queue(projected_qkvz);
  vllm::xpu::decode::gdn_conv_launch(
      queue,
      projected_dtype,
      projected_qkvz.data_ptr(),
      conv_weight.data_ptr(),
      conv_bias.data_ptr(),
      state_indices.data_ptr(),
      conv_state.data_ptr(),
      conv_dim_first,
      mixed_qkv.data_ptr(),
      z.data_ptr(),
      static_cast<std::size_t>(batch),
      static_cast<std::size_t>(conv_state.size(0)));
  vllm::xpu::decode::gdn_recur_launch(
      queue,
      projected_dtype,
      state_dtype,
      dt_dtype,
      mixed_qkv.data_ptr(),
      projected_ba.data_ptr(),
      a_log.data_ptr(),
      dt_bias.data_ptr(),
      state_indices.data_ptr(),
      ssm_state.data_ptr(),
      output.data_ptr(),
      static_cast<std::size_t>(batch),
      static_cast<std::size_t>(ssm_state.size(0)));
  return {output, z};
}
