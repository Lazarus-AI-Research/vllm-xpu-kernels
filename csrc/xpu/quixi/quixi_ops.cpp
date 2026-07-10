// SPDX-License-Identifier: Apache-2.0

#include "xpu/quixi/fp8_kernel.hpp"
#include "xpu/quixi/gdn_decode_kernel.hpp"
#include "xpu/quixi/nvfp4_kernel.hpp"
#include "xpu/quixi/nvfp4_moe_kernel.hpp"
#include "xpu/quixi/rmsnorm_kernel.hpp"

#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>

namespace {

quixi_nvfp4::ActDType activation_dtype(const torch::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case torch::kFloat32:
      return quixi_nvfp4::ActDType::f32;
    case torch::kFloat16:
      return quixi_nvfp4::ActDType::f16;
    case torch::kBFloat16:
      return quixi_nvfp4::ActDType::bf16;
    default:
      TORCH_CHECK(
          false,
          "Quixi kernels support float32, float16, and bfloat16 activations; "
          "got ",
          tensor.scalar_type());
  }
  return quixi_nvfp4::ActDType::f32;
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

torch::Tensor quixi_nvfp4_gemm(
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
  const int64_t element_size = x.element_size();
  auto* x_ptr = static_cast<char*>(x.data_ptr());
  auto* output_ptr = static_cast<char*>(output.data_ptr());
  const float scale = static_cast<float>(global_scale);
  auto& queue = current_queue(x);
  for (int64_t row = 0; row < m; ++row) {
    quixi_nvfp4::nvfp4_gemv_launch(
        queue,
        weight.data_ptr(),
        block_scales.data_ptr(),
        scale,
        x_ptr + row * k * element_size,
        output_ptr + row * n * element_size,
        static_cast<std::size_t>(n),
        static_cast<std::size_t>(k),
        dtype);
  }
  return output;
}

torch::Tensor quixi_fp8_gemm_w8a16(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& scale,
    int64_t kind,
    bool per_channel) {
  check_xpu_contiguous(x, "x");
  check_xpu_contiguous(weight, "weight");
  check_xpu_contiguous(scale, "scale");
  check_same_device(x, weight, "weight");
  check_same_device(x, scale, "scale");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "x must have a non-empty K dimension");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [N, K]");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8, "weight must be a uint8 view");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "scale must be float32");
  TORCH_CHECK(kind == 0 || kind == 1, "kind must be 0 (e4m3) or 1 (e5m2)");

  const int64_t k = x.size(-1);
  const int64_t m = x.numel() / k;
  const int64_t n = weight.size(0);
  TORCH_CHECK(k % 16 == 0, "K must be a multiple of 16");
  TORCH_CHECK(weight.size(1) == k, "weight K dimension mismatch");
  TORCH_CHECK(
      scale.numel() == (per_channel ? n : 1),
      per_channel ? "per-channel scale must contain N values"
                  : "per-tensor scale must contain one value");

  auto output_sizes = x.sizes().vec();
  output_sizes.back() = n;
  auto output = torch::empty(output_sizes, x.options());
  quixi_nvfp4::fp8_gemv_launch(
      current_queue(x),
      activation_dtype(x),
      static_cast<int>(kind),
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
  return {m, topk, k, intermediate};
}

}  // namespace

torch::Tensor quixi_nvfp4_moe(
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
  auto output = torch::zeros(
      {shape.m, shape.hidden_size}, hidden.options().dtype(torch::kFloat32));
  quixi_nvfp4::nvfp4_moe_launch(
      current_queue(hidden),
      activation_dtype(hidden),
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
      static_cast<std::size_t>(shape.hidden_size),
      static_cast<std::size_t>(shape.intermediate_size),
      multiply_router_weight);
  return output;
}

torch::Tensor quixi_nvfp4_moe_split(
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
  auto output = torch::zeros(
      {shape.m, shape.hidden_size}, hidden.options().dtype(torch::kFloat32));
  auto intermediate = torch::empty(
      {shape.m * shape.topk, 2 * shape.intermediate_size},
      hidden.options().dtype(torch::kFloat32));
  quixi_nvfp4::nvfp4_moe_split_launch(
      current_queue(hidden),
      activation_dtype(hidden),
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
      static_cast<std::size_t>(shape.hidden_size),
      static_cast<std::size_t>(shape.intermediate_size),
      multiply_router_weight);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> quixi_qwen_gdn_decode(
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

  const int64_t batch = projected_qkvz.size(0);
  TORCH_CHECK(
      projected_qkvz.dim() == 2 && projected_qkvz.size(1) == 12288,
      "projected_qkvz must have shape [B, 12288]");
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
          conv_weight.scalar_type() == projected_qkvz.scalar_type() &&
          conv_bias.scalar_type() == projected_qkvz.scalar_type(),
      "projected_ba, conv_weight, and conv_bias must match projected_qkvz dtype");
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

  auto mixed_qkv = torch::empty({batch, 8192}, projected_qkvz.options());
  auto z = torch::empty({batch, 32, 128}, projected_qkvz.options());
  auto output = torch::empty({batch, 32, 128}, projected_qkvz.options());
  auto& queue = current_queue(projected_qkvz);
  quixi_nvfp4::gdn_conv_launch(
      queue,
      activation_dtype(projected_qkvz),
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
  quixi_nvfp4::gdn_recur_launch(
      queue,
      activation_dtype(projected_qkvz),
      activation_dtype(ssm_state),
      activation_dtype(dt_bias),
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

void quixi_rms_norm(
    torch::Tensor& output,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double epsilon,
    const std::optional<torch::Tensor>& residual) {
  check_xpu_contiguous(output, "output");
  check_xpu_contiguous(x, "x");
  check_xpu_contiguous(weight, "weight");
  check_same_device(x, output, "output");
  check_same_device(x, weight, "weight");
  TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "x must have a non-empty hidden dimension");
  TORCH_CHECK(output.sizes() == x.sizes(), "output must match x shape");
  TORCH_CHECK(output.scalar_type() == x.scalar_type(), "output must match x dtype");
  TORCH_CHECK(weight.numel() == x.size(-1), "weight must match the hidden dimension");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight must match x dtype");

  void* residual_ptr = nullptr;
  if (residual.has_value()) {
    check_xpu_contiguous(*residual, "residual");
    check_same_device(x, *residual, "residual");
    TORCH_CHECK(residual->sizes() == x.sizes(), "residual must match x shape");
    TORCH_CHECK(residual->scalar_type() == x.scalar_type(), "residual must match x dtype");
    residual_ptr = residual->data_ptr();
  }

  quixi_nvfp4::rms_norm_launch(
      current_queue(x),
      activation_dtype(x),
      x.data_ptr(),
      residual_ptr,
      weight.data_ptr(),
      output.data_ptr(),
      static_cast<float>(epsilon),
      static_cast<std::size_t>(x.numel() / x.size(-1)),
      static_cast<std::size_t>(x.size(-1)));
}
