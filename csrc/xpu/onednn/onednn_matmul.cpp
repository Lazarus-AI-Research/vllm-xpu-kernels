#include <vector>
#include "fp4_gemm_w4a4.h"
#include "fp8_gemm_w8a8.h"
#include "fp8_gemm_w8a16.h"
#include "int4_gemm_w4a16.h"
#include "int4_gemm_w4a8.h"
#include "xpu/ops.h"

inline bool is_supported_fp8(at::ScalarType t) {
  return (t == at::ScalarType::Float8_e5m2) ||
         (t == at::ScalarType::Float8_e4m3fn);
}

inline bool is_supported_fp4(at::ScalarType t) {
  return t == at::ScalarType::Float4_e2m1fn_x2;
}

torch::Tensor check_and_create_output_tensor(
    const torch::Tensor& A,
    const torch::Tensor& B,
    std::optional<c10::ScalarType> out_dtype) {
  TORCH_CHECK(
      A.dim() == 2 || A.dim() == 3,
      "OneDNN Matmul only support 2D and 3D inputs!\n");
  TORCH_CHECK(
      B.dim() == 2 || B.dim() == 3,
      "OneDNN Matmul only support 2D and 3D weights!\n");
  if (B.dim() == 3) {
    TORCH_CHECK(
        A.dim() == 3,
        "OneDNN Matmul expects 3D input when using batched weights!\n");
    TORCH_CHECK(
        A.size(0) == B.size(0),
        "OneDNN Matmul expects input and weight batches to match, got ",
        A.size(0),
        " and ",
        B.size(0),
        ".");
  }

  if (B.scalar_type() == at::ScalarType::Int) {
    TORCH_CHECK(
        B.strides()[B.dim() - 2] == 1, "Int4 weight must be in NT format!\n");
  }

  std::vector<int64_t> result_shape;

  if (A.dim() == 2) {
    result_shape = {A.size(0), B.size(-1)};
    // src{m, k}, wei{k, n}, bias{n}, dst{m, n}
  } else {
    result_shape = {A.size(0), A.size(1), B.size(-1)};
    // src{b, m, k}, wei{k, n}, bias{n}, dst{b, m, n}
  }

  // deal with input shape [m, b, k] stride [k, m * k, 1]
  auto k = A.size(A.dim() - 1);
  auto n = result_shape.back();
  auto res_stride = A.strides().vec();
  for (int i = 0; i < res_stride.size() - 1; i++) {
    res_stride[i] = res_stride[i] / k * n;
  }

  // If out_dtype is not given, use fp16 as default
  const auto out_dtype_ = out_dtype.value_or(torch::kHalf);
  auto options = A.options().dtype(out_dtype_);
  return at::empty_strided(result_shape, res_stride, options);
}

torch::Tensor fp8_gemm(
    const torch::Tensor& A,  // [b, m ,k]
    const torch::Tensor& B,  // [k, n]
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_) {
  const at::DeviceGuard device_guard(A.device());
  torch::Tensor result = check_and_create_output_tensor(A, B, out_dtype);
  auto a_st = A.scalar_type();
  auto b_st = B.scalar_type();
  TORCH_CHECK(
      is_supported_fp8(a_st) && is_supported_fp8(b_st) && a_st == b_st,
      "input and weight must be f8_e5m2 or f8_e4m3fn for fp8 matmul");
  TORCH_CHECK(
      result.scalar_type() == torch::kFloat16 ||
          result.scalar_type() == torch::kBFloat16,
      "output must be float16 or bfloat16 for fp8 matmul");
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;

  torch::Tensor A_scale = A_scale_.value_or(at::ones({1}, torch::kFloat));
  torch::Tensor B_scale = B_scale_.value_or(at::ones({1}, torch::kFloat));
  oneDNN::dnnl_matmul_w8a8_fp8(result, A, B, is_nt, bias_, A_scale, B_scale);
  return result;
}

torch::Tensor fp8_bmm(
    const torch::Tensor& A,  // [b, m ,k]
    const torch::Tensor& B,  // [b, k, n]
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& A_scale_,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_) {
  const at::DeviceGuard device_guard(A.device());
  TORCH_CHECK(A.dim() == 3, "fp8_bmm expects A to be a 3D tensor");
  TORCH_CHECK(B.dim() == 3, "fp8_bmm expects B to be a 3D tensor");
  torch::Tensor result = check_and_create_output_tensor(A, B, out_dtype);
  auto a_st = A.scalar_type();
  auto b_st = B.scalar_type();
  TORCH_CHECK(
      is_supported_fp8(a_st) && is_supported_fp8(b_st) && a_st == b_st,
      "input and weight must be f8_e5m2 or f8_e4m3fn for fp8 matmul");
  TORCH_CHECK(
      result.scalar_type() == torch::kFloat16 ||
          result.scalar_type() == torch::kBFloat16,
      "output must be float16 or bfloat16 for fp8 matmul");
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;

  torch::Tensor A_scale = A_scale_.value_or(at::ones({1}, torch::kFloat));
  torch::Tensor B_scale = B_scale_.value_or(at::ones({1}, torch::kFloat));
  oneDNN::dnnl_batch_matmul_w8a8_fp8(
      result, A, B, is_nt, bias_, A_scale, B_scale);
  return result;
}

torch::Tensor fp8_gemm_w8a16(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const std::optional<torch::Tensor>& B_scale_,
    const std::optional<torch::Tensor>& bias_) {
  const at::DeviceGuard device_guard(A.device());
  TORCH_CHECK(
      is_supported_fp8(B.scalar_type()),
      "weight must be f8_e5m2 or f8_e4m3fn for fp8 matmul");
  TORCH_CHECK(
      A.dim() == 2 || A.dim() == 3,
      "FP8 W8A16 matmul only supports 2D and 3D inputs");

  const int64_t k = A.size(-1);
  TORCH_CHECK(k > 0, "FP8 W8A16 input K dimension must be non-zero");
  const int64_t m = A.numel() / k;
  const int64_t n = B.dim() == 2 ? B.size(1) : -1;
  const bool has_per_output_decode_scale =
      B_scale_.has_value() && B_scale_->numel() == n &&
      (B_scale_->dim() == 1 ||
       (B_scale_->dim() == 2 &&
        (B_scale_->size(0) == 1 || B_scale_->size(1) == 1)));
  const bool has_supported_decode_scale =
      !B_scale_.has_value() ||
      (B_scale_->is_contiguous() &&
       B_scale_->scalar_type() == torch::kFloat32 &&
       (B_scale_->numel() == 1 || has_per_output_decode_scale));
  // The public op's weight is logically [K, N]. The decode kernel can consume
  // it without a copy only when its transposed [N, K] view is contiguous.
  // Keeping this layout test explicit also avoids guessing for square weights.
  const bool has_decode_weight_layout =
      B.dim() == 2 && B.size(0) == k && B.transpose(0, 1).is_contiguous();
  const bool is_capturing =
      c10::xpu::getCurrentXPUStream(A.device().index()).is_capturing();
  // Direct B60 measurements select native only at M=1; oneDNN wins at M=2/4.
  // During graph capture, use the graph-safe native route through M=4 rather
  // than embedding the vendor primitive and its scratch lifecycle.
  const int64_t decode_max_m = is_capturing ? 4 : 1;
  const bool use_decode_gemv =
      has_decode_weight_layout && has_supported_decode_scale &&
      A.is_contiguous() && k % 16 == 0 && m <= decode_max_m;
  if (use_decode_gemv) {
    auto scale = B_scale_.has_value()
        ? *B_scale_
        : at::ones({1}, B.options().dtype(torch::kFloat32));
    auto result = fp8_gemv_w8a16(A, B.transpose(0, 1), scale);
    if (bias_.has_value() && bias_->numel() > 0) {
      result.add_(*bias_);
    }
    return result;
  }

  torch::Tensor result =
      check_and_create_output_tensor(A, B, A.scalar_type());
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;

  torch::Tensor B_scale = B_scale_.has_value()
                              ? B_scale_.value()
                              : at::ones(
                                    {1},
                                    B.options().dtype(A.dtype()));
  oneDNN::dnnl_matmul_w8a16_fp8(
      result, A, B, is_nt, bias_, B_scale);
  return result;
}

torch::Tensor fp4_gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& A_scale,
    const torch::Tensor& B_scale,
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A.device());
  torch::Tensor result = check_and_create_output_tensor(A, B, out_dtype);
  auto a_st = A.scalar_type();
  auto b_st = B.scalar_type();
  TORCH_CHECK(
      is_supported_fp4(a_st) && is_supported_fp4(b_st) && a_st == b_st,
      "input and weight must be f4_e2m1x2 or f4_e2m1x2 for fp4 matmul");
  TORCH_CHECK(
      result.scalar_type() == torch::kFloat16 ||
          result.scalar_type() == torch::kBFloat16,
      "output must be float16 or bfloat16 for fp4 matmul");
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;
  oneDNN::dnnl_matmul_w4a4_fp4(result, A, B, is_nt, bias, A_scale, B_scale);
  return result;
}

torch::Tensor int4_gemm_w4a16(
    const torch::Tensor& A_,  // src, [b, m, k]
    const torch::Tensor& B,   // quantized weight, [k, n]
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& B_scale,  // [k/group_size, n]
    const torch::Tensor& B_zp,     // [k/group_size, n/8]
    int64_t group_size,
    const std::optional<torch::Tensor>& g_idx) {
  const at::DeviceGuard device_guard(A_.device());

  // For GPTQ with desc_act=True scenario
  auto A = g_idx.has_value() ? A_.index_select(-1, g_idx.value()) : A_;
  torch::Tensor result = check_and_create_output_tensor(A, B, A.scalar_type());

  oneDNN::dnnl_matmul_w4a16_int4(result, A, B, bias, B_scale, B_zp, group_size);
  return result;
}

torch::Tensor int4_gemm_w4a8(
    const torch::Tensor& A_,       // quantized inputs, [b, m, k]
    const torch::Tensor& A_scale,  // [b * m, 1] or [b]
    const torch::Tensor& A_zp,     // [b * m, 1] or [b]
    const torch::Tensor& B,        // quantized weight, [k, n]
    const torch::Tensor& B_scale,  // [k/group_size, n]
    const torch::Tensor& B_zp,     // [k/group_size, n/8]
    int64_t group_size,
    const std::optional<torch::Tensor>& g_idx,
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A_.device());

  // Select indices if provided (for GPTQ with desc_act=True)
  const torch::Tensor& A =
      g_idx.has_value() ? A_.index_select(-1, g_idx.value()) : A_;

  // Validate quantization format for input A
  const bool per_token_A =
      (A_scale.dim() == 2 && A_scale.size(1) == 1 && A_zp.dim() == 2 &&
       A_zp.size(1) == 1);
  const bool per_tensor_A = (A_scale.dim() == 1 && A_zp.dim() == 1);
  TORCH_CHECK(
      per_token_A || per_tensor_A,
      "Int4-Int8 matmul expects quantized input A to be per-token ([b*m,1]) or "
      "per-tensor ([b]) quantized!");

  torch::Tensor result = check_and_create_output_tensor(A, B, torch::kHalf);

  oneDNN::dnnl_matmul_w4a8_int4(
      result, A, A_scale, A_zp, B, B_scale, B_zp, group_size, bias);

  return result;
}
