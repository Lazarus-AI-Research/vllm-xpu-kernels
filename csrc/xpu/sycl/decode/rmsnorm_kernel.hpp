// SPDX-License-Identifier: MIT
// Copyright (c) 2026 QuixiAI

// Native RMSNorm + fused-add-RMSNorm for Intel XPU.
//
// Decode on this stack is host-launch-bound; the vLLM IR ``rms_norm`` composite
// expands into ~10 aten ops (~64 us host/call) and fla's Triton path is ~47 us.
// A single native pybind submit is ~3.8 us/call, so folding the whole norm into
// one kernel removes ~60 us/call across the ~100 norms/token of decode.
//
// Semantics match vllm/ir/ops/layernorm.py exactly:
//   x32   = x.float()                       (fused-add: x32 = x.float()+res.float())
//   var   = mean(x32^2)  over H
//   nrm   = x32 * rsqrt(var + eps)
//   out   = (bf16)nrm * weight              (weight cast to activation dtype)
//   fused-add also writes residual_out = (T)x32.

#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "nvfp4_kernel.hpp"

namespace vllm::xpu::decode {
namespace rmsnorm_detail {

template <typename T>
inline float to_f32(T v) {
  return static_cast<float>(v);
}
template <typename T>
inline T from_f32(float v) {
  return static_cast<T>(v);
}

inline constexpr int kWG = 256;

template <typename T, bool FusedAdd>
class RmsNormKernel;

// One work-group per row. Each lane strides over the hidden dim, the group
// reduces sum(x^2), then every lane writes its normalized+weighted outputs.
template <typename T, bool FusedAdd>
sycl::event rms_norm_typed(sycl::queue& q, const T* x, T* residual,
                           const T* weight, T* out, float eps, std::size_t M,
                           std::size_t H) {
  const sycl::nd_range<1> ndr(sycl::range<1>(M * kWG), sycl::range<1>(kWG));
  const float inv_h = 1.0f / static_cast<float>(H);
  return q.parallel_for<RmsNormKernel<T, FusedAdd>>(
      ndr, [=](sycl::nd_item<1> it) {
        const std::size_t row = it.get_group(0);
        const std::size_t lid = it.get_local_id(0);
        const auto grp = it.get_group();
        const T* xr = x + row * H;
        T* outr = out + row * H;
        T* resr = FusedAdd ? residual + row * H : nullptr;

        // Pass 1: reduce sum(x^2) over the f32 sum (x + residual for fused-add).
        // The original residual is left intact so pass 2 can recompute the sum;
        // the IR composite normalizes the *unrounded* f32 sum, not the residual.
        float local = 0.0f;
        for (std::size_t i = lid; i < H; i += kWG) {
          float v = to_f32(xr[i]);
          if constexpr (FusedAdd) v += to_f32(resr[i]);
          local += v * v;
        }
        const float ss = sycl::reduce_over_group(grp, local, sycl::plus<float>());
        const float inv = sycl::rsqrt(ss * inv_h + eps);

        // Pass 2: write residual_out = (T)sum, out = (T)((T)(sum*inv) * weight).
        for (std::size_t i = lid; i < H; i += kWG) {
          float v = to_f32(xr[i]);
          if constexpr (FusedAdd) {
            v += to_f32(resr[i]);
            resr[i] = from_f32<T>(v);
          }
          const float nrm = to_f32(from_f32<T>(v * inv)) * to_f32(weight[i]);
          outr[i] = from_f32<T>(nrm);
        }
      });
}

}  // namespace rmsnorm_detail

inline sycl::event rms_norm_launch(sycl::queue& q, ActDType dt, const void* x,
                                   void* residual, const void* weight, void* out,
                                   float eps, std::size_t M, std::size_t H) {
  using namespace rmsnorm_detail;
  const bool fused = residual != nullptr;
#define DISPATCH(T)                                                            \
  do {                                                                         \
    if (fused)                                                                 \
      return rms_norm_typed<T, true>(                                          \
          q, static_cast<const T*>(x), static_cast<T*>(residual),             \
          static_cast<const T*>(weight), static_cast<T*>(out), eps, M, H);     \
    return rms_norm_typed<T, false>(                                           \
        q, static_cast<const T*>(x), nullptr,                                  \
        static_cast<const T*>(weight), static_cast<T*>(out), eps, M, H);       \
  } while (0)
  switch (dt) {
    case ActDType::bf16: DISPATCH(bf16_t);
    case ActDType::f16: DISPATCH(half_t);
    case ActDType::f32: DISPATCH(float);
  }
#undef DISPATCH
  return {};
}

}  // namespace vllm::xpu::decode
