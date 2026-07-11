// SPDX-License-Identifier: MIT
// Copyright (c) 2026 QuixiAI

// Decode-only Qwen3.5/Qwen3.6 Gated DeltaNet core for XPU.
//
// Prototype scope: non-interleaved qkvz layout, no prefill, no speculative
// decode, TP=1. This matches the deployed nvidia/Qwen3.6-35B-A3B-NVFP4 path
// through qwen3_5.py. The op mutates conv_state and ssm_state and returns
// core_attn_out plus z for the caller's RMSNormGated/out_proj.

#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "nvfp4_kernel.hpp"

namespace vllm::xpu::decode {
namespace gdn_detail {

inline constexpr std::size_t kH = 16;
inline constexpr std::size_t kHV = 32;
inline constexpr std::size_t kK = 128;
inline constexpr std::size_t kV = 128;
inline constexpr std::size_t kQDim = kH * kK;   // 2048
inline constexpr std::size_t kKDim = kH * kK;   // 2048
inline constexpr std::size_t kVDim = kHV * kV;  // 4096
inline constexpr std::size_t kConvDim = kQDim + kKDim + kVDim;
inline constexpr std::size_t kQKVZDim = kConvDim + kVDim;
inline constexpr std::size_t kBADim = 2 * kHV;

template <typename T>
inline float to_f32(T v) {
  return static_cast<float>(v);
}

template <typename T>
inline T from_f32(float v) {
  return static_cast<T>(v);
}

inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }

inline float softplus(float x) {
  return x <= 20.0f ? sycl::log(1.0f + sycl::exp(x)) : x;
}

template <typename T>
class QwenGdnConvKernel;

template <typename T, typename StateT, typename DtT>
class QwenGdnRecurKernel;

// projected_qkvz: [B, 12288] = [q, k, v, z], non-interleaved.
// projected_ba:   [B, 64]    = [b, a].
// conv_state:     [slots, 3, 8192] (SD) or [slots, 8192, 3] (DS).
// conv_weight:    [8192, 4].
template <typename T>
sycl::event gdn_conv_typed(
    sycl::queue& q,
    const T* projected_qkvz,
    const T* conv_weight,
    const T* conv_bias,
    const std::int32_t* state_indices,
    T* conv_state,
    bool conv_dim_first,
    T* mixed_qkv,
    T* z_out,
    std::size_t B,
    std::size_t nslots) {
  const sycl::nd_range<1> ndr(
      sycl::range<1>(((B * kConvDim + 255) / 256) * 256), sycl::range<1>(256));
  return q.parallel_for<QwenGdnConvKernel<T>>(ndr, [=](sycl::nd_item<1> it) {
    const std::size_t idx = it.get_global_id(0);
    if (idx >= B * kConvDim) return;
    const std::size_t b = idx / kConvDim;
    const std::size_t c = idx - b * kConvDim;
    if (c >= kConvDim - kVDim) {
      const std::size_t zc = c - (kConvDim - kVDim);
      z_out[b * kVDim + zc] = projected_qkvz[b * kQKVZDim + kConvDim + zc];
    }
    const std::int32_t state_idx = state_indices[b];
    if (state_idx < 0 || static_cast<std::size_t>(state_idx) >= nslots) {
      mixed_qkv[idx] = T(0);
      return;
    }

    const T x = projected_qkvz[b * kQKVZDim + c];
    T* state = conv_state + static_cast<std::size_t>(state_idx) * 3 * kConvDim;
    const std::size_t s0 = conv_dim_first ? c * 3 : c;
    const std::size_t s_stride = conv_dim_first ? 1 : kConvDim;

    const float h0 = to_f32(state[s0 + 0 * s_stride]);
    const float h1 = to_f32(state[s0 + 1 * s_stride]);
    const float h2 = to_f32(state[s0 + 2 * s_stride]);
    const float xf = to_f32(x);

    state[s0 + 0 * s_stride] = from_f32<T>(h1);
    state[s0 + 1 * s_stride] = from_f32<T>(h2);
    state[s0 + 2 * s_stride] = x;

    const T* w = conv_weight + c * 4;
    float acc = conv_bias == nullptr ? 0.0f : to_f32(conv_bias[c]);
    acc += h0 * to_f32(w[0]);
    acc += h1 * to_f32(w[1]);
    acc += h2 * to_f32(w[2]);
    acc += xf * to_f32(w[3]);
    mixed_qkv[idx] = from_f32<T>(silu(acc));
  });
}

template <typename T, typename StateT, typename DtT>
sycl::event gdn_recur_typed(
    sycl::queue& q,
    const T* mixed_qkv,
    const T* projected_ba,
    const float* A_log,
    const DtT* dt_bias,
    const std::int32_t* state_indices,
    StateT* ssm_state,
    T* core_out,
    std::size_t B,
    std::size_t nslots) {
  const sycl::nd_range<2> ndr(
      sycl::range<2>(B * kHV, kV), sycl::range<2>(1, 128));
  return q.submit([&](sycl::handler& hnd) {
    sycl::local_accessor<float, 1> q_cache(128, hnd);
    sycl::local_accessor<float, 1> k_cache(128, hnd);
    hnd.parallel_for<QwenGdnRecurKernel<T, StateT, DtT>>(
        ndr, [=](sycl::nd_item<2> it) {
          const std::size_t nhv = it.get_global_id(0);
          const std::size_t v = it.get_local_id(1);
          const std::size_t b = nhv / kHV;
          const std::size_t hv = nhv - b * kHV;
          const std::size_t h = hv / (kHV / kH);
          const std::int32_t state_idx = state_indices[b];
          if (state_idx < 0 || static_cast<std::size_t>(state_idx) >= nslots) {
            core_out[(b * kHV + hv) * kV + v] = T(0);
            return;
          }

          const T* qrow = mixed_qkv + b * kConvDim + h * kK;
          const T* krow = mixed_qkv + b * kConvDim + kQDim + h * kK;
          const T* vrow = mixed_qkv + b * kConvDim + kQDim + kKDim + hv * kV;
          StateT* srow =
              ssm_state +
              (((static_cast<std::size_t>(state_idx) * kHV + hv) * kV + v) *
               kK);

          const auto grp = it.get_group();
          const float q_lane = to_f32(qrow[v]);
          const float k_lane = to_f32(krow[v]);
          const float q_norm = sycl::reduce_over_group(
                                   grp, q_lane * q_lane, sycl::plus<float>()) +
                               1e-6f;
          const float k_norm = sycl::reduce_over_group(
                                   grp, k_lane * k_lane, sycl::plus<float>()) +
                               1e-6f;
          const float q_inv = sycl::rsqrt(q_norm) * 0.08838834764831845f;
          const float k_inv = sycl::rsqrt(k_norm);
          q_cache[v] = q_lane * q_inv;
          k_cache[v] = k_lane * k_inv;
          sycl::group_barrier(grp);

          const float a_val = to_f32(projected_ba[b * kBADim + kHV + hv]);
          const float b_val = to_f32(projected_ba[b * kBADim + hv]);
          const float g =
              -sycl::exp(A_log[hv]) * softplus(a_val + to_f32(dt_bias[hv]));
          const float decay = sycl::exp(g);
          const float beta = 1.0f / (1.0f + sycl::exp(-b_val));

          float pred = 0.0f;
          for (std::size_t kk = 0; kk < kK; ++kk) {
            const float kval = k_cache[kk];
            const float sval = to_f32(srow[kk]) * decay;
            pred += sval * kval;
          }

          const float delta = (to_f32(vrow[v]) - pred) * beta;
          float out = 0.0f;
          for (std::size_t kk = 0; kk < kK; ++kk) {
            const float kval = k_cache[kk];
            const float qval = q_cache[kk];
            const float updated = to_f32(srow[kk]) * decay + delta * kval;
            srow[kk] = from_f32<StateT>(updated);
            out += updated * qval;
          }
          core_out[(b * kHV + hv) * kV + v] = from_f32<T>(out);
        });
  });
}

}  // namespace gdn_detail

inline sycl::event gdn_conv_launch(
    sycl::queue& q,
    ActDType dt,
    const void* projected_qkvz,
    const void* conv_weight,
    const void* conv_bias,
    const void* state_indices,
    void* conv_state,
    bool conv_dim_first,
    void* mixed_qkv,
    void* z_out,
    std::size_t B,
    std::size_t nslots) {
  using namespace gdn_detail;
#define DISPATCH(T)                                    \
  return gdn_conv_typed<T>(                            \
      q,                                               \
      static_cast<const T*>(projected_qkvz),           \
      static_cast<const T*>(conv_weight),              \
      static_cast<const T*>(conv_bias),                \
      static_cast<const std::int32_t*>(state_indices), \
      static_cast<T*>(conv_state),                     \
      conv_dim_first,                                  \
      static_cast<T*>(mixed_qkv),                      \
      static_cast<T*>(z_out),                          \
      B,                                               \
      nslots)
  switch (dt) {
    case ActDType::bf16:
      DISPATCH(bf16_t);
    case ActDType::f16:
      DISPATCH(half_t);
    case ActDType::f32:
      DISPATCH(float);
  }
#undef DISPATCH
  return {};
}

template <typename T, typename StateT>
inline sycl::event gdn_recur_dt_dispatch(
    sycl::queue& q,
    ActDType dt_dt_bias,
    const void* mixed_qkv,
    const void* projected_ba,
    const void* A_log,
    const void* dt_bias,
    const void* state_indices,
    void* ssm_state,
    void* core_out,
    std::size_t B,
    std::size_t nslots) {
  using namespace gdn_detail;
  if (dt_dt_bias == ActDType::f32) {
    return gdn_recur_typed<T, StateT, float>(
        q,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(projected_ba),
        static_cast<const float*>(A_log),
        static_cast<const float*>(dt_bias),
        static_cast<const std::int32_t*>(state_indices),
        static_cast<StateT*>(ssm_state),
        static_cast<T*>(core_out),
        B,
        nslots);
  }
  if (dt_dt_bias == ActDType::f16) {
    return gdn_recur_typed<T, StateT, half_t>(
        q,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(projected_ba),
        static_cast<const float*>(A_log),
        static_cast<const half_t*>(dt_bias),
        static_cast<const std::int32_t*>(state_indices),
        static_cast<StateT*>(ssm_state),
        static_cast<T*>(core_out),
        B,
        nslots);
  }
  return gdn_recur_typed<T, StateT, bf16_t>(
      q,
      static_cast<const T*>(mixed_qkv),
      static_cast<const T*>(projected_ba),
      static_cast<const float*>(A_log),
      static_cast<const bf16_t*>(dt_bias),
      static_cast<const std::int32_t*>(state_indices),
      static_cast<StateT*>(ssm_state),
      static_cast<T*>(core_out),
      B,
      nslots);
}

inline sycl::event gdn_recur_launch(
    sycl::queue& q,
    ActDType dt,
    ActDType state_dt,
    ActDType dt_bias_dt,
    const void* mixed_qkv,
    const void* projected_ba,
    const void* A_log,
    const void* dt_bias,
    const void* state_indices,
    void* ssm_state,
    void* core_out,
    std::size_t B,
    std::size_t nslots) {
#define DISPATCH_T(T)                          \
  do {                                         \
    if (state_dt == ActDType::f32)             \
      return gdn_recur_dt_dispatch<T, float>(  \
          q,                                   \
          dt_bias_dt,                          \
          mixed_qkv,                           \
          projected_ba,                        \
          A_log,                               \
          dt_bias,                             \
          state_indices,                       \
          ssm_state,                           \
          core_out,                            \
          B,                                   \
          nslots);                             \
    if (state_dt == ActDType::f16)             \
      return gdn_recur_dt_dispatch<T, half_t>( \
          q,                                   \
          dt_bias_dt,                          \
          mixed_qkv,                           \
          projected_ba,                        \
          A_log,                               \
          dt_bias,                             \
          state_indices,                       \
          ssm_state,                           \
          core_out,                            \
          B,                                   \
          nslots);                             \
    return gdn_recur_dt_dispatch<T, bf16_t>(   \
        q,                                     \
        dt_bias_dt,                            \
        mixed_qkv,                             \
        projected_ba,                          \
        A_log,                                 \
        dt_bias,                               \
        state_indices,                         \
        ssm_state,                             \
        core_out,                              \
        B,                                     \
        nslots);                               \
  } while (0)
  switch (dt) {
    case ActDType::bf16:
      DISPATCH_T(bf16_t);
    case ActDType::f16:
      DISPATCH_T(half_t);
    case ActDType::f32:
      DISPATCH_T(float);
  }
#undef DISPATCH_T
  return {};
}

}  // namespace vllm::xpu::decode
