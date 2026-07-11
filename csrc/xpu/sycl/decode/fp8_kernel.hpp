// SPDX-License-Identifier: MIT
// Copyright (c) 2026 QuixiAI

// Native FP8 (bf16 activation × fp8 weight) decode GEMV for Intel XPU.
//
// For W8A16: activation stays bf16/fp16/f32, only the weight is fp8-decoded.
// The fp8 bit-relocation decode is vendored from QuixiCore's fp8_gemv
// (no LUT): e5m2 is a truncated f16; e4m3 relocates into the f16 grid with a
// uniform 2^8 factor, folded once into the per-output scale.
//
// Weight [N, K] fp8 (subgroup-per-output-row), per-channel scale[N] or a single
// per-tensor scale. y[n] = scale[n] * sum_k fp8(w[n,k]) * x[k]. K multiple
// of 16.

#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "nvfp4_kernel.hpp"  // ActDType, half_t, bf16_t

namespace vllm::xpu::decode {
namespace fp8_detail {

inline constexpr int kSG = 32;
inline constexpr int kRowsPerWG = 8;
inline constexpr int kWG = kSG * kRowsPerWG;

using U8x16 = sycl::vec<std::uint8_t, 16>;

// e4m3 byte -> f16 value*2^-8 ; e5m2 byte -> f16 (exact). KIND: 0=e4m3, 1=e5m2.
template <int KIND>
inline float fp8_dec(std::uint8_t b) {
  if constexpr (KIND == 1) {
    return static_cast<float>(
        sycl::bit_cast<sycl::half>(static_cast<std::uint16_t>(b << 8)));
  } else {
    const auto h =
        static_cast<std::uint16_t>(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
    return static_cast<float>(sycl::bit_cast<sycl::half>(h));
  }
}

template <typename T, int KIND>
class Fp8GemvKernel;

// x[M,K] (T) @ dequant(W[N,K])^T -> y[M,N]. scale is [N] (per-channel) or [1].
template <typename T, int KIND>
sycl::event fp8_gemv_typed(
    sycl::queue& q,
    const T* x,
    const std::uint8_t* w,
    const float* scale,
    bool per_channel,
    T* y,
    std::size_t M,
    std::size_t N,
    std::size_t K) {
  const std::size_t nchunks = K / 16;  // 16 fp8 per chunk
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  // e4m3 carries a uniform 2^8 per weight element; fold once here.
  const float kcomp = (KIND == 0) ? 256.0f : 1.0f;
  const sycl::nd_range<2> ndr(
      sycl::range<2>(M, nwg * kWG), sycl::range<2>(1, kWG));
  return q.parallel_for<Fp8GemvKernel<T, KIND>>(
      ndr, [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
        const std::size_t m = it.get_global_id(0);
        const sycl::sub_group sg = it.get_sub_group();
        const int sgid = static_cast<int>(sg.get_group_linear_id());
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const std::size_t n = it.get_group(1) * kRowsPerWG + sgid;
        if (n >= N) return;
        const std::uint8_t* wrow = w + n * K;
        const T* xrow = x + m * K;
        float acc = 0.0f;
        for (std::size_t c = lane; c < nchunks; c += kSG) {
          const U8x16 wv = *reinterpret_cast<const U8x16*>(wrow + c * 16);
          const sycl::vec<T, 8> x0 =
              *reinterpret_cast<const sycl::vec<T, 8>*>(xrow + c * 16);
          const sycl::vec<T, 8> x1 =
              *reinterpret_cast<const sycl::vec<T, 8>*>(xrow + c * 16 + 8);
#pragma unroll
          for (int i = 0; i < 8; ++i)
            acc += fp8_dec<KIND>(wv[i]) * static_cast<float>(x0[i]);
#pragma unroll
          for (int i = 0; i < 8; ++i)
            acc += fp8_dec<KIND>(wv[i + 8]) * static_cast<float>(x1[i]);
        }
        const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) {
          const float s = (per_channel ? scale[n] : scale[0]) * kcomp;
          y[m * N + n] = static_cast<T>(sum * s);
        }
      });
}

}  // namespace fp8_detail

// kind: 0=e4m3, 1=e5m2.
inline sycl::event fp8_gemv_launch(
    sycl::queue& q,
    ActDType dt,
    int kind,
    const void* x,
    const void* w,
    const void* scale,
    bool per_channel,
    void* y,
    std::size_t M,
    std::size_t N,
    std::size_t K) {
  using namespace fp8_detail;
  const auto* wu = static_cast<const std::uint8_t*>(w);
  const auto* sc = static_cast<const float*>(scale);
#define DISPATCH(T)                 \
  do {                              \
    if (kind == 1)                  \
      return fp8_gemv_typed<T, 1>(  \
          q,                        \
          static_cast<const T*>(x), \
          wu,                       \
          sc,                       \
          per_channel,              \
          static_cast<T*>(y),       \
          M,                        \
          N,                        \
          K);                       \
    return fp8_gemv_typed<T, 0>(    \
        q,                          \
        static_cast<const T*>(x),   \
        wu,                         \
        sc,                         \
        per_channel,                \
        static_cast<T*>(y),         \
        M,                          \
        N,                          \
        K);                         \
  } while (0)
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

}  // namespace vllm::xpu::decode
