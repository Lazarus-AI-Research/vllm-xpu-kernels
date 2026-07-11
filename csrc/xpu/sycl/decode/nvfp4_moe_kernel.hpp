// SPDX-License-Identifier: MIT
// Copyright (c) 2026 QuixiAI

// Native NVFP4 fused MoE expert kernel for Intel XPU (decode-oriented).
//
// Replaces vLLM's Triton NVFP4 emulation MoE at decode: computes exactly the
// routed (token, expert) pairs — no BLOCK_M padding — fusing both expert GEMVs
// and the SwiGLU into a single kernel launch. Reuses the vendored e2m1/e4m3
// bit-relocation decode (nvfp4_kernel.hpp).
//
// One work-group per routed (m, t) pair (grid = M*T):
//   g[2I] = dequant(w13[e]) @ hidden[m]         (subgroup-per-output-row GEMV)
//   a[I]  = silu(g[:I]) * g[I:]                  (SwiGLU)
//   o[K]  = dequant(w2[e]) @ a
//   out_f32[m] += topk_weight * o                (atomic; caller zero-inits +
//   casts)
//
// Weight layout (ModelOpt NVFP4, non-swizzled), per expert e:
//   w13[e]        u8   [2I, K/2]     w13_scale[e] fp8e4m3 [2I, K/16] w13_gs[e]
//   f32 w2[e]         u8   [K,  I/2]     w2_scale[e]  fp8e4m3 [K,  I/16]
//   w2_gs[e]  f32

#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "nvfp4_kernel.hpp"

namespace vllm::xpu::decode {
namespace moe_detail {

using detail::e2m1_dec2;
using detail::e4m3_dec_raw;
using detail::WVec;

inline constexpr int kSG = 32;
inline constexpr int kSubgroups = 8;
inline constexpr int kWG = kSG * kSubgroups;  // 256

// One subgroup computes y = sum_k dequant(w_row[K]) * x[k]. K multiple of 32.
// `gscale` must already include the 2^22 bit-relocation compensation.
template <typename XT>
inline float nvfp4_row_dot(
    const sycl::sub_group& sg,
    const std::uint8_t* w_row,
    const std::uint8_t* s_row,
    float gscale,
    const XT* x,
    std::size_t K) {
  const std::size_t nchunks = K / 32;  // 16-byte chunk = 32 fp4 = two 16-blocks
  const WVec* wv = reinterpret_cast<const WVec*>(w_row);
  const int lane = static_cast<int>(sg.get_local_linear_id());
  float acc = 0.0f;
  for (std::size_t c = lane; c < nchunks; c += kSG) {
    const WVec chunk = wv[c];
    const float s0 = e4m3_dec_raw(s_row[2 * c]) * gscale;
    const float s1 = e4m3_dec_raw(s_row[2 * c + 1]) * gscale;
    const XT* xc = x + c * 32;
    float aw[4];
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
      const std::uint32_t word = chunk[wi];
      const XT* xw = xc + wi * 8;
      float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const sycl::vec<float, 2> v = e2m1_dec2(word >> (4 * p));
        a0 += v[0] * static_cast<float>(xw[p]);
        a1 += v[1] * static_cast<float>(xw[p + 4]);
      }
      aw[wi] = a0 + a1;
    }
    acc += (aw[0] + aw[1]) * s0 + (aw[2] + aw[3]) * s1;
  }
  return sycl::reduce_over_group(sg, acc, sycl::plus<float>());
}

inline float siluf(float x) { return x / (1.0f + sycl::exp(-x)); }

template <typename T>
class Nvfp4MoeKernel;

// hidden[M,K] (T), topk_ids[M,Tk] int32, topk_weights[M,Tk] f32, out_f32[M,K].
template <typename T>
sycl::event nvfp4_moe_typed(
    sycl::queue& q,
    const T* hidden,
    const std::int32_t* topk_ids,
    const float* topk_w,
    const std::uint8_t* w13,
    const std::uint8_t* w13s,
    const float* w13_gs,
    const std::uint8_t* w2,
    const std::uint8_t* w2s,
    const float* w2_gs,
    float* out_f32,
    std::size_t M,
    std::size_t Tk,
    std::size_t E,
    std::size_t K,
    std::size_t I,
    bool mul_weight) {
  const std::size_t twoI = 2 * I;
  const std::size_t w13_estride = twoI * (K / 2);
  const std::size_t s13_estride = twoI * (K / 16);
  const std::size_t w2_estride = K * (I / 2);
  const std::size_t s2_estride = K * (I / 16);
  const std::size_t npairs = M * Tk;
  const sycl::nd_range<1> ndr(
      sycl::range<1>(npairs * kWG), sycl::range<1>(kWG));

  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> g(sycl::range<1>(twoI), h);
    sycl::local_accessor<float, 1> a(sycl::range<1>(I), h);
    h.parallel_for<Nvfp4MoeKernel<T>>(
        ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t pid = it.get_group(0);
          const std::size_t m = pid / Tk;
          const std::size_t t = pid % Tk;
          const std::int32_t e = topk_ids[m * Tk + t];
          if (e < 0 || static_cast<std::size_t>(e) >= E) return;
          const float rw = mul_weight ? topk_w[m * Tk + t] : 1.0f;

          const sycl::sub_group sg = it.get_sub_group();
          const int sgid = static_cast<int>(sg.get_group_linear_id());
          const int lane = static_cast<int>(sg.get_local_linear_id());
          const int tid = static_cast<int>(it.get_local_linear_id());

          const std::uint8_t* w13e = w13 + e * w13_estride;
          const std::uint8_t* s13e = w13s + e * s13_estride;
          const float gs13 = w13_gs[e] * 4194304.0f;  // 2^22
          const T* xrow = hidden + m * K;

          // g[row] = dequant(w13[e][row]) . hidden[m]
          for (std::size_t row = sgid; row < twoI; row += kSubgroups) {
            const float v = nvfp4_row_dot(
                sg, w13e + row * (K / 2), s13e + row * (K / 16), gs13, xrow, K);
            if (lane == 0) g[row] = v;
          }
          sycl::group_barrier(it.get_group());

          // SwiGLU: a[i] = silu(g[i]) * g[i+I]
          for (std::size_t i = tid; i < I; i += kWG)
            a[i] = siluf(g[i]) * g[i + I];
          sycl::group_barrier(it.get_group());

          // o[row] = dequant(w2[e][row]) . a ; accumulate weighted into
          // out_f32[m]
          const std::uint8_t* w2e = w2 + e * w2_estride;
          const std::uint8_t* s2e = w2s + e * s2_estride;
          const float gs2 = w2_gs[e] * 4194304.0f;
          const float* aptr = &a[0];
          for (std::size_t row = sgid; row < K; row += kSubgroups) {
            const float v = nvfp4_row_dot(
                sg, w2e + row * (I / 2), s2e + row * (I / 16), gs2, aptr, I);
            if (lane == 0) {
              sycl::atomic_ref<
                  float,
                  sycl::memory_order::relaxed,
                  sycl::memory_scope::device,
                  sycl::access::address_space::global_space>
                  ao(out_f32[m * K + row]);
              ao.fetch_add(rw * v);
            }
          }
        });
  });
}

// ---- High-occupancy 2-kernel split ----
// The fused (1 WG/pair) kernel is occupancy-starved at decode (M*T WGs). Split
// so output rows spread across many WGs: kernel G computes g[2I] per (pair,
// row-tile) -> g_buf; kernel O recomputes SwiGLU a[I] from g_buf and computes
// o[K] per (pair, row-tile), atomic weighted-sum into out_f32.

template <typename T>
class Nvfp4MoeGKernel;
template <typename T>
class Nvfp4MoeOKernel;

template <typename T>
sycl::event nvfp4_moe_g_typed(
    sycl::queue& q,
    const T* hidden,
    const std::int32_t* topk_ids,
    const std::uint8_t* w13,
    const std::uint8_t* w13s,
    const float* w13_gs,
    float* g_buf,
    std::size_t P,
    std::size_t Tk,
    std::size_t E,
    std::size_t K,
    std::size_t I) {
  const std::size_t twoI = 2 * I;
  const std::size_t w13_estride = twoI * (K / 2);
  const std::size_t s13_estride = twoI * (K / 16);
  const std::size_t nrt = (twoI + kSubgroups - 1) / kSubgroups;  // row-tiles
  const sycl::nd_range<2> ndr(
      sycl::range<2>(P, nrt * kWG), sycl::range<2>(1, kWG));
  return q.parallel_for<Nvfp4MoeGKernel<T>>(
      ndr, [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
        const std::size_t pid = it.get_global_id(0);
        const std::int32_t e = topk_ids[pid];
        if (e < 0 || static_cast<std::size_t>(e) >= E) return;
        const std::size_t m = pid / Tk;
        const sycl::sub_group sg = it.get_sub_group();
        const int sgid = static_cast<int>(sg.get_group_linear_id());
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const std::size_t row = it.get_group(1) * kSubgroups + sgid;
        if (row >= twoI) return;
        const float gs = w13_gs[e] * 4194304.0f;
        const float v = nvfp4_row_dot(
            sg,
            w13 + e * w13_estride + row * (K / 2),
            w13s + e * s13_estride + row * (K / 16),
            gs,
            hidden + m * K,
            K);
        if (lane == 0) g_buf[pid * twoI + row] = v;
      });
}

template <typename T>
sycl::event nvfp4_moe_o_typed(
    sycl::queue& q,
    const std::int32_t* topk_ids,
    const float* topk_w,
    const std::uint8_t* w2,
    const std::uint8_t* w2s,
    const float* w2_gs,
    const float* g_buf,
    float* out_f32,
    std::size_t P,
    std::size_t Tk,
    std::size_t E,
    std::size_t K,
    std::size_t I,
    bool mul_weight) {
  const std::size_t twoI = 2 * I;
  const std::size_t w2_estride = K * (I / 2);
  const std::size_t s2_estride = K * (I / 16);
  const std::size_t nrt = (K + kSubgroups - 1) / kSubgroups;
  const sycl::nd_range<2> ndr(
      sycl::range<2>(P, nrt * kWG), sycl::range<2>(1, kWG));
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> a(sycl::range<1>(I), h);
    h.parallel_for<Nvfp4MoeOKernel<T>>(
        ndr, [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t pid = it.get_global_id(0);
          const std::int32_t e = topk_ids[pid];
          if (e < 0 || static_cast<std::size_t>(e) >= E) return;
          const std::size_t m = pid / Tk;
          const float rw = mul_weight ? topk_w[pid] : 1.0f;
          const sycl::sub_group sg = it.get_sub_group();
          const int sgid = static_cast<int>(sg.get_group_linear_id());
          const int lane = static_cast<int>(sg.get_local_linear_id());
          const int tid = static_cast<int>(it.get_local_linear_id());
          // SwiGLU a[i] = silu(g[i]) * g[i+I] from g_buf (recomputed per WG).
          const float* gp = g_buf + pid * twoI;
          for (std::size_t i = tid; i < I; i += kWG)
            a[i] = siluf(gp[i]) * gp[i + I];
          sycl::group_barrier(it.get_group());
          const std::size_t row = it.get_group(1) * kSubgroups + sgid;
          if (row >= K) return;
          const float gs = w2_gs[e] * 4194304.0f;
          const float v = nvfp4_row_dot(
              sg,
              w2 + e * w2_estride + row * (I / 2),
              w2s + e * s2_estride + row * (I / 16),
              gs,
              &a[0],
              I);
          if (lane == 0) {
            sycl::atomic_ref<
                float,
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>
                ao(out_f32[m * K + row]);
            ao.fetch_add(rw * v);
          }
        });
  });
}

}  // namespace moe_detail

// Two-kernel high-occupancy MoE. g_buf is caller-provided f32 scratch [P, 2I].
inline void nvfp4_moe_split_launch(
    sycl::queue& q,
    ActDType dt,
    const void* hidden,
    const void* topk_ids,
    const void* topk_w,
    const void* w13,
    const void* w13s,
    const void* w13_gs,
    const void* w2,
    const void* w2s,
    const void* w2_gs,
    void* g_buf,
    void* out_f32,
    std::size_t M,
    std::size_t Tk,
    std::size_t E,
    std::size_t K,
    std::size_t I,
    bool mul_weight) {
  using namespace moe_detail;
  const std::size_t P = M * Tk;
  const auto* ti = static_cast<const std::int32_t*>(topk_ids);
  const auto* tw = static_cast<const float*>(topk_w);
  const auto* w13u = static_cast<const std::uint8_t*>(w13);
  const auto* s13u = static_cast<const std::uint8_t*>(w13s);
  const auto* g13 = static_cast<const float*>(w13_gs);
  const auto* w2u = static_cast<const std::uint8_t*>(w2);
  const auto* s2u = static_cast<const std::uint8_t*>(w2s);
  const auto* g2 = static_cast<const float*>(w2_gs);
  auto* gb = static_cast<float*>(g_buf);
  auto* of = static_cast<float*>(out_f32);
#define MOE_G(TY)                     \
  nvfp4_moe_g_typed(                  \
      q,                              \
      static_cast<const TY*>(hidden), \
      ti,                             \
      w13u,                           \
      s13u,                           \
      g13,                            \
      gb,                             \
      P,                              \
      Tk,                             \
      E,                              \
      K,                              \
      I)
  switch (dt) {
    case ActDType::bf16:
      MOE_G(bf16_t);
      break;
    case ActDType::f16:
      MOE_G(half_t);
      break;
    case ActDType::f32:
      MOE_G(float);
      break;
  }
#undef MOE_G
  nvfp4_moe_o_typed<bf16_t>(
      q, ti, tw, w2u, s2u, g2, gb, of, P, Tk, E, K, I, mul_weight);
}

inline sycl::event nvfp4_moe_launch(
    sycl::queue& q,
    ActDType dt,
    const void* hidden,
    const void* topk_ids,
    const void* topk_w,
    const void* w13,
    const void* w13s,
    const void* w13_gs,
    const void* w2,
    const void* w2s,
    const void* w2_gs,
    void* out_f32,
    std::size_t M,
    std::size_t Tk,
    std::size_t E,
    std::size_t K,
    std::size_t I,
    bool mul_weight) {
  using namespace moe_detail;
  const auto* ti = static_cast<const std::int32_t*>(topk_ids);
  const auto* tw = static_cast<const float*>(topk_w);
  const auto* w13u = static_cast<const std::uint8_t*>(w13);
  const auto* s13u = static_cast<const std::uint8_t*>(w13s);
  const auto* g13 = static_cast<const float*>(w13_gs);
  const auto* w2u = static_cast<const std::uint8_t*>(w2);
  const auto* s2u = static_cast<const std::uint8_t*>(w2s);
  const auto* g2 = static_cast<const float*>(w2_gs);
  auto* of = static_cast<float*>(out_f32);
  switch (dt) {
    case ActDType::bf16:
      return nvfp4_moe_typed(
          q,
          static_cast<const bf16_t*>(hidden),
          ti,
          tw,
          w13u,
          s13u,
          g13,
          w2u,
          s2u,
          g2,
          of,
          M,
          Tk,
          E,
          K,
          I,
          mul_weight);
    case ActDType::f16:
      return nvfp4_moe_typed(
          q,
          static_cast<const half_t*>(hidden),
          ti,
          tw,
          w13u,
          s13u,
          g13,
          w2u,
          s2u,
          g2,
          of,
          M,
          Tk,
          E,
          K,
          I,
          mul_weight);
    case ActDType::f32:
      return nvfp4_moe_typed(
          q,
          static_cast<const float*>(hidden),
          ti,
          tw,
          w13u,
          s13u,
          g13,
          w2u,
          s2u,
          g2,
          of,
          M,
          Tk,
          E,
          K,
          I,
          mul_weight);
  }
  return {};
}

}  // namespace vllm::xpu::decode
