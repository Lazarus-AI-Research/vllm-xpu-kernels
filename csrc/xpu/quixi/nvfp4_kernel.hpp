// SPDX-License-Identifier: MIT
// Copyright (c) 2026 QuixiAI

// Native NVFP4 (NVIDIA FP4) SYCL decoder + GEMV for Intel XPU.
//
// Vendored from QuixiCore-XPU
// (kernels/quantization/nvfp4_gemv/variants/xpu_sycl/nvfp4_gemv.sycl.cpp) so the
// kernel is editable in-tree. This is the entire numeric core: edit the decode
// helpers or the GEMV loop below and rebuild with ./build.sh.
//
// Format (matches ModelOpt NVFP4, swizzle=False):
//   weight W[N, K]   e2m1 fp4, packed 2 nibbles/byte along K (low nibble = even k)
//   scale  S[N, K/16] one e4m3 (fp8) byte per 16 weight elements
//   global per-tensor fp32 scale (amax / (6*448))
//   dequant:  w = e2m1(nibble) * e4m3(block_scale) * global_scale
//
// Decode trick: both e2m1 and e4m3 bit-relocate exactly into f16 (no LUT, no
// sign-select, no ldexp). Each product carries a uniform 2^-22, compensated by
// ONE multiply folded into the global scale (see nvfp4_gemv_launch).

#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

namespace quixi_nvfp4 {

using half_t = sycl::half;
using bf16_t = sycl::ext::oneapi::bfloat16;

enum class ActDType : std::uint8_t { f32, f16, bf16 };

namespace detail {

inline constexpr int kSG = 32;          // subgroup size (one per output row)
inline constexpr int kRowsPerWG = 8;    // output rows per work-group
inline constexpr int kWG = kSG * kRowsPerWG;

using WVec = sycl::vec<std::uint32_t, 4>;  // 16 bytes = 32 fp4 = two nvfp4 blocks

// Named kernel tag (torch's SyclExtension does not enable unnamed lambdas).
template <typename T>
class Nvfp4GemvKernel;

// e2m1 nibble s.e1e0.m -> f16 (s<<15)|(e<<10)|(m<<9) = value * 2^-14.
// Decodes nibbles j and j+4 of a word (pass word>>4j) in one masked-shift pass.
inline sycl::vec<float, 2> e2m1_dec2(std::uint32_t w) {
  const std::uint32_t t = w & 0x000F000Fu;
  const std::uint32_t h2 = ((t & 0x00080008u) << 12) | ((t & 0x00070007u) << 9);
  return sycl::bit_cast<sycl::vec<sycl::half, 2>>(h2).template convert<float>();
}

// e4m3 byte -> f16 ((b&0x80)<<8)|((b&0x7f)<<7) = value * 2^-8.
inline float e4m3_dec_raw(std::uint8_t b) {
  const auto h = static_cast<std::uint16_t>(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
  return static_cast<float>(sycl::bit_cast<sycl::half>(h));
}

// Single-vector NVFP4 GEMV: y[N] = dequant(W[N,K]) @ x[K]. `gscale` must arrive
// pre-multiplied by 2^22 to undo the raw bit-relocation decode factors.
template <typename T>
sycl::event nvfp4_gemv_typed(sycl::queue& q, const std::uint8_t* w,
                             const std::uint8_t* bscale, float gscale, const T* x,
                             T* y, std::size_t N, std::size_t K) {
  const std::size_t bytes_per_row = K / 2;
  const std::size_t blocks_per_row = K / 16;         // e4m3 scales per row
  const std::size_t nchunks = bytes_per_row / 16;     // 16-byte chunks (2 blocks)
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for<Nvfp4GemvKernel<T>>(ndr, [=](sycl::nd_item<1> it)
                                 [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;

    const WVec* wvrow = reinterpret_cast<const WVec*>(w + n * bytes_per_row);
    const std::uint8_t* srow = bscale + n * blocks_per_row;

    float acc = 0.0f;
    for (std::size_t c = lane; c < nchunks; c += kSG) {
      const WVec chunk = wvrow[c];
      const float s0 = e4m3_dec_raw(srow[2 * c]) * gscale;
      const float s1 = e4m3_dec_raw(srow[2 * c + 1]) * gscale;
      const T* xc = x + c * 32;
      float aw[4];
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {
        const std::uint32_t word = chunk[wi];
        const sycl::vec<T, 8> xv = *reinterpret_cast<const sycl::vec<T, 8>*>(xc + wi * 8);
        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {  // pair p decodes nibbles p and p+4
          const sycl::vec<float, 2> v = e2m1_dec2(word >> (4 * p));
          a0 += v[0] * static_cast<float>(xv[p]);
          a1 += v[1] * static_cast<float>(xv[p + 4]);
        }
        aw[wi] = a0 + a1;
      }
      acc += (aw[0] + aw[1]) * s0 + (aw[2] + aw[3]) * s1;
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// ---- M-tiled GEMM: decode each weight row once, reuse across M rows ----
// For batched decode (2<=M<=kMaxM) the plain gemv re-reads the full weight per
// row; here one subgroup owns output row n and accumulates M dot products,
// decoding each weight element a single time.
inline constexpr int kMaxM = 8;

template <typename T>
class Nvfp4GemmMKernel;

template <typename T>
sycl::event nvfp4_gemm_mtiled_typed(sycl::queue& q, const std::uint8_t* w,
                                    const std::uint8_t* bscale, float gscale,
                                    const T* x, T* y, std::size_t M, std::size_t N,
                                    std::size_t K) {
  const std::size_t bytes_per_row = K / 2;
  const std::size_t blocks_per_row = K / 16;
  const std::size_t nchunks = bytes_per_row / 16;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for<Nvfp4GemmMKernel<T>>(
      ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int sgid = static_cast<int>(sg.get_group_linear_id());
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
        if (n >= N) return;
        const WVec* wvrow = reinterpret_cast<const WVec*>(w + n * bytes_per_row);
        const std::uint8_t* srow = bscale + n * blocks_per_row;
        float acc[kMaxM];
#pragma unroll
        for (int m = 0; m < kMaxM; ++m) acc[m] = 0.0f;
        for (std::size_t c = lane; c < nchunks; c += kSG) {
          const WVec chunk = wvrow[c];
          const float s0 = e4m3_dec_raw(srow[2 * c]) * gscale;
          const float s1 = e4m3_dec_raw(srow[2 * c + 1]) * gscale;
          // Decode the 32 fp4 of this chunk once into scaled weight values.
          float wv[32];
#pragma unroll
          for (int wi = 0; wi < 4; ++wi) {
            const std::uint32_t word = chunk[wi];
            const float s = (wi < 2) ? s0 : s1;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
              const sycl::vec<float, 2> v = e2m1_dec2(word >> (4 * p));
              wv[wi * 8 + p] = v[0] * s;
              wv[wi * 8 + p + 4] = v[1] * s;
            }
          }
          const std::size_t base = c * 32;
          for (std::size_t m = 0; m < M; ++m) {
            const T* xc = x + m * K + base;
            float a = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) a += wv[i] * static_cast<float>(xc[i]);
            acc[m] += a;
          }
        }
        for (std::size_t m = 0; m < M; ++m) {
          const float sum = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
          if (lane == 0) y[m * N + n] = static_cast<T>(sum);
        }
      });
}

}  // namespace detail

inline sycl::event nvfp4_gemm_mtiled_launch(
    sycl::queue& q, const void* w_packed, const void* block_scales,
    float global_scale, const void* x, void* y, std::size_t M, std::size_t N,
    std::size_t K, ActDType act_dt) {
  const auto* w = static_cast<const std::uint8_t*>(w_packed);
  const auto* s = static_cast<const std::uint8_t*>(block_scales);
  global_scale *= 4194304.0f;  // 2^22
  using namespace detail;
  switch (act_dt) {
    case ActDType::f32:
      return nvfp4_gemm_mtiled_typed(q, w, s, global_scale,
                                     static_cast<const float*>(x),
                                     static_cast<float*>(y), M, N, K);
    case ActDType::f16:
      return nvfp4_gemm_mtiled_typed(q, w, s, global_scale,
                                     static_cast<const half_t*>(x),
                                     static_cast<half_t*>(y), M, N, K);
    case ActDType::bf16:
      return nvfp4_gemm_mtiled_typed(q, w, s, global_scale,
                                     static_cast<const bf16_t*>(x),
                                     static_cast<bf16_t*>(y), M, N, K);
  }
  return {};
}

// Dispatch a single NVFP4 GEMV for the given activation dtype. Applies the 2^22
// decode compensation to the per-tensor global scale.
inline sycl::event nvfp4_gemv_launch(sycl::queue& q, const void* w_packed,
                                     const void* block_scales, float global_scale,
                                     const void* x, void* y, std::size_t N,
                                     std::size_t K, ActDType act_dt) {
  const auto* w = static_cast<const std::uint8_t*>(w_packed);
  const auto* s = static_cast<const std::uint8_t*>(block_scales);
  global_scale *= 4194304.0f;  // 2^22
  using namespace detail;
  switch (act_dt) {
    case ActDType::f32:
      return nvfp4_gemv_typed(q, w, s, global_scale,
                              static_cast<const float*>(x), static_cast<float*>(y), N, K);
    case ActDType::f16:
      return nvfp4_gemv_typed(q, w, s, global_scale,
                              static_cast<const half_t*>(x), static_cast<half_t*>(y), N, K);
    case ActDType::bf16:
      return nvfp4_gemv_typed(q, w, s, global_scale,
                              static_cast<const bf16_t*>(x), static_cast<bf16_t*>(y), N, K);
  }
  return {};
}

}  // namespace quixi_nvfp4
