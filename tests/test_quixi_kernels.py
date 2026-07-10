# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

import vllm_xpu_kernels._xpu_C  # noqa: F401


DEVICE = "xpu"
FP4_VALUES = torch.tensor([
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
])


def _make_nvfp4(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, shape, dtype=torch.int64,
                          generator=generator)
    packed = (codes[..., 0::2] |
              (codes[..., 1::2] << 4)).to(torch.uint8).to(DEVICE)
    scale_values = torch.rand((*shape[:-1], shape[-1] // 16),
                              generator=generator) + 0.5
    scales = scale_values.to(DEVICE).to(torch.float8_e4m3fn)
    decoded = FP4_VALUES[codes].to(DEVICE)
    decoded = decoded * scales.float().repeat_interleave(16, dim=-1)
    return packed.contiguous(), scales.contiguous(), decoded


@torch.inference_mode()
def test_quixi_nvfp4_gemm_matches_dequantized_reference():
    torch.manual_seed(0)
    m, n, k = 3, 40, 64
    global_scale = 0.125
    x = (torch.randn(m, k, device=DEVICE) * 0.1).to(torch.bfloat16)
    weight, scales, decoded = _make_nvfp4((n, k), seed=1)

    output = torch.ops._xpu_C.quixi_nvfp4_gemm(
        x, weight, scales, global_scale)
    reference = x.float() @ (decoded * global_scale).T

    torch.testing.assert_close(output.float(), reference, atol=4e-2,
                               rtol=2e-2)


@torch.inference_mode()
def test_quixi_fp8_gemm_matches_dequantized_reference():
    torch.manual_seed(2)
    m, n, k = 3, 40, 64
    x = (torch.randn(m, k, device=DEVICE) * 0.1).to(torch.bfloat16)
    weight = (torch.randn(n, k, device=DEVICE) * 0.2).to(
        torch.float8_e4m3fn)
    scale = torch.rand(n, device=DEVICE, dtype=torch.float32) + 0.5

    output = torch.ops._xpu_C.quixi_fp8_gemm_w8a16(
        x, weight.view(torch.uint8), scale, 0, True)
    reference = x.float() @ (weight.float() * scale[:, None]).T

    torch.testing.assert_close(output.float(), reference, atol=4e-2,
                               rtol=2e-2)


@torch.inference_mode()
def test_quixi_nvfp4_moe_variants_match_reference():
    torch.manual_seed(3)
    m, experts, topk, hidden_size, intermediate_size = 2, 3, 2, 32, 32
    hidden = (torch.randn(m, hidden_size, device=DEVICE) * 0.1).to(
        torch.bfloat16)
    topk_ids = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32,
                            device=DEVICE)
    topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]],
                                dtype=torch.float32, device=DEVICE)

    w13, w13_scale, w13_decoded = _make_nvfp4(
        (experts, 2 * intermediate_size, hidden_size), seed=4)
    w2, w2_scale, w2_decoded = _make_nvfp4(
        (experts, hidden_size, intermediate_size), seed=5)
    w13_global = torch.tensor([0.025, 0.02, 0.03], device=DEVICE)
    w2_global = torch.tensor([0.025, 0.03, 0.02], device=DEVICE)

    reference = torch.zeros(m, hidden_size, dtype=torch.float32, device=DEVICE)
    for row in range(m):
        for route in range(topk):
            expert = int(topk_ids[row, route])
            gate_up = hidden[row].float() @ (
                w13_decoded[expert] * w13_global[expert]).T
            activated = F.silu(gate_up[:intermediate_size]) * gate_up[
                intermediate_size:]
            expert_output = activated @ (
                w2_decoded[expert] * w2_global[expert]).T
            reference[row] += topk_weights[row, route] * expert_output

    args = (hidden, topk_ids, topk_weights, w13, w13_scale, w13_global,
            w2, w2_scale, w2_global, True)
    fused = torch.ops._xpu_C.quixi_nvfp4_moe(*args)
    split = torch.ops._xpu_C.quixi_nvfp4_moe_split(*args)

    torch.testing.assert_close(fused, reference, atol=3e-3, rtol=2e-2)
    torch.testing.assert_close(split, reference, atol=3e-3, rtol=2e-2)


@torch.inference_mode()
def test_quixi_rms_norm_and_fused_add_match_reference():
    torch.manual_seed(6)
    epsilon = 1e-6
    x = torch.randn(3, 128, device=DEVICE, dtype=torch.bfloat16)
    weight = (torch.randn(128, device=DEVICE) * 0.1 + 1).to(torch.bfloat16)

    output = torch.empty_like(x)
    torch.ops._xpu_C.quixi_rms_norm(output, x, weight, epsilon, None)
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(dim=-1, keepdim=True) + epsilon)
    reference = normalized.to(x.dtype).mul(weight)
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)

    residual = torch.randn_like(x)
    residual_reference = (x.float() + residual.float()).to(x.dtype)
    summed = x.float() + residual.float()
    normalized = summed * torch.rsqrt(
        summed.square().mean(dim=-1, keepdim=True) + epsilon)
    reference = normalized.to(x.dtype).mul(weight)
    torch.ops._xpu_C.quixi_rms_norm(output, x, weight, epsilon, residual)

    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(residual, residual_reference, atol=0, rtol=0)


def _reference_gdn_decode(projected_qkvz, projected_ba, conv_state,
                          ssm_state, conv_weight, conv_bias, a_log, dt_bias,
                          state_indices):
    batch = projected_qkvz.shape[0]
    num_k_heads, num_v_heads, head_dim = 16, 32, 128
    q_dim = num_k_heads * head_dim
    k_dim = num_k_heads * head_dim
    v_dim = num_v_heads * head_dim
    conv_dim = q_dim + k_dim + v_dim
    slots = state_indices.long()

    x = projected_qkvz[:, :conv_dim]
    z = projected_qkvz[:, conv_dim:].reshape(batch, num_v_heads,
                                             head_dim).clone()
    history = conv_state[slots].clone()
    x4 = torch.cat([history, x.unsqueeze(1)], dim=1).float()
    mixed = F.silu(
        (x4 * conv_weight.T.unsqueeze(0).float()).sum(dim=1) +
        conv_bias.float()).to(projected_qkvz.dtype)

    conv_state[slots, 0] = history[:, 1]
    conv_state[slots, 1] = history[:, 2]
    conv_state[slots, 2] = x

    q = mixed[:, :q_dim].reshape(batch, num_k_heads, head_dim).float()
    k = mixed[:, q_dim:q_dim + k_dim].reshape(
        batch, num_k_heads, head_dim).float()
    v = mixed[:, q_dim + k_dim:].reshape(
        batch, num_v_heads, head_dim).float()
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    q = q * (head_dim**-0.5)

    beta = torch.sigmoid(projected_ba[:, :num_v_heads].float())
    decay = torch.exp(-torch.exp(a_log.float()) * F.softplus(
        projected_ba[:, num_v_heads:].float() + dt_bias.float()))
    state = ssm_state[slots].float()
    output = torch.empty(batch, num_v_heads, head_dim, device=DEVICE)
    heads_per_k = num_v_heads // num_k_heads
    for v_head in range(num_v_heads):
        k_head = v_head // heads_per_k
        current = state[:, v_head] * decay[:, v_head, None, None]
        prediction = (current * k[:, k_head, None]).sum(dim=-1)
        delta = (v[:, v_head] - prediction) * beta[:, v_head, None]
        current = current + delta[..., None] * k[:, k_head, None]
        output[:, v_head] = (current * q[:, k_head, None]).sum(dim=-1)
        state[:, v_head] = current
    ssm_state[slots] = state.to(ssm_state.dtype)
    return output.to(projected_qkvz.dtype), z


@torch.inference_mode()
def test_quixi_qwen_gdn_decode_matches_reference():
    torch.manual_seed(7)
    batch, slots = 2, 4
    dtype = torch.bfloat16
    projected_qkvz = (torch.randn(batch, 12288, device=DEVICE) * 0.03).to(
        dtype)
    projected_ba = (torch.randn(batch, 64, device=DEVICE) * 0.03).to(dtype)
    conv_state = (torch.randn(slots, 3, 8192, device=DEVICE) * 0.03).to(dtype)
    ssm_state = torch.randn(slots, 32, 128, 128, device=DEVICE) * 0.03
    conv_weight = (torch.randn(8192, 4, device=DEVICE) * 0.03).to(dtype)
    conv_bias = (torch.randn(8192, device=DEVICE) * 0.01).to(dtype)
    a_log = (torch.randn(32, device=DEVICE) * 0.02).float()
    dt_bias = (torch.randn(32, device=DEVICE) * 0.02).to(dtype)
    state_indices = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)

    conv_reference = conv_state.clone()
    ssm_reference = ssm_state.clone()
    output_reference, z_reference = _reference_gdn_decode(
        projected_qkvz, projected_ba, conv_reference, ssm_reference,
        conv_weight, conv_bias, a_log, dt_bias, state_indices)

    output, z = torch.ops._xpu_C.quixi_qwen_gdn_decode(
        projected_qkvz, projected_ba, conv_state, ssm_state, conv_weight,
        conv_bias, a_log, dt_bias, state_indices)

    torch.testing.assert_close(output, output_reference, atol=2e-5, rtol=1e-2)
    torch.testing.assert_close(z, z_reference, atol=0, rtol=0)
    torch.testing.assert_close(conv_state, conv_reference, atol=0, rtol=0)
    torch.testing.assert_close(ssm_state, ssm_reference, atol=2e-7, rtol=1e-2)
