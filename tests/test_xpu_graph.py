# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_xpu_kernels.xpu_graph import XPUGraph, synchronize_current_stream

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU is required"
)


def test_xpu_graph_replays_on_capture_stream():
    stream = torch.xpu.Stream()
    value = torch.zeros(4, device="xpu")
    value.add_(0)
    synchronize_current_stream()

    graph = XPUGraph()
    with torch.xpu.stream(stream):
        graph.capture_begin()
        value.add_(1)
        graph.capture_end()
    graph.synchronize()

    value.zero_()
    synchronize_current_stream()
    graph.replay()
    graph.synchronize()

    torch.testing.assert_close(value.cpu(), torch.ones(4))
    graph.reset()


def test_xpu_graph_pool_instantiation_and_reset():
    stream = torch.xpu.Stream()
    pool = torch.xpu.graph_pool_handle()
    value = torch.zeros(1, device="xpu")
    graph = XPUGraph(keep_graph=True)

    with torch.xpu.stream(stream):
        graph.capture_begin(pool)
        value.add_(2)
        graph.capture_end()
    graph.instantiate()
    graph.synchronize()

    assert graph.pool() == pool
    value.zero_()
    synchronize_current_stream()
    graph.replay()
    graph.synchronize()
    torch.testing.assert_close(value.cpu(), torch.full((1, ), 2.0))

    graph.reset()
    with pytest.raises(RuntimeError, match="no capture stream"):
        graph.synchronize()


def test_xpu_graph_rejects_negative_pool_values():
    graph = XPUGraph()
    with pytest.raises(RuntimeError, match="must be non-negative"):
        graph.capture_begin((-1, 0))


def test_fp8_decode_fast_path_graph_capture():
    torch.manual_seed(9)
    stream = torch.xpu.Stream()
    x = torch.randn(2, 64, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(40, 64, device="xpu").to(torch.float8_e4m3fn)
    scale = torch.rand(40, device="xpu", dtype=torch.float32) + 0.5

    weight_t = weight.transpose(0, 1)
    torch.ops._xpu_C.fp8_gemm_w8a16(x, weight_t, scale, None)
    synchronize_current_stream()
    graph = XPUGraph()
    with torch.xpu.stream(stream):
        graph.capture_begin()
        output = torch.ops._xpu_C.fp8_gemm_w8a16(
            x, weight_t, scale, None)
        graph.capture_end()
    graph.synchronize()

    new_x = torch.randn_like(x)
    x.copy_(new_x)
    synchronize_current_stream()
    reference = new_x.float() @ (weight.float() * scale[:, None]).T
    graph.replay()
    graph.synchronize()

    torch.testing.assert_close(output.float(), reference, atol=4e-2,
                               rtol=2e-2)
    graph.reset()


def test_nvfp4_split_moe_graph_capture():
    torch.manual_seed(10)
    stream = torch.xpu.Stream()
    m, experts, topk, hidden_size, intermediate_size = 1, 2, 1, 32, 32
    hidden = torch.randn(
        m, hidden_size, device="xpu", dtype=torch.bfloat16)
    topk_ids = torch.zeros((m, topk), device="xpu", dtype=torch.int32)
    topk_weights = torch.ones((m, topk), device="xpu")
    w13 = torch.full(
        (experts, 2 * intermediate_size, hidden_size // 2),
        0x11,
        device="xpu",
        dtype=torch.uint8,
    )
    w13_scale = torch.ones(
        (experts, 2 * intermediate_size, hidden_size // 16),
        device="xpu",
        dtype=torch.float8_e4m3fn,
    )
    w2 = torch.full(
        (experts, hidden_size, intermediate_size // 2),
        0x11,
        device="xpu",
        dtype=torch.uint8,
    )
    w2_scale = torch.ones(
        (experts, hidden_size, intermediate_size // 16),
        device="xpu",
        dtype=torch.float8_e4m3fn,
    )
    global_scale = torch.full((experts,), 2.0**-22, device="xpu")
    args = (
        hidden,
        topk_ids,
        topk_weights,
        w13,
        w13_scale,
        global_scale,
        w2,
        w2_scale,
        global_scale,
        True,
    )

    torch.ops._xpu_C.nvfp4_moe(*args)
    synchronize_current_stream()
    graph = XPUGraph()
    with torch.xpu.stream(stream):
        graph.capture_begin()
        output = torch.ops._xpu_C.nvfp4_moe(*args)
        graph.capture_end()
    graph.synchronize()

    hidden.copy_(torch.randn_like(hidden))
    synchronize_current_stream()
    reference = torch.ops._xpu_C.nvfp4_moe(*args)
    synchronize_current_stream()
    graph.replay()
    graph.synchronize()

    torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)
    graph.reset()
