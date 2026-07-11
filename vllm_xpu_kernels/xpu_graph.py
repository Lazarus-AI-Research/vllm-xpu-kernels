# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


class XPUGraph:
    """Current-stream XPU graph wrapper used by vLLM."""

    def __init__(self, keep_graph: bool = False) -> None:
        self._graph = torch.classes._xpu_C.XPUGraph(keep_graph)

    def capture_begin(self, pool: tuple[int, int] | None = None) -> None:
        normalized_pool = (
            None if pool is None else (int(pool[0]), int(pool[1]))
        )
        self._graph.capture_begin(normalized_pool)

    def capture_end(self) -> None:
        self._graph.capture_end()

    def instantiate(self) -> None:
        self._graph.instantiate()

    def replay(self) -> None:
        self._graph.replay()

    def reset(self) -> None:
        self._graph.reset()

    def pool(self) -> tuple[int, int]:
        pool = self._graph.pool()
        return int(pool[0]), int(pool[1])

    def synchronize(self) -> None:
        self._graph.synchronize()

    def enable_debug_mode(self) -> None:
        self._graph.enable_debug_mode()

    def debug_dump(self, path: str) -> None:
        self._graph.debug_dump(path)


def synchronize_current_stream() -> None:
    """Wait for work submitted to the current XPU stream."""
    torch.ops._xpu_C.synchronize_current_stream()
