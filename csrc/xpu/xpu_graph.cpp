// SPDX-License-Identifier: Apache-2.0

#include "xpu/xpu_graph.h"

#include <c10/util/Exception.h>

VllmXpuGraph::VllmXpuGraph(bool keep_graph) : graph_(keep_graph) {}

void VllmXpuGraph::capture_begin(
    std::optional<std::tuple<std::int64_t, std::int64_t>> pool) {
  const auto capture_stream = c10::xpu::getCurrentXPUStream();
  c10::MempoolId_t pool_id{0, 0};
  if (pool.has_value()) {
    TORCH_CHECK(
        std::get<0>(*pool) >= 0 && std::get<1>(*pool) >= 0,
        "XPU graph pool values must be non-negative");
    pool_id = {
        static_cast<std::uint64_t>(std::get<0>(*pool)),
        static_cast<std::uint64_t>(std::get<1>(*pool)),
    };
  }
  graph_.capture_begin(pool_id);
  capture_stream_ = capture_stream;
}

void VllmXpuGraph::capture_end() { graph_.capture_end(); }

void VllmXpuGraph::instantiate() { graph_.instantiate(); }

void VllmXpuGraph::replay() { graph_.replay(); }

void VllmXpuGraph::reset() {
  graph_.reset();
  capture_stream_.reset();
}

std::tuple<std::int64_t, std::int64_t> VllmXpuGraph::pool() const {
  const auto pool_id = graph_.pool();
  return {
      static_cast<std::int64_t>(pool_id.first),
      static_cast<std::int64_t>(pool_id.second),
  };
}

void VllmXpuGraph::synchronize() {
  TORCH_CHECK(capture_stream_.has_value(), "XPU graph has no capture stream");
  capture_stream_->synchronize();
}

void VllmXpuGraph::enable_debug_mode() { graph_.enable_debug_mode(); }

void VllmXpuGraph::debug_dump(const std::string& path) {
  graph_.debug_dump(path);
}

void synchronize_current_stream() {
  c10::xpu::getCurrentXPUStream().synchronize();
}
