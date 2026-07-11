// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/xpu/XPUGraph.h>
#include <c10/xpu/XPUStream.h>
#include <torch/custom_class.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

class VllmXpuGraph final : public torch::CustomClassHolder {
 public:
  explicit VllmXpuGraph(bool keep_graph);

  void
  capture_begin(std::optional<std::tuple<std::int64_t, std::int64_t>> pool);
  void capture_end();
  void instantiate();
  void replay();
  void reset();
  std::tuple<std::int64_t, std::int64_t> pool() const;
  void synchronize();
  void enable_debug_mode();
  void debug_dump(const std::string& path);

 private:
  at::xpu::XPUGraph graph_;
  std::optional<c10::xpu::XPUStream> capture_stream_;
};

void synchronize_current_stream();
