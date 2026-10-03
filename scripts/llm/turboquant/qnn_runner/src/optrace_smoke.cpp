// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// Single-graph plumbing check with synthetic zero inputs, NOT a benchmark.
#include "qnn_api.h"

#include <cstdio>
#include <exception>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 6 && argc != 8) {
      std::fprintf(stderr, "usage: smoke backend system context graph trace-dir [package provider]\n");
      return 2;
    }
    tqrun::QnnRuntime rt(argv[1], argv[2]);
    if (argc == 8) rt.registerOpPackage(argv[6], argv[7]);
    rt.setBurstPower();
    rt.enableOptrace(argv[5]);
    tqrun::ContextBinary context(rt, argv[3], {argv[4]});
    auto& graph = context.graph(argv[4]);
    std::vector<std::vector<uint8_t>> storage;
    storage.reserve(graph.inputs.size() + graph.outputs.size());
    auto allocate = [&](const std::vector<tqrun::TensorMeta>& metadata) {
      std::vector<Qnn_Tensor_t> tensors;
      for (const auto& meta : metadata) {
        storage.emplace_back(meta.bytes(), 0);
        auto tensor = meta.prototype;
        tensor.v1.name = meta.name.c_str();
        tensor.v1.dimensions = const_cast<uint32_t*>(meta.dims.data());
        tensor.v1.clientBuf = {storage.back().data(), static_cast<uint32_t>(meta.bytes())};
        tensors.push_back(tensor);
      }
      return tensors;
    };
    auto inputs = allocate(graph.inputs);
    auto outputs = allocate(graph.outputs);
    double seconds = 0;
    auto events = rt.executeProfiled(graph, inputs, outputs, &seconds);
    std::printf("Synthetic plumbing check complete: graph=%s events=%zu execute_s=%f\n",
                graph.name.c_str(), events.size(), seconds);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
