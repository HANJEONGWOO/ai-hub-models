// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
// Split delta-KV LLM session over QNN context binaries.
//
// Graph I/O roles come from tensor names, so TurboQuant packed KV streams
// (tq_<kind>_<layer>_{packed,norm}_{in,out}) are handled like int8 KV streams.
// The "pyref" layout reproduces ai-hub-models HubCompatibleGenerator inputs:
// tokens and past KV right-aligned, pad tokens in front, mask clipped to maskMin.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "qnn_api.h"

namespace tqrun {

struct RopeTable {
  size_t positions = 0;
  size_t half = 0;
  std::vector<float> cos;  // [positions * half]
  std::vector<float> sin;
};

struct SessionOptions {
  std::vector<std::string> bins;
  int contextLength = 1024;
  std::vector<int> sequenceLengths{128, 1};
  std::string graphSuffix;
  int32_t padToken = 151645;
  double maskMin = -100.0;
};

struct StepRecord {
  int ar = 0;
  int newTokens = 0;
  size_t cachedBefore = 0;
  double prepareSeconds = 0.0;
  double commitSeconds = 0.0;
  std::vector<double> partSeconds;
  std::vector<std::vector<ProfileEvent>> partProfiles;  // filled only for profiled steps
  double totalSeconds = 0.0;
};

struct KvStreamSummary {
  std::string name;
  std::string dtype;
  size_t bytesPerToken = 0;
};

class LlmSession {
 public:
  LlmSession(QnnRuntime& rt, const SessionOptions& options, const RopeTable& rope);
  ~LlmSession();

  // Runs one graph set over `tokens` (at most `ar`) and appends them to the KV cache.
  StepRecord step(const std::vector<int32_t>& tokens, int ar, bool profile = false);
  void reset();

  size_t cached() const { return cached_; }
  int contextLength() const { return options_.contextLength; }
  // Row i of the last step's valid logits, dequantized.
  void logitsRow(int i, std::vector<float>& out) const;
  size_t vocabSize() const;

  double loadSeconds() const { return loadSeconds_; }
  const std::vector<std::unique_ptr<ContextBinary>>& contexts() const { return contexts_; }
  std::vector<KvStreamSummary> kvStreams() const;
  size_t kvStoreBytes() const;
  size_t ioBufferBytes() const;
  uint64_t kvCopyBytes() const { return kvCopyBytes_; }

 private:
  struct Buffer;
  struct KvStream;
  struct GraphSet;

  void bindGraphSet(int ar);
  Buffer& sharedBuffer(GraphSet& set, const TensorMeta& meta);

  QnnRuntime& rt_;
  SessionOptions options_;
  const RopeTable& rope_;
  std::vector<std::unique_ptr<ContextBinary>> contexts_;
  std::map<int, std::unique_ptr<GraphSet>> sets_;
  std::map<std::string, std::unique_ptr<KvStream>> streams_;
  size_t cached_ = 0;
  int lastAr_ = 0;
  int lastNew_ = 0;
  double loadSeconds_ = 0.0;
  uint64_t kvCopyBytes_ = 0;
};

}  // namespace tqrun
