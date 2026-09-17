// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include "session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <regex>
#include <stdexcept>

#include "numeric.h"
#include "buckets.h"

namespace tqrun {
namespace {

using Clock = std::chrono::steady_clock;

double since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

const std::regex kKvIn(R"((past_(key|value)_\d+|tq_(key|value)_\d+_(packed|norm|scale))_in)");
const std::regex kKvOut(R"((past_(key|value)_\d+|tq_(key|value)_\d+_(packed|norm|scale))_out)");

// Token axis of a KV tensor: hub keys are (heads, 1, head_dim, tokens); everything else is -2.
int tokenAxis(const std::string& base, size_t rank) {
  if (rank != 4) fail("KV tensor " + base + " must be rank 4");
  return base.rfind("past_key_", 0) == 0 ? 3 : 2;
}

// Tiles one element's bytes over [dst, dst + bytes) with O(log n) memcpy calls.
void fillPattern(uint8_t* dst, size_t bytes, const std::vector<uint8_t>& element) {
  if (bytes == 0) return;
  const size_t elem = element.size();
  if (std::all_of(element.begin(), element.end(), [&](uint8_t b) { return b == element[0]; })) {
    std::memset(dst, element[0], bytes);
    return;
  }
  std::memcpy(dst, element.data(), elem);
  size_t filled = elem;
  while (filled < bytes) {
    const size_t chunk = std::min(filled, bytes - filled);
    std::memcpy(dst + filled, dst, chunk);
    filled += chunk;
  }
}

std::string graphName(int ar, int contextLength, size_t part, size_t numParts) {
  return std::string(ar == 1 ? "token" : "prompt") + "_ar" + std::to_string(ar) + "_cl" +
         std::to_string(contextLength) + "_" + std::to_string(part) + "_of_" + std::to_string(numParts);
}

}  // namespace

struct LlmSession::Buffer {
  const TensorMeta* meta = nullptr;
  std::vector<uint8_t> data;
};

struct LlmSession::KvStream {
  std::string base;
  int axis = 2;
  size_t outer = 0;
  size_t inner = 0;  // bytes per (outer row, token)
  std::vector<uint8_t> clear;  // one element's bytes for an empty slot
  std::map<std::pair<int, int>, const TensorMeta*> inMeta, outMeta;
  std::map<std::pair<int, int>, Buffer*> inBuf, outBuf;
  std::vector<uint8_t> store;  // [outer][contextLength][inner], tokens left-aligned
  std::string dtype;
};

struct LlmSession::GraphSet {
  int ar = 0;
  std::vector<GraphInfo*> parts;
  std::vector<std::vector<Qnn_Tensor_t>> inputs, outputs;
  std::map<std::string, std::unique_ptr<Buffer>> buffers;
  Buffer* tokens = nullptr;
  Buffer* mask = nullptr;
  Buffer* cos = nullptr;
  Buffer* sin = nullptr;
  Buffer* logits = nullptr;
};

LlmSession::LlmSession(QnnRuntime& rt, const SessionOptions& options, const RopeTable& rope)
    : rt_(rt), options_(options), rope_(rope) {
  if (rope_.positions < static_cast<size_t>(options_.contextLength)) {
    fail("RoPE table has fewer positions than the context length");
  }
  auto start = Clock::now();
  options_.contextBuckets = contextBuckets(options_.contextBuckets, options_.contextLength);
  const size_t numParts = options_.bins.size();
  for (size_t p = 0; p < numParts; ++p) {
    std::vector<std::string> names;
    for (int c : options_.contextBuckets) {
      for (int ar : options_.sequenceLengths) {
        if (ar < c) names.push_back(graphName(ar, c, p + 1, numParts) + options_.graphSuffix);
      }
    }
    contexts_.push_back(std::make_unique<ContextBinary>(rt_, options_.bins[p], names));
  }
  for (int c : options_.contextBuckets) {
    for (int ar : options_.sequenceLengths) if (ar < c) bindGraphSet(ar, c);
  }
  loadSeconds_ = since(start);
  reset();
}

LlmSession::~LlmSession() = default;

LlmSession::Buffer& LlmSession::sharedBuffer(GraphSet& set, const TensorMeta& meta) {
  auto it = set.buffers.find(meta.name);
  if (it == set.buffers.end()) {
    auto buf = std::make_unique<Buffer>();
    buf->meta = &meta;
    buf->data.assign(meta.bytes(), 0);
    it = set.buffers.emplace(meta.name, std::move(buf)).first;
    return *it->second;
  }
  const TensorMeta& prev = *it->second->meta;
  if (prev.bytes() != meta.bytes() || prev.dataType != meta.dataType || prev.quantized != meta.quantized ||
      prev.scale != meta.scale || prev.offset != meta.offset) {
    fail("tensor " + meta.name + " differs between parts (shape, dtype or quantization)");
  }
  return *it->second;
}

void LlmSession::bindGraphSet(int ar, int context) {
  const auto key = std::make_pair(ar, context);
  auto set = std::make_unique<GraphSet>();
  set->ar = ar;
  const size_t numParts = contexts_.size();
  for (size_t p = 0; p < numParts; ++p) {
    set->parts.push_back(&contexts_[p]->graph(graphName(ar, context, p + 1, numParts) + options_.graphSuffix));
  }
  std::map<std::string, bool> produced;
  auto bindTensor = [](const TensorMeta& meta, Buffer& buf) {
    Qnn_Tensor_t t = meta.prototype;
    t.v1.name = meta.name.c_str();
    t.v1.dimensions = const_cast<uint32_t*>(meta.dims.data());
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    t.v1.clientBuf.data = buf.data.data();
    t.v1.clientBuf.dataSize = static_cast<uint32_t>(buf.data.size());
    return t;
  };

  for (GraphInfo* graph : set->parts) {
    std::vector<Qnn_Tensor_t> ins, outs;
    for (const TensorMeta& meta : graph->inputs) {
      std::smatch match;
      Buffer* buf = nullptr;
      if (std::regex_match(meta.name, match, kKvIn)) {
        std::string base = match[1];
        auto& stream = streams_[base];
        if (!stream) {
          stream = std::make_unique<KvStream>();
          stream->base = base;
        }
        auto owned = std::make_unique<Buffer>();
        owned->meta = &meta;
        owned->data.assign(meta.bytes(), 0);
        buf = owned.get();
        set->buffers.emplace(meta.name, std::move(owned));
        stream->inMeta[key] = &meta;
        stream->inBuf[key] = buf;
      } else {
        const bool known = meta.name == "input_ids" || meta.name == "attention_mask" ||
                           meta.name == "position_ids_cos" || meta.name == "position_ids_sin";
        if (!known && !produced.count(meta.name)) {
          fail("graph " + graph->name + " input " + meta.name + " is not produced by an earlier part");
        }
        buf = &sharedBuffer(*set, meta);
        if (meta.name == "input_ids") set->tokens = buf;
        if (meta.name == "attention_mask") set->mask = buf;
        if (meta.name == "position_ids_cos") set->cos = buf;
        if (meta.name == "position_ids_sin") set->sin = buf;
      }
      ins.push_back(bindTensor(meta, *buf));
    }
    for (const TensorMeta& meta : graph->outputs) {
      std::smatch match;
      Buffer* buf = nullptr;
      if (std::regex_match(meta.name, match, kKvOut)) {
        std::string base = match[1];
        auto it = streams_.find(base);
        if (it == streams_.end()) fail("KV output " + meta.name + " has no matching input");
        auto owned = std::make_unique<Buffer>();
        owned->meta = &meta;
        owned->data.assign(meta.bytes(), 0);
        buf = owned.get();
        set->buffers.emplace(meta.name, std::move(owned));
        it->second->outMeta[key] = &meta;
        it->second->outBuf[key] = buf;
      } else {
        buf = &sharedBuffer(*set, meta);
        produced[meta.name] = true;
        if (meta.name == "logits") set->logits = buf;
      }
      outs.push_back(bindTensor(meta, *buf));
    }
    set->inputs.push_back(std::move(ins));
    set->outputs.push_back(std::move(outs));
  }
  if (!set->tokens || !set->logits) fail("graph set ar" + std::to_string(ar) + " lacks input_ids or logits");

  const size_t past = context - ar;
  for (auto& [base, stream] : streams_) {
    if (!stream->inMeta.count(key) || !stream->outMeta.count(key)) {
      fail("KV stream " + base + " is missing an input or output in graph set ar" + std::to_string(ar));
    }
    const TensorMeta& in = *stream->inMeta[key];
    const TensorMeta& out = *stream->outMeta[key];
    const int axis = tokenAxis(base, in.dims.size());
    if (in.dims[axis] != past || out.dims[axis] != static_cast<uint32_t>(ar)) {
      fail("KV stream " + base + " token axis does not match context/sequence length");
    }
    if (in.dataType != out.dataType || in.quantized != out.quantized || in.scale != out.scale ||
        in.offset != out.offset) {
      fail("KV stream " + base + " input/output quantization differs");
    }
    size_t outer = 1, inner = in.elementBytes();
    for (int d = 0; d < axis; ++d) outer *= in.dims[d];
    for (size_t d = axis + 1; d < in.dims.size(); ++d) inner *= in.dims[d];
    if (stream->outer == 0) {
      stream->axis = axis;
      stream->outer = outer;
      stream->inner = inner;
      stream->clear.resize(in.elementBytes());
      encodeScalar(in, 0.0, stream->clear.data());
      stream->dtype = dataTypeName(in.dataType);
      stream->store.assign(outer * options_.contextLength * inner, 0);
    } else if (stream->outer != outer || stream->inner != inner) {
      fail("KV stream " + base + " layout differs between sequence lengths");
    }
    const TensorMeta& first = *stream->inMeta.begin()->second;
    if (first.dataType != in.dataType || first.quantized != in.quantized ||
        first.scale != in.scale || first.offset != in.offset || first.dims.size() != in.dims.size()) {
      fail("KV stream " + base + " encoding differs between graph buckets");
    }
    for (size_t d = 0; d < in.dims.size(); ++d) {
      if (static_cast<int>(d) != axis && (in.dims[d] != first.dims[d] || in.dims[d] != out.dims[d])) {
        fail("KV stream " + base + " dimensions differ between graph buckets");
      }
    }
  }
  sets_[key] = std::move(set);
}

void LlmSession::reset() {
  for (auto& [base, stream] : streams_) {
    fillPattern(stream->store.data(), stream->store.size(), stream->clear);
  }
  cached_ = 0;
  lastAr_ = 0;
  lastContext_ = 0;
  lastNew_ = 0;
}

StepRecord LlmSession::step(const std::vector<int32_t>& tokens, int ar, bool profile) {
  auto stepStart = Clock::now();
  const int C = selectContext(options_.contextBuckets, ar, cached_);
  const auto key = std::make_pair(ar, C);
  auto setIt = sets_.find(key);
  if (setIt == sets_.end()) fail("no graphs for sequence length " + std::to_string(ar));
  GraphSet& set = *setIt->second;
  const int m = static_cast<int>(tokens.size());
  const int S = C - ar;
  const int n = static_cast<int>(cached_);
  if (m < 1 || m > ar) fail("step needs 1..ar new tokens");
  if (n > S) fail("cached tokens exceed the past capacity of the ar" + std::to_string(ar) + " graph");
  if (n + m > C) fail("context length exhausted; the KV cache does not slide");

  StepRecord record;
  record.ar = ar;
  record.graphContext = C;
  record.newTokens = m;
  record.cachedBefore = cached_;
  auto prepStart = Clock::now();

  auto* ids = reinterpret_cast<int32_t*>(set.tokens->data.data());
  for (int k = 0; k < ar; ++k) ids[k] = k < ar - m ? options_.padToken : tokens[k - (ar - m)];

  if (set.mask) {
    const TensorMeta& meta = *set.mask->meta;
    const size_t elem = meta.elementBytes();
    std::vector<uint8_t> attend(elem), blocked(elem);
    encodeScalar(meta, 0.0, attend.data());
    encodeScalar(meta, options_.maskMin, blocked.data());
    uint8_t* dst = set.mask->data.data();
    for (int i = 0; i < ar; ++i) {
      for (int j = 0; j < C; ++j) {
        const bool valid = (j >= S - n && j < S) || j >= C - m;
        const bool visible = valid && j <= S + i;
        std::memcpy(dst + (static_cast<size_t>(i) * C + j) * elem, visible ? attend.data() : blocked.data(), elem);
      }
    }
  }
  if (set.cos && set.sin) {
    const TensorMeta& cosMeta = *set.cos->meta;
    const TensorMeta& sinMeta = *set.sin->meta;
    const size_t elem = cosMeta.elementBytes();
    const size_t half = rope_.half;
    if (cosMeta.dims.back() != half || sinMeta.dims.back() != half) fail("RoPE width mismatch");
    for (int k = 0; k < ar; ++k) {
      int pos = k < ar - m ? std::max(n - 1, 0) : n + (k - (ar - m));
      pos = std::min(pos, options_.contextLength - 1);
      for (size_t f = 0; f < half; ++f) {
        encodeScalar(cosMeta, rope_.cos[pos * half + f], set.cos->data.data() + (k * half + f) * elem);
        encodeScalar(sinMeta, rope_.sin[pos * half + f], set.sin->data.data() + (k * half + f) * elem);
      }
    }
  }
  for (auto& [base, stream] : streams_) {
    uint8_t* dst = stream->inBuf[key]->data.data();
    const size_t rowIn = static_cast<size_t>(S) * stream->inner;
    const size_t emptyBytes = static_cast<size_t>(S - n) * stream->inner;
    for (size_t o = 0; o < stream->outer; ++o) {
      uint8_t* row = dst + o * rowIn;
      fillPattern(row, emptyBytes, stream->clear);
      std::memcpy(row + emptyBytes, stream->store.data() + o * options_.contextLength * stream->inner, n * stream->inner);
    }
    kvCopyBytes_ += stream->outer * static_cast<uint64_t>(n) * stream->inner;
  }
  record.prepareSeconds = since(prepStart);

  for (size_t p = 0; p < set.parts.size(); ++p) {
    auto partStart = Clock::now();
    if (profile) {
      record.partProfiles.push_back(rt_.executeProfiled(*set.parts[p], set.inputs[p], set.outputs[p]));
    } else {
      rt_.execute(*set.parts[p], set.inputs[p], set.outputs[p]);
    }
    record.partSeconds.push_back(since(partStart));
  }

  auto commitStart = Clock::now();
  for (auto& [base, stream] : streams_) {
    const uint8_t* src = stream->outBuf[key]->data.data();
    for (size_t o = 0; o < stream->outer; ++o) {
      std::memcpy(stream->store.data() + (o * options_.contextLength + n) * stream->inner,
                  src + (o * ar + (ar - m)) * stream->inner, m * stream->inner);
    }
    kvCopyBytes_ += stream->outer * static_cast<uint64_t>(m) * stream->inner;
  }
  cached_ += m;
  lastAr_ = ar;
  lastContext_ = C;
  lastNew_ = m;
  record.commitSeconds = since(commitStart);
  record.totalSeconds = since(stepStart);
  return record;
}

size_t LlmSession::vocabSize() const {
  const GraphSet& set = *sets_.begin()->second;
  return set.logits->meta->dims.back();
}

void LlmSession::logitsRow(int i, std::vector<float>& out) const {
  if (lastAr_ == 0 || i < 0 || i >= lastNew_) fail("logits row out of range");
  const GraphSet& set = *sets_.at({lastAr_, lastContext_});
  const TensorMeta& meta = *set.logits->meta;
  const size_t vocab = meta.dims.back();
  const size_t elem = meta.elementBytes();
  const size_t row = static_cast<size_t>(lastAr_ - lastNew_ + i);
  out.resize(vocab);
  const uint8_t* src = set.logits->data.data() + row * vocab * elem;
  for (size_t v = 0; v < vocab; ++v) out[v] = static_cast<float>(decodeScalar(meta, src + v * elem));
}

std::vector<KvStreamSummary> LlmSession::kvStreams() const {
  std::vector<KvStreamSummary> result;
  for (const auto& [base, stream] : streams_) {
    result.push_back({base, stream->dtype, stream->outer * stream->inner});
  }
  return result;
}

size_t LlmSession::kvStoreBytes() const {
  size_t total = 0;
  for (const auto& [base, stream] : streams_) total += stream->store.size();
  return total;
}

size_t LlmSession::ioBufferBytes() const {
  size_t total = 0;
  for (const auto& [ar, set] : sets_) {
    for (const auto& [name, buf] : set->buffers) total += buf->data.size();
  }
  return total;
}

}  // namespace tqrun
