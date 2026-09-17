// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
// qnn-llm-runner: prefill + greedy decode, or teacher-forced scoring, for split
// delta-KV LLM context binaries on the QNN HTP backend.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "qnn_api.h"
#include "session.h"

namespace {

using tqrun::LlmSession;
using tqrun::StepRecord;
using Clock = std::chrono::steady_clock;

struct Args {
  std::string backend = "libQnnHtp.so";
  std::string system = "libQnnSystem.so";
  std::vector<std::string> bins;
  int contextLength = 1024;
  std::vector<int> contextBuckets;
  std::string rope;
  int ropeHalf = 64;
  std::string mode = "generate";
  std::string tokens;
  int nGen = 128;
  bool stopOnEos = false;
  std::vector<int32_t> eos{151645, 151643};
  std::string report = "report.json";
  std::string dumpLogits;
  std::string graphSuffix;
  int profileDecodeStep = -1;
  bool profilePrefill = false;
  int sessions = 1;
  bool burst = true;
};

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

Args parseArgs(int argc, char** argv) {
  Args a;
  std::map<std::string, std::string> kv;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--stop-on-eos") {
      a.stopOnEos = true;
    } else if (key == "--no-burst") {
      a.burst = false;
    } else if (key == "--profile-prefill") {
      a.profilePrefill = true;
    } else if (key.rfind("--", 0) == 0 && i + 1 < argc) {
      kv[key] = argv[++i];
    } else {
      throw std::runtime_error("unexpected argument " + key);
    }
  }
  auto get = [&](const char* k, const std::string& def) { return kv.count(k) ? kv[k] : def; };
  a.backend = get("--backend", a.backend);
  a.system = get("--system", a.system);
  a.bins = split(get("--bins", ""), ',');
  a.contextLength = std::stoi(get("--context-length", std::to_string(a.contextLength)));
  for (const auto& c : split(get("--context-buckets", ""), ',')) a.contextBuckets.push_back(std::stoi(c));
  a.rope = get("--rope", "");
  a.ropeHalf = std::stoi(get("--rope-half", std::to_string(a.ropeHalf)));
  a.mode = get("--mode", a.mode);
  a.tokens = get("--tokens", "");
  a.nGen = std::stoi(get("--n-gen", std::to_string(a.nGen)));
  a.report = get("--report", a.report);
  a.dumpLogits = get("--dump-logits", "");
  a.graphSuffix = get("--graph-suffix", "");
  a.profileDecodeStep = std::stoi(get("--profile-decode-step", "-1"));
  a.sessions = std::max(1, std::stoi(get("--sessions", "1")));
  if (kv.count("--eos")) {
    a.eos.clear();
    for (const auto& t : split(kv["--eos"], ',')) a.eos.push_back(std::stoi(t));
  }
  if (a.bins.empty() || a.rope.empty() || a.tokens.empty()) {
    throw std::runtime_error("usage: --bins p1,p2,... --rope rope.bin --tokens ids.bin [--mode generate|score]");
  }
  if (a.mode != "generate" && a.mode != "score") throw std::runtime_error("--mode must be generate or score");
  return a;
}

template <typename T>
std::vector<T> readVector(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const auto size = static_cast<size_t>(f.tellg());
  if (size % sizeof(T)) throw std::runtime_error(path + " size is not a multiple of the element size");
  std::vector<T> data(size / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
  return data;
}

tqrun::RopeTable loadRope(const std::string& path, int half) {
  std::vector<float> raw = readVector<float>(path);
  if (raw.size() % (2 * half)) throw std::runtime_error("RoPE file must hold cos then sin, [positions, half] each");
  tqrun::RopeTable table;
  table.half = half;
  table.positions = raw.size() / (2 * half);
  table.cos.assign(raw.begin(), raw.begin() + raw.size() / 2);
  table.sin.assign(raw.begin() + raw.size() / 2, raw.end());
  return table;
}

std::map<std::string, long> readProcStatus() {
  std::map<std::string, long> out;
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    for (const char* key : {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"}) {
      if (line.rfind(std::string(key) + ":", 0) == 0) out[key] = std::atol(line.c_str() + std::strlen(key) + 1);
    }
  }
  return out;
}

// Chunks the way HubCompatibleGenerator does: a partial chunk first, then full chunks.
std::vector<std::pair<size_t, size_t>> planChunks(size_t count, int ar) {
  std::vector<std::pair<size_t, size_t>> chunks;
  size_t start = 0;
  const size_t rem = count % ar;
  if (rem) {
    chunks.emplace_back(0, rem);
    start = rem;
  }
  for (; start < count; start += ar) chunks.emplace_back(start, ar);
  return chunks;
}

double logSoftmaxAt(const std::vector<float>& logits, int32_t index) {
  const float maxLogit = *std::max_element(logits.begin(), logits.end());
  double sum = 0.0;
  for (float v : logits) sum += std::exp(static_cast<double>(v) - maxLogit);
  return static_cast<double>(logits[index]) - maxLogit - std::log(sum);
}

int32_t argmax(const std::vector<float>& logits) {
  return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

class Json {
 public:
  void raw(const std::string& key, const std::string& value) { fields_.push_back("\"" + key + "\": " + value); }
  void str(const std::string& key, const std::string& value) { raw(key, "\"" + escape(value) + "\""); }
  void num(const std::string& key, double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", value);
    raw(key, buf);
  }
  std::string dump() const {
    std::string s = "{";
    for (size_t i = 0; i < fields_.size(); ++i) s += (i ? ", " : "") + fields_[i];
    return s + "}";
  }
  static std::string escape(const std::string& v) {
    std::string out;
    for (char c : v) {
      if (c == '"' || c == '\\') out += '\\';
      out += c;
    }
    return out;
  }

 private:
  std::vector<std::string> fields_;
};

std::string numberList(const std::vector<double>& values) {
  std::string s = "[";
  char buf[64];
  for (size_t i = 0; i < values.size(); ++i) {
    std::snprintf(buf, sizeof(buf), "%s%.9g", i ? ", " : "", values[i]);
    s += buf;
  }
  return s + "]";
}

std::string stepJson(const StepRecord& r, const char* kind) {
  Json j;
  j.str("kind", kind);
  j.num("ar", r.ar);
  j.num("graph_context", r.graphContext);
  j.num("new_tokens", r.newTokens);
  j.num("cached_before", static_cast<double>(r.cachedBefore));
  j.num("prepare_s", r.prepareSeconds);
  j.num("commit_s", r.commitSeconds);
  j.raw("part_s", numberList(r.partSeconds));
  j.num("total_s", r.totalSeconds);
  return j.dump();
}

// Per-part summary of a detailed QNN profile: all op cycle events plus the TurboQuant codec ops.
std::string profileJson(const StepRecord& r, const std::string& label) {
  std::string parts = "[";
  for (size_t p = 0; p < r.partProfiles.size(); ++p) {
    size_t cycleEvents = 0, codecEvents = 0, codecZero = 0;
    double cycles = 0.0, codecCycles = 0.0;
    std::string timings = "[", codecList = "[";
    for (const auto& e : r.partProfiles[p]) {
      if (e.unit == QNN_PROFILE_EVENTUNIT_CYCLES && e.depth > 0) {
        ++cycleEvents;
        cycles += static_cast<double>(e.value);
        if (e.identifier.rfind("tq_", 0) == 0) {
          ++codecEvents;
          codecCycles += static_cast<double>(e.value);
          codecZero += e.value == 0;
          Json c;
          c.str("op", e.identifier);
          c.num("cycles", static_cast<double>(e.value));
          codecList += (codecEvents > 1 ? ", " : "") + c.dump();
        }
      } else if (e.unit == QNN_PROFILE_EVENTUNIT_MICROSEC && e.depth <= 1) {
        Json t;
        t.str("event", e.identifier);
        t.num("us", static_cast<double>(e.value));
        timings += (timings.size() > 1 ? ", " : "") + t.dump();
      }
    }
    Json part;
    part.num("part", static_cast<double>(p + 1));
    part.num("events", static_cast<double>(r.partProfiles[p].size()));
    part.num("op_cycle_events", static_cast<double>(cycleEvents));
    part.num("op_cycles_sum", cycles);
    part.num("codec_op_events", static_cast<double>(codecEvents));
    part.num("codec_op_zero_cycle_events", static_cast<double>(codecZero));
    part.num("codec_op_cycles_sum", codecCycles);
    part.raw("timings", timings + "]");
    part.raw("codec_ops", codecList + "]");
    parts += (p ? ", " : "") + part.dump();
  }
  Json j;
  j.str("step", label);
  j.raw("parts", parts + "]");
  return j.dump();
}

std::string memJson(const std::map<std::string, long>& m) {
  Json j;
  for (const auto& [k, v] : m) j.num(k + "_kb", static_cast<double>(v));
  return j.dump();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parseArgs(argc, argv);
    auto processStart = Clock::now();
    std::vector<std::string> memSamples;
    tqrun::RopeTable rope = loadRope(args.rope, args.ropeHalf);
    std::vector<int32_t> tokens = readVector<int32_t>(args.tokens);
    if (tokens.empty()) throw std::runtime_error("token file is empty");

    tqrun::QnnRuntime rt(args.backend, args.system);
    memSamples.push_back("{\"at\": \"runtime\", \"mem\": " + memJson(readProcStatus()) + "}");
    const bool burst = args.burst && rt.setBurstPower();
    if (args.profilePrefill || args.profileDecodeStep >= 0) rt.enableDetailedProfiling();

    tqrun::SessionOptions options;
    options.bins = args.bins;
    options.contextLength = args.contextLength;
    options.graphSuffix = args.graphSuffix;
    options.contextBuckets = args.contextBuckets;
    LlmSession session(rt, options, rope);
    memSamples.push_back("{\"at\": \"contexts_loaded\", \"mem\": " + memJson(readProcStatus()) + "}");

    const int C = args.contextLength;
    const size_t promptCount = tokens.size();
    if (args.mode == "generate" && promptCount + args.nGen - 1 > static_cast<size_t>(C)) {
      throw std::runtime_error("prompt + n_gen - 1 exceeds the context length");
    }
    if (args.mode == "score" && promptCount > static_cast<size_t>(C)) {
      throw std::runtime_error("score sequence exceeds the context length");
    }
    const int prefillAr = promptCount == 1 ? 1 : 128;
    const auto chunks = planChunks(promptCount, prefillAr);
    FILE* dump = args.dumpLogits.empty() ? nullptr : std::fopen(args.dumpLogits.c_str(), "wb");
    auto isEos = [&](int32_t t) { return std::find(args.eos.begin(), args.eos.end(), t) != args.eos.end(); };

    std::vector<std::string> steps, profiles, sessionSummaries;
    std::vector<int32_t> generated;
    std::vector<double> decodeTimes;
    std::vector<float> row;
    double nllSum = 0.0, ttft = 0.0, prefillSeconds = 0.0;
    size_t scored = 0, top1Match = 0;
    std::string stopReason = "length";
    Clock::time_point prefillStart;
    // Later sessions reuse the loaded graphs after reset(); identical outputs show no state leaks.
    for (int s = 0; s < args.sessions; ++s) {
      const bool first = s == 0;
      if (!first) session.reset();
      steps.clear();
      generated.clear();
      decodeTimes.clear();
      nllSum = 0.0;
      scored = top1Match = 0;
      stopReason = "length";
      int32_t next = -1;
      prefillStart = Clock::now();
      for (const auto& [start, count] : chunks) {
        std::vector<int32_t> chunk(tokens.begin() + start, tokens.begin() + start + count);
        const bool profiled = first && args.profilePrefill && start == chunks.back().first;
        StepRecord rec = session.step(chunk, prefillAr, profiled);
        steps.push_back(stepJson(rec, profiled ? "prefill_profiled" : "prefill"));
        if (profiled) profiles.push_back(profileJson(rec, "prefill_last_chunk"));
        if (args.mode == "score") {
          for (size_t i = 0; i < count; ++i) {
            const size_t target = start + i + 1;
            if (target >= promptCount) break;
            session.logitsRow(static_cast<int>(i), row);
            nllSum -= logSoftmaxAt(row, tokens[target]);
            top1Match += argmax(row) == tokens[target];
            ++scored;
          }
        }
      }
      session.logitsRow(static_cast<int>(chunks.back().second) - 1, row);
      prefillSeconds = std::chrono::duration<double>(Clock::now() - prefillStart).count();
      if (first) memSamples.push_back("{\"at\": \"prefill_done\", \"mem\": " + memJson(readProcStatus()) + "}");

      if (args.mode == "generate") {
        next = argmax(row);
        ttft = std::chrono::duration<double>(Clock::now() - prefillStart).count();
        generated.push_back(next);
        if (dump && first) std::fwrite(row.data(), sizeof(float), row.size(), dump);
        while (static_cast<int>(generated.size()) < args.nGen) {
          if (args.stopOnEos && isEos(next)) {
            stopReason = "eos";
            break;
          }
          auto t0 = Clock::now();
          const bool profiled = first && static_cast<int>(generated.size()) - 1 == args.profileDecodeStep;
          StepRecord rec = session.step({next}, 1, profiled);
          session.logitsRow(0, row);
          next = argmax(row);
          if (profiled) {
            profiles.push_back(profileJson(rec, "decode_" + std::to_string(args.profileDecodeStep)));
          } else {
            decodeTimes.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
          }
          steps.push_back(stepJson(rec, profiled ? "decode_profiled" : "decode"));
          generated.push_back(next);
          if (dump && first) std::fwrite(row.data(), sizeof(float), row.size(), dump);
        }
      }
      Json summary;
      summary.num("session", s);
      summary.num("cached_tokens_at_end", static_cast<double>(session.cached()));
      if (args.mode == "generate") {
        std::string ids = "[";
        for (size_t i = 0; i < generated.size(); ++i) ids += (i ? ", " : "") + std::to_string(generated[i]);
        summary.raw("generated", ids + "]");
        summary.str("stop_reason", stopReason);
        summary.num("ttft_s", ttft);
        summary.num("prefill_tok_per_s", promptCount / prefillSeconds);
        if (!decodeTimes.empty()) {
          double sum = 0.0;
          for (double t : decodeTimes) sum += t;
          summary.num("decode_tok_per_s", decodeTimes.size() / sum);
          summary.num("decode_ms_per_token", 1000.0 * sum / decodeTimes.size());
        }
      } else {
        summary.num("nll_sum", nllSum);
        summary.num("scored_tokens", static_cast<double>(scored));
      }
      sessionSummaries.push_back(summary.dump());
    }
    if (dump) std::fclose(dump);
    memSamples.push_back("{\"at\": \"end\", \"mem\": " + memJson(readProcStatus()) + "}");

    Json report;
    report.str("mode", args.mode);
    report.num("context_length", C);
    report.num("prompt_tokens", static_cast<double>(promptCount));
    report.str("backend_api", rt.backendVersion());
    report.raw("burst_power", burst ? "true" : "false");
    std::string bins = "[";
    for (size_t i = 0; i < session.contexts().size(); ++i) {
      const auto& c = session.contexts()[i];
      Json b;
      b.str("path", c->path());
      b.num("bytes", static_cast<double>(c->fileBytes()));
      b.num("load_s", c->loadSeconds());
      bins += (i ? ", " : "") + b.dump();
    }
    report.raw("bins", bins + "]");
    report.num("load_s", session.loadSeconds());
    report.num("process_to_ready_s",
               std::chrono::duration<double>(prefillStart - processStart).count());
    report.num("prefill_s", prefillSeconds);
    if (args.mode == "generate") {
      report.num("ttft_s", ttft);
      std::string ids = "[";
      for (size_t i = 0; i < generated.size(); ++i) ids += (i ? ", " : "") + std::to_string(generated[i]);
      report.raw("generated", ids + "]");
      report.str("stop_reason", stopReason);
      report.raw("decode_step_s", numberList(decodeTimes));
      if (!decodeTimes.empty()) {
        double sum = 0.0;
        for (double t : decodeTimes) sum += t;
        report.num("decode_tok_per_s", decodeTimes.size() / sum);
      }
      report.num("prefill_tok_per_s", promptCount / prefillSeconds);
    } else {
      report.num("scored_tokens", static_cast<double>(scored));
      report.num("nll_sum", nllSum);
      report.num("ppl", std::exp(nllSum / std::max<size_t>(scored, 1)));
      report.num("top1_self_consistency", static_cast<double>(top1Match) / std::max<size_t>(scored, 1));
    }
    std::string streams = "[";
    const auto summaries = session.kvStreams();
    for (size_t i = 0; i < summaries.size(); ++i) {
      Json s;
      s.str("name", summaries[i].name);
      s.str("dtype", summaries[i].dtype);
      s.num("bytes_per_token", static_cast<double>(summaries[i].bytesPerToken));
      streams += (i ? ", " : "") + s.dump();
    }
    report.raw("kv_streams", streams + "]");
    report.num("kv_store_bytes", static_cast<double>(session.kvStoreBytes()));
    report.num("io_buffer_bytes", static_cast<double>(session.ioBufferBytes()));
    report.num("kv_host_copy_bytes", static_cast<double>(session.kvCopyBytes()));
    std::string mem = "[";
    for (size_t i = 0; i < memSamples.size(); ++i) mem += (i ? ", " : "") + memSamples[i];
    report.raw("memory", mem + "]");
    std::string stepList = "[";
    for (size_t i = 0; i < steps.size(); ++i) stepList += (i ? ", " : "") + steps[i];
    report.raw("steps", stepList + "]");
    std::string profileList = "[";
    for (size_t i = 0; i < profiles.size(); ++i) profileList += (i ? ", " : "") + profiles[i];
    report.raw("op_profiles", profileList + "]");
    std::string sessionList = "[";
    for (size_t i = 0; i < sessionSummaries.size(); ++i) sessionList += (i ? ", " : "") + sessionSummaries[i];
    report.raw("sessions", sessionList + "]");

    std::ofstream out(args.report);
    out << report.dump() << "\n";
    std::printf("wrote %s\n", args.report.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
