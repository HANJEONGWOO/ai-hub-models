// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tqrun {

inline std::vector<int> contextBuckets(std::vector<int> buckets, int capacity) {
  if (capacity <= 1) throw std::runtime_error("context length must exceed one");
  buckets.push_back(capacity);
  std::sort(buckets.begin(), buckets.end());
  buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());
  for (int c : buckets) {
    if (c <= 1 || c > capacity) throw std::runtime_error("invalid context bucket");
  }
  return buckets;
}

inline int selectContext(const std::vector<int>& buckets, int ar, std::size_t cached) {
  if (ar <= 0) throw std::runtime_error("invalid sequence length");
  for (int c : buckets) {
    if (c > ar && cached <= static_cast<std::size_t>(c - ar)) return c;
  }
  throw std::runtime_error("no context bucket can hold the cache and sequence");
}

}  // namespace tqrun
