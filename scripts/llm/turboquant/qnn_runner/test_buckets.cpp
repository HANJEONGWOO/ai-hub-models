// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
#include <cassert>
#include <stdexcept>
#include <vector>

#include "src/buckets.h"

int main() {
  const auto b = tqrun::contextBuckets({512, 128, 256, 128}, 1024);
  assert((b == std::vector<int>{128, 256, 512, 1024}));
  assert(tqrun::selectContext(b, 1, 0) == 128);
  assert(tqrun::selectContext(b, 1, 127) == 128);
  assert(tqrun::selectContext(b, 1, 128) == 256);
  assert(tqrun::selectContext(b, 1, 255) == 256);
  assert(tqrun::selectContext(b, 1, 256) == 512);
  assert(tqrun::selectContext(b, 1, 511) == 512);
  assert(tqrun::selectContext(b, 1, 512) == 1024);
  assert(tqrun::selectContext(b, 1, 1023) == 1024);
  for (int n = 0; n < 1024; ++n) {
    const int c = tqrun::selectContext(b, 1, n);
    assert(c - 1 >= n);
  }
  assert(tqrun::selectContext(b, 128, 0) == 256);
  assert(tqrun::selectContext(b, 128, 128) == 256);
  assert(tqrun::selectContext(b, 128, 129) == 512);
  assert(tqrun::selectContext(b, 128, 384) == 512);
  assert(tqrun::selectContext(b, 128, 385) == 1024);
  assert(tqrun::selectContext(b, 128, 896) == 1024);
  for (auto [ar, n] : std::vector<std::pair<int, int>>{{1, 1024}, {128, 897}, {0, 0}}) {
    bool threw = false;
    try { tqrun::selectContext(b, ar, n); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
  }
  for (int c : {0, 1, 1025}) {
    bool threw = false;
    try { tqrun::contextBuckets({c}, 1024); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
  }
  assert(tqrun::selectContext(tqrun::contextBuckets({}, 1024), 1, 35) == 1024);
}
