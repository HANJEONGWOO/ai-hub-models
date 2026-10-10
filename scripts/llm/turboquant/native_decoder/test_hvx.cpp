// SPDX-License-Identifier: BSD-3-Clause
// Host Hexagon libnative test of packing, LUT layout, FP16 multiply and tails.
#include "hvx_decode.h"
#include <stdint.h>
#include <stdio.h>
#include <vector>

#if defined(__unix__) && !defined(__hexagon__)
#include <sys/mman.h>
#include <unistd.h>
#define TQ_TEST_GUARD_PAGES 1
#endif

// Allocator padding can conceal vector reads outside a slice. Check each edge
// separately: an end-aligned load must not precede a short slice, and a full
// vector load must not extend beyond the final row.
#ifdef TQ_TEST_GUARD_PAGES
class GuardedBytes {
 public:
  GuardedBytes(size_t bytes, bool guard_after) {
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t readable = ((bytes + page - 1) / page) * page;
    size_ = readable + 2 * page;
    mapping_ = mmap(nullptr, size_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping_ == MAP_FAILED) return;
    auto* first = static_cast<uint8_t*>(mapping_) + page;
    if (mprotect(first, readable, PROT_READ | PROT_WRITE) != 0) return;
    data_ = guard_after ? first + readable - bytes : first;
  }
  ~GuardedBytes() {
    if (mapping_ != MAP_FAILED) munmap(mapping_, size_);
  }
  uint8_t* data() const { return data_; }

 private:
  void* mapping_ = MAP_FAILED;
  size_t size_ = 0;
  uint8_t* data_ = nullptr;
};
#endif

template<unsigned Bits>
int test_case(uint8_t* packed, unsigned rows, unsigned offset, const char* layout) {
  __fp16 table[1u << Bits];
  for (unsigned i = 0; i < (1u << Bits); ++i)
    table[i] = (float(i) - ((1u << Bits) - 1) / 2.0f) / 128;
  // Non-vector-aligned scales/output are deliberate. Sentinels cover zero rows
  // too, and ensure a slice cannot overwrite its neighbour's output.
  std::vector<__fp16> scales(rows + 1);
  std::vector<uint16_t> output(rows * 128 + 2, 0x1234);
  for (unsigned i = 0; i < rows * 16 * Bits; ++i)
    packed[i] = (i * 73 + i / 7) % 256;
  // Keep the first row nonzero so one-row/tail tests cannot hide bad indices.
  for (unsigned i = 0; i < rows; ++i)
    scales[i + 1] = i % 12 == 11 ? 0.0f : (i % 11 + 1) / 16.0f;
  if constexpr (Bits == 4)
    tq_decode_hvx(packed, reinterpret_cast<uint16_t*>(scales.data() + 1),
                  reinterpret_cast<uint16_t*>(table), output.data() + 1, rows);
  else
    tq_decode_bits_hvx<Bits>(packed, reinterpret_cast<uint16_t*>(scales.data() + 1),
                            reinterpret_cast<uint16_t*>(table), output.data() + 1, rows);
  if (output.front() != 0x1234 || output.back() != 0x1234) {
    printf("output sentinel overwritten: bits=%u rows=%u offset=%u layout=%s\n",
           Bits, rows, offset, layout);
    return 2;
  }
  for (unsigned i = 0; i < rows * 128; ++i) {
    unsigned index = 0;
    // Independent bit-at-a-time MSB-first reference, not the vector extractor.
    for (unsigned b = 0; b < Bits; ++b) {
      const unsigned bit = i * Bits + b;
      index = 2 * index + ((packed[bit / 8] >> (7 - bit % 8)) & 1);
    }
    const __fp16 expected = float(table[index]) * float(scales[i / 128 + 1]);
    uint16_t value;
    memcpy(&value, &expected, 2);
    if (output[i + 1] != value) {
      printf("bits=%u rows=%u offset=%u layout=%s lane=%u code=%u actual=%04x expected=%04x\n",
             Bits, rows, offset, layout, i, index, output[i + 1], value);
      return 1;
    }
  }
  return 0;
}

template<unsigned Bits> int test_bits() {
  // Straddle 128-byte loads, typical work slices and 256-token tile boundaries.
  // Small counts exercise all possible final-row load shapes.
  for (unsigned rows : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                        63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513,
                        1023, 1024}) {
    std::vector<uint8_t> packed(rows * 16 * Bits + 1);
    if (test_case<Bits>(packed.data() + 1, rows, 1, "heap")) return 1;
  }
  for (unsigned offset : {0, 3, 15, 31, 63, 127}) {
    for (unsigned rows : {1, 2, 3, 5, 7, 8, 9, 17}) {
      std::vector<uint8_t> packed(rows * 16 * Bits + offset + 1);
      if (test_case<Bits>(packed.data() + offset, rows, offset, "heap")) return 1;
    }
  }
#ifdef TQ_TEST_GUARD_PAGES
  for (bool guard_after : {false, true}) {
    for (unsigned rows : {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 63, 64, 65,
                          127, 128, 129, 255, 256, 257}) {
      GuardedBytes packed(rows * 16 * Bits, guard_after);
      if (!packed.data()) {
        perror("mmap/mprotect for exact-size packed slice");
        return 3;
      }
      // Flush the exact failing shape before a guard-page violation can abort.
      printf("guard check: bits=%u rows=%u edge=%s\n", Bits, rows,
             guard_after ? "end" : "start");
      fflush(stdout);
      if (test_case<Bits>(packed.data(), rows, 0,
                          guard_after ? "guard-end" : "guard-start")) return 1;
    }
  }
#endif
  return 0;
}

int main() {
  if (test_bits<2>() || test_bits<3>() || test_bits<4>() || test_bits<5>() || test_bits<6>())
    return 1;
  puts("HVX packing, LUT lane order, exact FP16 products, unaligned I/O, slice tails and guard pages passed");
}
