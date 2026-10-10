// SPDX-License-Identifier: BSD-3-Clause
// Host Hexagon libnative test of packing, LUT layout, FP16 multiply and tails.
#include "hvx_decode.h"
#include <stdint.h>
#include <stdio.h>
#include <vector>

template<unsigned Bits> int test_bits() {
  __fp16 table[1u << Bits];
  for (unsigned i = 0; i < (1u << Bits); ++i)
    table[i] = (float(i) - ((1u << Bits) - 1) / 2.0f) / 128;
  for (unsigned rows : {1, 2, 3, 8, 127, 128, 255, 256, 1023}) {
    std::vector<uint8_t> packed(rows * 16 * Bits + 1);
    std::vector<__fp16> scales(rows);
    std::vector<uint16_t> output(rows * 128 + 2, 0x1234);
    for (unsigned i = 0; i < rows * 16 * Bits; ++i) packed[i + 1] = (i * 73 + i / 7) % 256;
    for (unsigned i = 0; i < rows; ++i) scales[i] = (i % 11) / 16.0f;
    tq_decode_bits_hvx<Bits>(packed.data() + 1, reinterpret_cast<uint16_t*>(scales.data()),
                            reinterpret_cast<uint16_t*>(table), output.data() + 1, rows);
    if (output.front() != 0x1234 || output.back() != 0x1234) return 2;
    for (unsigned i = 0; i < rows * 128; ++i) {
      unsigned index = 0;
      for (unsigned b = 0; b < Bits; ++b) {
        const unsigned bit = i * Bits + b;
        index = 2 * index + ((packed[1 + bit / 8] >> (7 - bit % 8)) & 1);
      }
      const __fp16 expected = float(table[index]) * float(scales[i / 128]);
      uint16_t value; memcpy(&value, &expected, 2);
      if (output[i + 1] != value) {
        printf("bits=%u rows=%u lane=%u code=%u actual=%04x expected=%04x\n",
               Bits, rows, i, index, output[i + 1], value);
        return 1;
      }
    }
  }
  return 0;
}

int main() {
  if (test_bits<2>() || test_bits<3>() || test_bits<5>() || test_bits<6>()) return 1;
  __fp16 table[16];
  for (unsigned i = 0; i < 16; ++i) table[i] = (float(i) - 7.5f) / 32;
  for (unsigned rows : {1, 2, 3, 8, 127, 128, 255, 256, 1023}) {
    std::vector<uint8_t> packed(rows * 64 + 1);
    std::vector<__fp16> scales(rows);
    std::vector<uint16_t> output(rows * 128 + 2, 0x1234);
    for (unsigned i = 0; i < rows * 64; ++i) packed[i + 1] = i % 256;
    for (unsigned i = 0; i < rows; ++i) scales[i] = (i % 11) / 16.0f;
    tq_decode_hvx(packed.data() + 1, reinterpret_cast<uint16_t*>(scales.data()),
                  reinterpret_cast<uint16_t*>(table), output.data() + 1, rows);
    if (output.front() != 0x1234 || output.back() != 0x1234) return 2;
    for (unsigned i = 0; i < rows * 128; ++i) {
      const unsigned byte = packed[i / 2 + 1];
      const unsigned index = i % 2 ? byte & 15 : byte >> 4;
      const __fp16 expected = float(table[index]) * float(scales[i / 128]);
      uint16_t bits;
      memcpy(&bits, &expected, 2);
      if (output[i + 1] != bits) {
        printf("rows=%u index=%u actual=%04x expected=%04x\n", rows, i, output[i + 1], bits);
        return 1;
      }
    }
  }
  puts("HVX packing, LUT lane order, FP16 products, unaligned I/O and odd tails passed");
}
