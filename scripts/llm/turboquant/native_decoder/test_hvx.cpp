// SPDX-License-Identifier: BSD-3-Clause
// Host Hexagon libnative test of packing, LUT layout, FP16 multiply and tails.
#include "hvx_decode.h"
#include <stdint.h>
#include <stdio.h>
#include <vector>

int main() {
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
