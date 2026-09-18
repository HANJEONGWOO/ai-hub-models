// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <string.h>

// vlut16 selects halfwords from 16 word slots. Its two outputs contain
// even/odd input lanes; interleave halfwords to restore contiguous KV order.
inline void tq_store_row(HVX_Vector indices, HVX_Vector lut, uint16_t scale,
                         uint16_t* output) {
  const auto selected = Q6_Wh_vlut16_VbVhI(indices, lut, 0);
  const auto ordered = Q6_W_vshuff_VVR(Q6_V_hi_W(selected), Q6_V_lo_W(selected), -2);
  const auto factor = Q6_Vh_vsplat_R(scale);
  const auto first = Q6_Vhf_vmpy_VhfVhf(Q6_V_lo_W(ordered), factor);
  const auto second = Q6_Vhf_vmpy_VhfVhf(Q6_V_hi_W(ordered), factor);
  memcpy(output, &first, 128);
  memcpy(output + 64, &second, 128);
}

inline void tq_decode_hvx(const uint8_t* packed, const uint16_t* scales,
                           const uint16_t* centroids, uint16_t* output,
                           uint32_t rows) {
  alignas(128) uint16_t table[64];
  for (unsigned i = 0; i < 64; ++i) table[i] = centroids[(i / 2) % 16];
  HVX_Vector lut;
  memcpy(&lut, table, 128);
  const auto mask = Q6_Vb_vsplat_R(15);
  for (uint32_t row = 0; row < rows; row += 2) {
    auto bytes = Q6_V_vzero();
    // memcpy avoids alignment assumptions and never reads beyond an odd tail.
    if (row + 1 < rows) memcpy(&bytes, packed + row * 64, 128);
    else memcpy(&bytes, packed + row * 64, 64);
    const auto low = Q6_V_vand_VV(bytes, mask);
    const auto high = Q6_Vub_vlsr_VubR(bytes, 4);
    const auto indices = Q6_W_vshuff_VVR(low, high, -1);
    tq_store_row(Q6_V_lo_W(indices), lut, scales[row], output + row * 128);
    if (row + 1 < rows)
      tq_store_row(Q6_V_hi_W(indices), lut, scales[row + 1], output + (row + 1) * 128);
  }
}
