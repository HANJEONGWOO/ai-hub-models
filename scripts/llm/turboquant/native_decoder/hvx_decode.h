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

// Tight MSB-first 2/3/5/6-bit streams. Byte gathers, variable halfword shifts,
// multi-bank centroid lookup and FP16 products all execute on HVX. Full vector
// loads avoid scalarizing a partial-row memcpy into one byte load per byte.
// Bounds are the current QHPI slice, not allocator padding or another slice.
template<unsigned Bits>
inline void tq_decode_bits_hvx(const uint8_t* packed, const uint16_t* scales,
                              const uint16_t* centroids, uint16_t* output,
                              uint32_t rows) {
  static_assert(Bits >= 2 && Bits <= 6);
  if (!rows) return;
  alignas(128) uint8_t offsets[128], next[128];
  alignas(128) uint16_t shifts[128], table[64];
  for (unsigned j = 0; j < 128; ++j) {
    offsets[j] = j * Bits / 8;
    next[j] = offsets[j] + 1;
    shifts[j] = 16 - Bits - (j * Bits % 8);
  }
  for (unsigned j = 0; j < 64; ++j) {
    const unsigned index = j / 2 + (j % 2) * 32;
    table[j] = index < (1u << Bits) ? centroids[index] : 0;
  }
  HVX_Vector off, nxt, shift0, shift1, lut;
  memcpy(&off, offsets, 128); memcpy(&nxt, next, 128);
  memcpy(&shift0, shifts, 128); memcpy(&shift1, shifts + 64, 128);
  memcpy(&lut, table, 128);
  const auto mask = Q6_Vh_vsplat_R((1u << Bits) - 1);
  constexpr uint32_t row_bytes = 16 * Bits;
  const uint32_t byte_count = rows * row_bytes;
  const uint32_t tail_start = byte_count >= 128 ? byte_count - 128 : 0;
  HVX_Vector tail;
  if (byte_count >= 128) {
    // End-anchored vector fits entirely in this slice. Final row(s) rotate
    // their valid bytes to lane zero; unused lanes cannot affect masked codes.
    memcpy(&tail, packed + tail_start, 128);
  } else {
    // Only very small slices (e.g. current KV) need a partial copy. Stage it
    // once per invocation, rather than scalarizing every historical KV row.
    alignas(128) uint8_t scratch[128] = {};
    memcpy(scratch, packed, byte_count);
    memcpy(&tail, scratch, 128);
  }
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t offset = row * row_bytes;
    HVX_Vector bytes;
    if (offset + 128 <= byte_count) memcpy(&bytes, packed + offset, 128);
    else bytes = Q6_V_vror_VR(tail, offset - tail_start);
    // vlut32 tables interleave entries 0..63 with entries 64..127.
    const auto byte_lut = Q6_Vb_vshuff_Vb(bytes);
    auto first = Q6_Vb_vlut32_VbVbI(off, byte_lut, 0);
    auto second = Q6_Vb_vlut32_VbVbI(nxt, byte_lut, 0);
    if constexpr (Bits >= 3) {
      first = Q6_Vb_vlut32or_VbVbVbI(first, off, byte_lut, 1);
      second = Q6_Vb_vlut32or_VbVbVbI(second, nxt, byte_lut, 1);
    }
    if constexpr (Bits >= 5) {
      first = Q6_Vb_vlut32or_VbVbVbI(first, off, byte_lut, 2);
      second = Q6_Vb_vlut32or_VbVbVbI(second, nxt, byte_lut, 2);
    }
    const auto words = Q6_W_vshuff_VVR(first, second, -1);
    const auto lo = Q6_V_vand_VV(Q6_Vh_vlsr_VhVh(Q6_V_lo_W(words), shift0), mask);
    const auto hi = Q6_V_vand_VV(Q6_Vh_vlsr_VhVh(Q6_V_hi_W(words), shift1), mask);
    const auto indices = Q6_Vb_vpacke_VhVh(hi, lo);
    auto selected = Q6_Wh_vlut16_VbVhI(indices, lut, 0);
    if constexpr (Bits >= 5)
      selected = Q6_Wh_vlut16or_WhVbVhI(selected, indices, lut, 1);
    if constexpr (Bits == 6) {
      selected = Q6_Wh_vlut16or_WhVbVhI(selected, indices, lut, 2);
      selected = Q6_Wh_vlut16or_WhVbVhI(selected, indices, lut, 3);
    }
    const auto ordered = Q6_W_vshuff_VVR(Q6_V_hi_W(selected), Q6_V_lo_W(selected), -2);
    const auto factor = Q6_Vh_vsplat_R(scales[row]);
    const auto out0 = Q6_Vhf_vmpy_VhfVhf(Q6_V_lo_W(ordered), factor);
    const auto out1 = Q6_Vhf_vmpy_VhfVhf(Q6_V_hi_W(ordered), factor);
    memcpy(output + row * 128, &out0, 128);
    memcpy(output + row * 128 + 64, &out1, 128);
  }
}
