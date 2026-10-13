// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <string.h>

// Structured K8 uses the same reflected byte ABI as the symmetric Decode8
// LUT. Constants are [offset0, step0, offset1, step1] from structured_k8.py.
// Each intrinsic rounds to IEEE FP16 separately: never fuse product and add.
// One 128-byte input load produces two 128-byte output vectors. No LUT loads,
// centroid scratch table, variable gather or scalar coordinate loop.
inline void tq_decode_structured8_hvx(const uint8_t* packed, const uint16_t* scales,
                                     const uint16_t* coefficients, uint16_t* output,
                                     uint32_t rows) {
  if (!rows) return;
  const auto offset0 = Q6_Vh_vsplat_R(coefficients[0]);
  const auto step0 = Q6_Vh_vsplat_R(coefficients[1]);
  const auto offset1 = Q6_Vh_vsplat_R(coefficients[2]);
  const auto step1 = Q6_Vh_vsplat_R(coefficients[3]);
  const auto mask = Q6_Vh_vsplat_R(63);
  const auto sign_bit = Q6_Vh_vsplat_R(128);
  const auto half = Q6_Vb_vsplat_R(127);
  for (uint32_t row = 0; row < rows; ++row) {
    HVX_Vector bytes;
    memcpy(&bytes, packed + row * 128, 128);
    const auto negative = Q6_Q_vcmp_gt_VubVub(Q6_Vb_vsplat_R(128), bytes);
    const auto reflected = Q6_Vb_vsub_VbVb(half, bytes);
    const auto magnitude = Q6_V_vand_VV(Q6_V_vmux_QVV(negative, reflected, bytes), half);
    const auto wide = Q6_W_vshuff_VVR(Q6_V_vzero(), magnitude, -1);
    const auto codes = Q6_W_vshuff_VVR(Q6_V_vzero(), bytes, -1);
    const auto factor = Q6_Vh_vsplat_R(scales[row]);
    const auto decode_half = [&](HVX_Vector mag, HVX_Vector code) {
      const auto tail = Q6_Q_vcmp_gt_VuhVuh(mag, mask);
      const auto step = Q6_V_vmux_QVV(tail, step1, step0);
      const auto offset = Q6_V_vmux_QVV(tail, offset1, offset0);
      const auto local = Q6_Vhf_vcvt_Vuh(Q6_V_vand_VV(mag, mask));
      const auto product = Q6_Vhf_vmpy_VhfVhf(local, step);
      const auto centroid = Q6_Vhf_vadd_VhfVhf(product, offset);
      const auto sign = Q6_Vh_vasl_VhR(Q6_V_vxor_VV(Q6_V_vand_VV(code, sign_bit), sign_bit), 8);
      return Q6_Vhf_vmpy_VhfVhf(Q6_V_vxor_VV(centroid, sign), factor);
    };
    const auto first = decode_half(Q6_V_lo_W(wide), Q6_V_lo_W(codes));
    const auto second = decode_half(Q6_V_hi_W(wide), Q6_V_hi_W(codes));
    memcpy(output + row * 128, &first, 128);
    memcpy(output + row * 128 + 64, &second, 128);
  }
}

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
// Large Lloyd-Max tables: keep raw UINT8 indices, select 64-entry HVX banks.
// No FP32 decoder or affine INT8 interpretation. Products round once to FP16.
template<unsigned Bits, bool Symmetric = false>
inline void tq_store_large_row(HVX_Vector indices, const HVX_Vector* tables,
                               uint16_t scale, uint16_t* output,
                               HVX_Vector original = Q6_V_vzero()) {
  const auto local = Q6_V_vand_VV(indices, Q6_Vb_vsplat_R(63));
  const auto wide = Q6_W_vshuff_VVR(Q6_V_vzero(), indices, -1);
  const auto mask = Q6_Vh_vsplat_R(192);
  auto out0 = Q6_V_vzero(), out1 = Q6_V_vzero();
  for (unsigned bank = 0; bank < (1u << (Bits - 6)); ++bank) {
    auto selected = Q6_Wh_vlut16_VbVhI(local, tables[bank], 0);
    selected = Q6_Wh_vlut16or_WhVbVhI(selected, local, tables[bank], 1);
    selected = Q6_Wh_vlut16or_WhVbVhI(selected, local, tables[bank], 2);
    selected = Q6_Wh_vlut16or_WhVbVhI(selected, local, tables[bank], 3);
    const auto ordered = Q6_W_vshuff_VVR(Q6_V_hi_W(selected), Q6_V_lo_W(selected), -2);
    const auto tag = Q6_Vh_vsplat_R(bank * 64);
    out0 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(Q6_V_lo_W(wide), mask), tag), Q6_V_lo_W(ordered), out0);
    out1 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VhVh(Q6_V_vand_VV(Q6_V_hi_W(wide), mask), tag), Q6_V_hi_W(ordered), out1);
  }
  const auto factor = Q6_Vh_vsplat_R(scale);
  out0 = Q6_Vhf_vmpy_VhfVhf(out0, factor);
  out1 = Q6_Vhf_vmpy_VhfVhf(out1, factor);
  if constexpr (Symmetric) {
    // Positive-half LUT plus an exact FP16 sign flip (including signed zero).
    const auto codes = Q6_W_vshuff_VVR(Q6_V_vzero(), original, -1);
    const auto sign = Q6_Vh_vsplat_R(128);
    const auto neg0 = Q6_Vh_vasl_VhR(Q6_V_vxor_VV(Q6_V_vand_VV(Q6_V_lo_W(codes), sign), sign), 8);
    const auto neg1 = Q6_Vh_vasl_VhR(Q6_V_vxor_VV(Q6_V_vand_VV(Q6_V_hi_W(codes), sign), sign), 8);
    out0 = Q6_V_vxor_VV(out0, neg0);
    out1 = Q6_V_vxor_VV(out1, neg1);
  }
  memcpy(output, &out0, 128);
  memcpy(output + 64, &out1, 128);
}

template<unsigned Bits>
inline void tq_decode_bits_hvx(const uint8_t* packed, const uint16_t* scales,
                              const uint16_t* centroids, uint16_t* output,
                              uint32_t rows) {
  static_assert(Bits >= 2 && Bits <= 8);
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
  HVX_Vector large_tables[4];
  bool symmetric = false;
  if constexpr (Bits == 8) {
    symmetric = true;
    for (unsigned i = 0; i < 128; ++i)
      symmetric &= centroids[i] == (centroids[255 - i] ^ 0x8000u);
  }
  if constexpr (Bits >= 7) {
    for (unsigned bank = 0; bank < (symmetric ? 2u : (1u << (Bits - 6))); ++bank) {
      alignas(128) uint16_t values[64];
      for (unsigned j = 0; j < 64; ++j)
        values[j] = centroids[(symmetric ? 128 : 0) + bank * 64 + j / 2 + (j % 2) * 32];
      memcpy(&large_tables[bank], values, 128);
    }
  }
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
    if constexpr (Bits == 8) {
      if (symmetric) {
        const auto half = Q6_Vb_vsplat_R(127);
        const auto negative = Q6_Q_vcmp_gt_VubVub(Q6_Vb_vsplat_R(128), bytes);
        const auto reflected = Q6_Vb_vsub_VbVb(half, bytes);
        const auto magnitude = Q6_V_vand_VV(Q6_V_vmux_QVV(negative, reflected, bytes), half);
        tq_store_large_row<7, true>(magnitude, large_tables, scales[row], output + row * 128, bytes);
      } else {
        // Retain exact generic LUT semantics for a non-symmetric caller.
        tq_store_large_row<8>(bytes, large_tables, scales[row], output + row * 128);
      }
      continue;
    }
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
    if constexpr (Bits == 7) {
      first = Q6_Vb_vlut32or_VbVbVbI(first, off, byte_lut, 3);
      second = Q6_Vb_vlut32or_VbVbVbI(second, nxt, byte_lut, 3);
    }
    const auto words = Q6_W_vshuff_VVR(first, second, -1);
    const auto lo = Q6_V_vand_VV(Q6_Vh_vlsr_VhVh(Q6_V_lo_W(words), shift0), mask);
    const auto hi = Q6_V_vand_VV(Q6_Vh_vlsr_VhVh(Q6_V_hi_W(words), shift1), mask);
    const auto indices = Q6_Vb_vpacke_VhVh(hi, lo);
    if constexpr (Bits == 7) {
      tq_store_large_row<7>(indices, large_tables, scales[row], output + row * 128);
      continue;
    }
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
