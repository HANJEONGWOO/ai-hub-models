// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <string.h>

// One packed 128-coordinate row and one FP16 query. Only registers contain
// signs/signed query lanes: no K reconstruction or bit-plane tensor is stored.
// Pairwise signed sums and horizontal reduction accumulate in IEEE FP32.
inline float tq_bitplane_dot_hvx(const uint8_t* packed, const uint16_t* query,
                                 const float* beta) {
  HVX_Vector bytes = Q6_V_vzero(), q0, q1;
  memcpy(&bytes, packed, 64);
  memcpy(&q0, query, 128);
  memcpy(&q1, query + 64, 128);
  const auto lo = Q6_V_vand_VV(bytes, Q6_Vb_vsplat_R(15));
  const auto hi = Q6_Vub_vlsr_VubR(bytes, 4);
  const auto idx = Q6_V_lo_W(Q6_W_vshuff_VVR(lo, hi, -1));
  const auto words = Q6_Wuh_vunpack_Vub(idx);
  const auto sign = Q6_Vh_vsplat_R(0x8000);
  const auto one = Q6_Vh_vsplat_R(0x3c00); // FP16 1.0
  float dot = 0.0f;
  for (unsigned m = 0; m < 4; ++m) {
    const auto mask0 = Q6_V_vxor_VV(sign, Q6_V_vand_VV(sign,
        Q6_Vh_vasl_VhR(Q6_V_lo_W(words), 15 - m)));
    const auto mask1 = Q6_V_vxor_VV(sign, Q6_V_vand_VV(sign,
        Q6_Vh_vasl_VhR(Q6_V_hi_W(words), 15 - m)));
    const auto s0 = Q6_V_vxor_VV(q0, mask0);
    const auto s1 = Q6_V_vxor_VV(q1, mask1);
    auto sums = Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vdmpy_VhfVhf(s0, one),
                                  Q6_Vsf_vdmpy_VhfVhf(s1, one));
    // Each lane receives the same 128-coordinate sum after five butterfly steps.
    sums = Q6_Vsf_vadd_VsfVsf(sums, Q6_V_vror_VR(sums, 64));
    sums = Q6_Vsf_vadd_VsfVsf(sums, Q6_V_vror_VR(sums, 32));
    sums = Q6_Vsf_vadd_VsfVsf(sums, Q6_V_vror_VR(sums, 16));
    sums = Q6_Vsf_vadd_VsfVsf(sums, Q6_V_vror_VR(sums, 8));
    sums = Q6_Vsf_vadd_VsfVsf(sums, Q6_V_vror_VR(sums, 4));
    float plane;
    memcpy(&plane, &sums, sizeof(plane));
    dot += beta[m] * plane;
  }
  return dot;
}
