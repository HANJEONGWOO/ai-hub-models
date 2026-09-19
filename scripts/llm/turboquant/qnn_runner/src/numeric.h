// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
// Scalar encode/decode for QNN tensor element types.
//
// Fixed-point tensors use QNN's convention float = (q + offset) * scale. Encoding
// matches ai-hub-models' Python path: q = clip(rint(x / scale) - offset) in double
// precision with round-half-to-even.
#pragma once

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "qnn_api.h"

namespace tqrun {

inline uint16_t floatToHalf(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffff;
  if (((bits >> 23) & 0xff) == 0xff) return static_cast<uint16_t>(sign | 0x7c00 | (mantissa ? 0x200 : 0));
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00);
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    mantissa |= 0x800000;
    const int shift = 14 - exponent;
    uint32_t half = mantissa >> shift;
    const uint32_t rem = mantissa & ((1u << shift) - 1);
    const uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
  const uint32_t rem = mantissa & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
  return static_cast<uint16_t>(half);
}

inline float halfToFloat(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000) << 16;
  uint32_t exponent = (half >> 10) & 0x1f;
  uint32_t mantissa = half & 0x3ff;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while (!(mantissa & 0x400)) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x3ff;
      bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1f) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline const char* dataTypeName(Qnn_DataType_t type) {
  switch (type) {
    case QNN_DATATYPE_UFIXED_POINT_8: return "ufxp8";
    case QNN_DATATYPE_UFIXED_POINT_16: return "ufxp16";
    case QNN_DATATYPE_SFIXED_POINT_8: return "sfxp8";
    case QNN_DATATYPE_SFIXED_POINT_16: return "sfxp16";
    case QNN_DATATYPE_UINT_8: return "uint8";
    case QNN_DATATYPE_INT_32: return "int32";
    case QNN_DATATYPE_FLOAT_16: return "float16";
    case QNN_DATATYPE_FLOAT_32: return "float32";
    default: return "other";
  }
}

inline double quantizeClip(const TensorMeta& meta, double value, double lo, double hi) {
  std::fesetround(FE_TONEAREST);
  double q = std::nearbyint(value / meta.scale) - meta.offset;
  return q < lo ? lo : (q > hi ? hi : q);
}

inline void encodeScalar(const TensorMeta& meta, double value, uint8_t* dst) {
  switch (meta.dataType) {
    case QNN_DATATYPE_UFIXED_POINT_8: {
      const uint8_t q = static_cast<uint8_t>(quantizeClip(meta, value, 0, 255));
      std::memcpy(dst, &q, 1);
      return;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      const uint16_t q = static_cast<uint16_t>(quantizeClip(meta, value, 0, 65535));
      std::memcpy(dst, &q, 2);
      return;
    }
    case QNN_DATATYPE_SFIXED_POINT_8: {
      const int8_t q = static_cast<int8_t>(quantizeClip(meta, value, -128, 127));
      std::memcpy(dst, &q, 1);
      return;
    }
    case QNN_DATATYPE_SFIXED_POINT_16: {
      const int16_t q = static_cast<int16_t>(quantizeClip(meta, value, -32768, 32767));
      std::memcpy(dst, &q, 2);
      return;
    }
    case QNN_DATATYPE_UINT_8: {
      const uint8_t q = static_cast<uint8_t>(value);
      std::memcpy(dst, &q, 1);
      return;
    }
    case QNN_DATATYPE_FLOAT_16: {
      const uint16_t h = floatToHalf(static_cast<float>(value));
      std::memcpy(dst, &h, 2);
      return;
    }
    case QNN_DATATYPE_FLOAT_32: {
      const float f = static_cast<float>(value);
      std::memcpy(dst, &f, 4);
      return;
    }
    default:
      throw std::runtime_error("cannot encode scalars for tensor " + meta.name);
  }
}

inline double decodeScalar(const TensorMeta& meta, const uint8_t* src) {
  switch (meta.dataType) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      return (static_cast<double>(src[0]) + meta.offset) * meta.scale;
    case QNN_DATATYPE_UFIXED_POINT_16: {
      uint16_t q;
      std::memcpy(&q, src, 2);
      return (static_cast<double>(q) + meta.offset) * meta.scale;
    }
    case QNN_DATATYPE_SFIXED_POINT_16: {
      int16_t q;
      std::memcpy(&q, src, 2);
      return (static_cast<double>(q) + meta.offset) * meta.scale;
    }
    case QNN_DATATYPE_FLOAT_16: {
      uint16_t h;
      std::memcpy(&h, src, 2);
      return halfToFloat(h);
    }
    case QNN_DATATYPE_FLOAT_32: {
      float f;
      std::memcpy(&f, src, 4);
      return f;
    }
    default:
      throw std::runtime_error("cannot decode scalars for tensor " + meta.name);
  }
}

}  // namespace tqrun
