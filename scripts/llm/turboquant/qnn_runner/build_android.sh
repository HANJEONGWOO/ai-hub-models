#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Builds qnn-llm-runner for arm64 Android against the QAIRT headers (not vendored).
set -euo pipefail

NDK=${ANDROID_NDK_ROOT:-$HOME/android-ndk-r26c}
QNN_SDK_ROOT=${QNN_SDK_ROOT:-$HOME/qairt/2.48.0.260626}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:-$HOME/.qaihm/tmp/turboquant/qnn_runner/android-arm64}
mkdir -p "$OUT"

"$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++" \
  --target=aarch64-linux-android31 -std=c++17 -O2 -Wall -Wextra -fPIE -pie \
  -static-libstdc++ -Wl,-z,max-page-size=16384 \
  -I"$QNN_SDK_ROOT/include/QNN" \
  "$HERE/src/main.cpp" "$HERE/src/session.cpp" "$HERE/src/qnn_api.cpp" \
  -ldl -llog -o "$OUT/qnn-llm-runner"
echo "$OUT/qnn-llm-runner"
