#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$APP_DIR/build-android"
NDK_ROOT="${ANDROID_NDK_ROOT:-/home/assassinaj602/Android/Sdk/ndk/30.0.16248370}"
TOOLCHAIN="$NDK_ROOT/build/cmake/android.toolchain.cmake"

if [ ! -f "$TOOLCHAIN" ]; then
  echo "Android NDK toolchain not found at: $TOOLCHAIN" >&2
  exit 1
fi

echo ">>> Building train_multimodal for Android arm64-v8a..."
cmake -S "$APP_DIR" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" --target train_multimodal -j4

echo "========================================================================="
echo "  Android ARM64 Build Complete: $BUILD_DIR/train_multimodal"
echo "========================================================================="
