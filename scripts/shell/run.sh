#!/bin/bash

set -e

TARGET="single_run"
INPUT="examples/single_run/Cl2.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE=Release "$PROJECT_ROOT"

echo "Building $TARGET..."
compile_start=$(date +%s.%N)

cmake --build . --target "$TARGET" -j

compile_end=$(date +%s.%N)

compile_time=$(python3 - <<EOF
print(${compile_end} - ${compile_start})
EOF
)

echo "Compile time: ${compile_time} s"

echo
echo "========================================"
echo "Running: $TARGET"
echo "Input:   $INPUT"
echo "========================================"

run_start=$(date +%s.%N)

./"$TARGET" "$PROJECT_ROOT/$INPUT"

run_end=$(date +%s.%N)

run_time=$(python3 - <<EOF
print(${run_end} - ${run_start})
EOF
)

echo
echo "Run time: ${run_time} s"
echo "Finished."
