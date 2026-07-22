#!/bin/bash
# Build script for standalone kernel tests

set -e

echo "Building MFMA GEMM test..."
hipcc -o test_mfma_simple test_mfma_simple.hip \
    -D__HIP_PLATFORM_AMD__ \
    -munsafe-fp-atomics \
    -O3 \
    -std=c++17

echo "Build complete!"
echo ""
echo "Run with: ./test_mfma_simple"
