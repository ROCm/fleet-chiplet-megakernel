cmake_minimum_required(VERSION 3.21)
project(mirage_kernel_test LANGUAGES CXX HIP)

# Find HIP
find_package(hip REQUIRED)
find_package(rocblas REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CK Tile include path
set(CK_TILE_INCLUDE "/opt/rocm/include")

# Add executable
add_executable(test_kernels test_kernels.cpp)

target_include_directories(test_kernels PRIVATE
    ${CK_TILE_INCLUDE}
    ${CMAKE_SOURCE_DIR}/../../include
)

target_link_libraries(test_kernels PRIVATE
    hip::device
    roc::rocblas
)

target_compile_options(test_kernels PRIVATE
    -D__HIP_PLATFORM_AMD__
    -DMIRAGE_AMD_MI300
    -munsafe-fp-atomics
    -ffast-math
)

# Simple MFMA test
add_executable(test_mfma_simple test_mfma_simple.hip)

target_include_directories(test_mfma_simple PRIVATE
    ${CK_TILE_INCLUDE}
)

target_link_libraries(test_mfma_simple PRIVATE
    hip::device
)

target_compile_options(test_mfma_simple PRIVATE
    -D__HIP_PLATFORM_AMD__
    -munsafe-fp-atomics
)
