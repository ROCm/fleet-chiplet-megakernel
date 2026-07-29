cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
message(STATUS "CMake Version: ${CMAKE_VERSION}")

# ----------------------------
# Options
# ----------------------------
option(USE_ROCM "Build with HIP backend" ON)
option(USE_CUDA "Build with CUDA backend" OFF)
option(USE_NKI "Build with NKI backend" OFF)
option(USE_FORMAL_VERIFIER "Enable formal verifier" OFF)
option(BUILD_CPP_EXAMPLES "Build C++ examples" OFF)
option(MIRAGE_BUILD_UNIT_TEST "Build unit tests" OFF)

# ROCm path
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm install path")

# Hard fail if CUDA requested (this file is HIP-only)
if (USE_CUDA)
  message(FATAL_ERROR "This CMakeLists.txt is HIP-only. Configure with -DUSE_CUDA=OFF.")
endif()

# ----------------------------
# Load optional config.cmake
# ----------------------------
if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
elseif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
endif()

# ----------------------------
# IMPORTANT: For CMake 4.2, HIP compiler must be Clang (NOT hipcc).
# Set compiler BEFORE project().
# ----------------------------
if (USE_ROCM)
  set(ROCM_CLANGXX "${ROCM_PATH}/llvm/bin/clang++")

  if (NOT EXISTS "${ROCM_CLANGXX}")
    message(FATAL_ERROR
      "ROCm clang++ not found at ${ROCM_CLANGXX}\n"
      "Check ROCM_PATH or install rocm-llvm.\n"
      "Try: ls -l /opt/rocm/llvm/bin/clang++")
  endif()

  # Tell CMake to use ROCm clang++ for HIP
  set(CMAKE_HIP_COMPILER "${ROCM_CLANGXX}" CACHE FILEPATH "HIP compiler (clang++)" FORCE)

  # Default arch if not provided
  if (NOT DEFINED CMAKE_HIP_ARCHITECTURES OR "${CMAKE_HIP_ARCHITECTURES}" STREQUAL "")
    if (DEFINED ENV{AMDGPU_TARGETS} AND NOT "$ENV{AMDGPU_TARGETS}" STREQUAL "")
      set(CMAKE_HIP_ARCHITECTURES "$ENV{AMDGPU_TARGETS}" CACHE STRING "" FORCE)
    else()
      set(CMAKE_HIP_ARCHITECTURES "gfx950" CACHE STRING "" FORCE)
    endif()
  endif()
endif()

if (USE_ROCM)
  set(ROCM_CLANGXX "${ROCM_PATH}/llvm/bin/clang++")
  set(CMAKE_CXX_COMPILER "${ROCM_CLANGXX}" CACHE FILEPATH "C++ compiler" FORCE)
endif()

# ----------------------------
# Project languages
# ----------------------------
if (USE_ROCM)
  project(MIRAGE LANGUAGES C CXX HIP)
else()
  project(MIRAGE LANGUAGES C CXX)
endif()

message(STATUS "CMAKE_HIP_COMPILER=${CMAKE_HIP_COMPILER}")
message(STATUS "CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")

# ----------------------------
# General config
# ----------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message(STATUS "Build in Debug mode")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O0 -g -Wall -fPIC")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2 -Wall -fPIC")
endif()

# HIP flags (optional)
if (USE_ROCM)
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O0 -g")
  else()
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O2")
  endif()
endif()

# ----------------------------
# Include dirs
# ----------------------------
include_directories(${CMAKE_INCLUDE_PATH})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

# ROCm headers/libs
if (USE_ROCM)
  include_directories("${ROCM_PATH}/include")
  link_directories("${ROCM_PATH}/lib" "${ROCM_PATH}/lib64")
endif()

# CUTLASS vs rocblas cutlass-compat: use rocblas for ROCm, cutlass for CUDA
if (USE_ROCM)
  include_directories(${PROJECT_SOURCE_DIR}/deps/rocblas/include)
else()
  include_directories(${PROJECT_SOURCE_DIR}/deps/cutlass/include)
  include_directories(${PROJECT_SOURCE_DIR}/deps/cutlass/tools/util/include)
endif()

# ----------------------------
# Sources
# ----------------------------
file(GLOB_RECURSE MIRAGE_CC_SRCS
  ${PROJECT_SOURCE_DIR}/src/*.cc
)

file(GLOB_RECURSE MIRAGE_GPU_SRCS
  ${PROJECT_SOURCE_DIR}/src/*.hip
  ${PROJECT_SOURCE_DIR}/src/*.hip.cu
  ${PROJECT_SOURCE_DIR}/src/kernel/cuda/*.cu
  ${PROJECT_SOURCE_DIR}/src/kernel/device_memory_manager.cu
  ${PROJECT_SOURCE_DIR}/src/utils/rocm_helper.cu
)

set(MIRAGE_SRCS ${MIRAGE_CC_SRCS})

if (USE_ROCM)
  list(APPEND MIRAGE_SRCS ${MIRAGE_GPU_SRCS})
endif()

# Exclude transpiler if desired (kept behavior from your original "no CUDA" path)
if (NOT USE_ROCM)
  file(GLOB_RECURSE MIRAGE_EXCLUDE_SRCS
    ${PROJECT_SOURCE_DIR}/src/transpiler/*.cc
  )
  list(REMOVE_ITEM MIRAGE_SRCS ${MIRAGE_EXCLUDE_SRCS})
endif()

# Force .cu/.hip to compile as HIP when using HIP
if (USE_ROCM AND MIRAGE_GPU_SRCS)
  foreach(src_file ${MIRAGE_GPU_SRCS})
    set_source_files_properties(${src_file} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()

message(STATUS "Total sources: ${MIRAGE_SRCS}")

# ----------------------------
# Dependencies: Z3
# ----------------------------
if (Z3_CXX_INCLUDE_DIRS AND Z3_LIBRARIES)
  message(STATUS "Z3_CXX_INCLUDE_DIRS: ${Z3_CXX_INCLUDE_DIRS}")
  message(STATUS "Z3_LIBRARIES: ${Z3_LIBRARIES}")
else()
  find_package(Z3 REQUIRED)
  message(STATUS "Z3_FOUND: ${Z3_FOUND}")
  message(STATUS "Found Z3 ${Z3_VERSION_STRING}")
  message(STATUS "Z3_DIR: ${Z3_DIR}")
endif()
include_directories(${Z3_CXX_INCLUDE_DIRS})

# JSON
add_subdirectory(deps/json)

# Link dirs (kept)
link_directories(${CMAKE_SOURCE_DIR}/build/release)

# ----------------------------
# Library target
# ----------------------------
add_library(mirage_runtime ${MIRAGE_SRCS})
target_compile_features(mirage_runtime PUBLIC cxx_std_17)
if (USE_ROCM)
  target_compile_definitions(mirage_runtime PUBLIC
    MIRAGE_BACKEND_USE_ROCM
    __HIP_PLATFORM_AMD__=1
  )
  target_link_libraries(mirage_runtime PUBLIC amdhip64)
endif()

# Backend defines
if (USE_ROCM)
  target_compile_definitions(mirage_runtime PUBLIC __HIP_PLATFORM_AMD__=1 MIRAGE_FINGERPRINT_USE_ROCM)
endif()

if (USE_NKI)
  target_compile_definitions(mirage_runtime PUBLIC MIRAGE_BACKEND_USE_NKI)
  target_compile_definitions(mirage_runtime PUBLIC MIRAGE_FINGERPRINT_USE_CPU)
endif()

if (USE_FORMAL_VERIFIER)
  target_compile_definitions(mirage_runtime PUBLIC MIRAGE_USE_FORMAL_VERIFIER)
endif()

# Core libs
target_link_libraries(mirage_runtime PUBLIC ${Z3_LIBRARIES} nlohmann_json::nlohmann_json)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mirage_runtime PUBLIC OpenMP::OpenMP_CXX)
else()
  # Fallback: link gomp directly if find_package fails
  target_link_libraries(mirage_runtime PUBLIC gomp)
endif()

# HIP runtime lib (works without HIPConfig.cmake)
if (USE_ROCM)
  target_link_libraries(mirage_runtime PUBLIC amdhip64)
endif()

# Your existing Rust shared libs (kept)
set(ABSTRACT_SUBEXPR_LIBRARIES ${PROJECT_SOURCE_DIR}/build/abstract_subexpr/release/libabstract_subexpr.so)
set(FORMAL_VERIFIER_LIBRARIES ${PROJECT_SOURCE_DIR}/build/formal_verifier/release/libformal_verifier.so)

target_link_libraries(mirage_runtime PUBLIC abstract_subexpr)
target_link_libraries(mirage_runtime PUBLIC "-lstdc++fs")
target_link_libraries(mirage_runtime PUBLIC ${ABSTRACT_SUBEXPR_LIBRARIES})
target_link_libraries(mirage_runtime PUBLIC ${FORMAL_VERIFIER_LIBRARIES})

target_include_directories(mirage_runtime PUBLIC ${PROJECT_SOURCE_DIR}/include)

# ----------------------------
# Rust toolchain checks (kept)
# ----------------------------
execute_process(COMMAND rustc --version
  RESULT_VARIABLE RUST_COMMAND_RESULT
  OUTPUT_VARIABLE RUSTC_OUTPUT
  ERROR_QUIET)

if (NOT RUST_COMMAND_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Rust is not installed on the system. Please install it by running:\n"
    "'curl https://sh.rustup.rs -sSf | sh -s -- -y'\n")
endif()

execute_process(COMMAND cargo --version
  RESULT_VARIABLE CARGO_RESULT
  OUTPUT_QUIET ERROR_QUIET)

if (NOT CARGO_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Rust is installed, but cargo is not. Please install it by running:\n"
    "'curl https://sh.rustup.rs -sSf | sh -s -- -y'\n")
endif()

# ----------------------------
# Install rules
# ----------------------------
install(TARGETS mirage_runtime LIBRARY DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include DESTINATION .)
install(FILES ${ABSTRACT_SUBEXPR_LIBRARIES} DESTINATION lib)

# ----------------------------
# Unit tests
# ----------------------------
if (MIRAGE_BUILD_UNIT_TEST)
  enable_testing()
  add_subdirectory(tests)
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(mirage_runtime stdc++fs)
  endif()
endif()

