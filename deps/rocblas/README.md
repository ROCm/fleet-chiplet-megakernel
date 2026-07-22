# ROCm cutlass-compat

Minimal CUTLASS compatibility layer for building Mirage with the ROCm/HIP backend.

When `USE_ROCM` is set, CMake adds `deps/rocblas/include` to the include path
**instead of** `deps/cutlass`. Any `#include "cutlass/..."` then resolves to
these headers rather than NVIDIA CUTLASS.

## Layout (mirrors cutlass include structure)

- `include/cutlass/cutlass.h` – minimal `cutlass::Status`, macros, etc.
- `include/cutlass/fast_math.h` – minimal math helpers
- `include/cutlass/array.h` – forwards to `mirage/utils/rocblas_helper.h` for `cutlass::Array`, `half_t`, `fast_exp_op`, etc.

Device macros (`CUTLASS_HOST_DEVICE`, `CUTLASS_DEVICE`) come from
`mirage/utils/device_macros.h` when using ROCm.

## Usage

No code changes required. Use `#include "cutlass/cutlass.h"` and
`#include "cutlass/fast_math.h"` as in the CUDA build; for ROCm builds
these resolve to this compat layer.
