#!/bin/bash
# Codegen-only compile check for gang_moe_fused_mxfp4_mi300.cuh.
#
# The megakernel is JIT-compiled from a generated TU at run time, so nothing in
# the tree type-checks a `.cuh` edit until a full run -- which needs the model
# weights. This drives hipcc over a one-instantiation TU instead, all the way
# to gfx950 object code so the *assembler* sees the inline asm. A malformed
# operand list or a tripped static_assert fails here in seconds.
#
# The define set is not hand-maintained: it is whatever
# mirage.mpk.persistent_kernel.get_compile_command() would pass for an offline
# bs=1 gfx950 build, so this cannot drift from the real build. Extra defines
# for the flag under test are passed as arguments:
#
#   ./compile_check_moe_fused.sh
#   ./compile_check_moe_fused.sh -DMPK_MFMA_PINGPONG_SCHED
#   ./compile_check_moe_fused.sh -DMPK_MFMA_PINGPONG_SCHED \
#       -DMPK_MOE_DUAL_ACCUMULATOR -DMPK_MOE_QUAD_ACCUMULATOR
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")"
F="$(cd ../.. && pwd)"
PY_INC="$(ls -d /usr/include/python3.* | head -1)"
ARCH="${ARCH:-gfx950}"

DEFINES=$(MIRAGE_HOME="$F" PYTHONPATH="$F/python" python3 - "$F" "$PY_INC" <<'EOF'
import os, sys, types
sys.path.insert(0, os.path.join(sys.argv[1], "python"))
from mirage.mpk.persistent_kernel import get_compile_command

home, py_inc = sys.argv[1], sys.argv[2]
mpk = types.SimpleNamespace(mode="offline", max_num_batched_requests=1,
                            max_num_batched_tokens=1, max_num_pages=16,
                            page_size=4096, max_seq_length=512)
cmd = get_compile_command(
    mpk=mpk, target_cc=95, cc="hipcc", file_name="X.cu",
    py_include_dir=py_inc, mirage_home_path=home,
    mirage_inc_path=os.path.join(home, "include"),
    mirage_deps_path=os.path.join(home, "deps"),
    nvshmem_inc_path=None, nvshmem_lib_path=None,
    mpi_inc_path=None, mpi_lib_path=None, py_so_path="X.so",
    profiling=False, use_nvshmem=False, num_workers=248,
    num_local_schedulers=8, num_remote_schedulers=0, use_cutlass_kernel=False)
print(" ".join(c for c in cmd if c.startswith("-D")))
EOF
)

# shellcheck disable=SC2086
hipcc -x hip compile_check_moe_fused.hip -O2 -c -o /tmp/compile_check_moe_fused.o \
  -I"$PY_INC" -I"$F/include" -I"$F/include/mirage/hip_compat" \
  -I"$F/include/mirage/persistent_kernel" -I"$F/deps/rocblas/include" \
  -I"$F/deps/cutlass/include" -I"$F/deps/cutlass/tools/util/include" \
  -I"$F/deps/composable_kernel/include" -I/opt/rocm/include \
  -I"$F/deps/json/include" \
  --offload-arch="$ARCH" -std=c++17 -fPIC \
  $DEFINES "$@"
echo "OK: compiled for $ARCH with extra defines: $*"
