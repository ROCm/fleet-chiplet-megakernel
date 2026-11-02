/* Copyright 2026 Nikhil (github.com/badnikhil)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Kernel switches. Defaults = the measured best configuration on MI300X (with FLEET_KNOBS defaults
// dag=1,fold=1): DSK_COMBINE_BLOCK 1, DSK_NORM_FIXED 1, DSK_MERGE_FIXED 2, the rest 0. Value 0 everywhere = the kernels as measured
// before the DAG round: -DDSK_COMBINE_BLOCK=0 -DDSK_NORM_FIXED=0
// -DDSK_MERGE_FIXED=0 or edit below, with FLEET_KNOBS=dag=0,fold=0 for the chain. Every value is bit-identical:
// demo/deepseek_v2_lite/tests/test_kernels_cpu.py builds the harness with each switch set and asserts torch.equal on every entry point.
// Flip one per compile on the box: edit the default below or pass -D<NAME>=<v>.
// Each switch is also the default of a template parameter, so a register string may pin a value explicitly.
#pragma once
#ifndef DSK_COMBINE_BLOCK  // F1 moe_combine_task_impl<E, TOPK, N, BLOCK>: 1 = one contiguous 16-byte block per thread
#define DSK_COMBINE_BLOCK 1
#endif
#ifndef DSK_NORM_FIXED  // F3 rmsnorm_rows<FIXED> (the dsk norm prologues; the RMS_NORM task itself is Fleet's rmsnorm_mi300)
#define DSK_NORM_FIXED 1
#endif
#ifndef DSK_MERGE_FIXED  // F2 o_proj_merge_task_impl<..., FIXED>: 0 runtime nc (perf/fold-full), 1 unrolled NCHUNK, 2 = 1 + 16-byte o_part loads
#define DSK_MERGE_FIXED 2
#endif
#ifndef DSK_ATTN_SCORE_HOIST  // A1 attn_chunk_task_impl<..., ROPE, HOIST>: 1 = the 24 K-row 16-byte loads issued before the fma chain
#define DSK_ATTN_SCORE_HOIST 0
#endif
#ifndef DSK_GEMV_RPW  // A2 gemv_rows<ROWS, K, STAGED, RPW_SW>: rows per wave in flight for EVERY gemv tile incl. the gate (0 = 4, 2 for 88 rows)
#define DSK_GEMV_RPW 0
#endif
