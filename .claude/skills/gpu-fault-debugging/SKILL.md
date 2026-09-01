---
name: gpu-fault-debugging
description: Find the exact source line of a HIP kernel memory fault using rocgdb before theorizing about causes. Use when a kernel hits a memory access fault, 'Page not present', or 'Write access to a read-only page' (or similar) on AMD GPUs.
---

# GPU Fault Debugging with rocgdb

## When to use

A HIP kernel hits a memory access fault ("Write access to a read-only page", "Page not present", or similar). You need to find the **exact source line** causing the fault. Do this FIRST before theorizing about the cause — guessing at fault causes without precise location data wastes time and leads to wrong conclusions.

## Process

### Step 1: Isolate the offending kernel

Run with `AMD_SERIALIZE_KERNEL=3` to force synchronous kernel launches. This makes the fault report point to the correct kernel (otherwise async dispatch can make it ambiguous).

```bash
AMD_SERIALIZE_KERNEL=3 python3 your_script.py --args
```

This may or may not narrow things down, but it's a 0-effort first step.

### Step 2: Build with debug info

Add `-ggdb` to the compiler options for the target that contains the faulting kernel. In CMakeLists.txt, this is typically a one-line change:

```cmake
target_compile_options(your_target PRIVATE -ggdb)
```

Or for the whole HIP target library:

```cmake
target_compile_options(<your-target> PRIVATE -ggdb)
```

Rebuild the target.

### Step 3: Launch under rocgdb

```bash
rocgdb python3
```

Or for a standalone HIP binary:

```bash
rocgdb ./your_binary
```

### Step 4: Enable precise memory fault reporting

Inside rocgdb:

```
(gdb) set amdgpu precise-memory on
```

This makes the GPU report the exact instruction that faulted, rather than a later instruction in the pipeline. Without this, the reported PC can be several instructions past the actual fault.

### Step 5: Run the program

```
(gdb) run your_script.py --args
```

Or for a standalone binary:

```
(gdb) run arg1 arg2
```

### Step 6: Backtrace at the fault

When the fault fires, rocgdb stops at the faulting instruction. Run:

```
(gdb) bt
```

This gives you the exact source file and line number of the faulting memory access. From here, the root cause is usually obvious — you can see exactly which pointer and index are involved.

## Why this process matters

Agents are very good at fixing bugs once they see the precise fault location. Without it, they tend to guess at causes by reading code and reasoning about buffer sizes — which is slow, error-prone, and often leads to wrong conclusions. A 5-minute rocgdb session replaces hours of speculation.

## Quick reference

```bash
# Full sequence:
AMD_SERIALIZE_KERNEL=3 rocgdb ./my_binary
(gdb) set amdgpu precise-memory on
(gdb) run
# ... fault happens ...
(gdb) bt
```
