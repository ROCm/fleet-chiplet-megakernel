# Step 6, batch 3: the launch geometry comment.
#
# The dense arm already emits this line from cfg; the MoE arm, pasted verbatim,
# still carries the GPT-OSS numbers as text. Same line, same three fields, so
# it gets the same treatment -- otherwise a second MoE config emits a comment
# describing a launch it is not performing.
#
# `dim3 block(256, 1, 1)` is deliberately NOT substituted. 256 is the kernel's
# NUM_THREADS, which the kernel emitter static_asserts against blockDim; it is
# a property of the device-function library's wave layout, not of the model,
# and the dense arm leaves it literal too. Parameterizing it here without a
# matching cfg field on the kernel side would create exactly the two-sided
# disagreement this batch exists to remove.

SUBS = [
    ("    // 240 threadblocks = 30 workers/XCD x 8 XCDs",
     "    // {cfg.total_workers} threadblocks = {cfg.workers_per_xcd} "
     "workers/XCD x {cfg.num_xcds} XCDs", 1),
]
