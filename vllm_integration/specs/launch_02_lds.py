# Step 6, batch 2: the six LDS-size sites.
#
# 155648 B is what gfx950 leaves usable of its 160 KB per-workgroup LDS. It
# appears twice as a `constexpr int LDS_SIZE` (the direct launch and the
# decode_loop) and four times as `kparams*.sharedMemBytes` on the hipGraph
# paths. All six must agree: hipGraphAddKernelNode does not cross-check them,
# so a disagreement is a launch failure on one path and an LDS overrun on
# another -- and only whichever of the four decode strategies the env var
# selects would show it.
#
# The two `constexpr int LDS_SIZE` lines are identical strings in two different
# functions, so they go in as one sub with count 2.

SUBS = [
    ("    constexpr int LDS_SIZE = 155648;",
     "    constexpr int LDS_SIZE = {cfg.lds_bytes};", 2),
    ("    kparams.sharedMemBytes = 155648;",
     "    kparams.sharedMemBytes = {cfg.lds_bytes};", 2),
    ("    kparams_first.sharedMemBytes = 155648;",
     "    kparams_first.sharedMemBytes = {cfg.lds_bytes};", 1),
    ("    kparams_pipe.sharedMemBytes = 155648;",
     "    kparams_pipe.sharedMemBytes = {cfg.lds_bytes};", 1),
]
