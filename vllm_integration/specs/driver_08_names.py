# Step 5, batch 8: the model name, in both spellings.
#
# `gpt_oss_120b` is the C symbol prefix and the .so / .cuh stem;
# `gpt-oss-120b` is the human-readable name in prose and prints. cfg.name and
# cfg.name_clean already hold both, and the kernel emitter already uses them.
#
# Mechanical, but it is the substitution that decides whether the emitter is a
# GPT-OSS emitter or a MoE emitter: leave the name baked in and a second MoE
# config silently produces a driver that dlopen's the wrong library and calls
# the wrong 24 entrypoints.
#
# Substituted as bare strings with a total count rather than line by line --
# every occurrence is the model name and every one wants the same treatment.

# One of the 44 was in the emitter's own docstring, which is not emitted
# output -- reverted by hand afterwards. The count guard cannot tell prose
# from payload; only the whole-function span it operates on is mechanical.

SUBS = [
    ("gpt_oss_120b", "{cfg.name_clean}", 44),
    ("gpt-oss-120b", "{cfg.name}", 7),
]
