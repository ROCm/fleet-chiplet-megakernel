# Step 6, batch 4: the model name, both spellings.
#
# Same substitution as specs/driver_08_names.py, and the same reason: the name
# is the C symbol prefix (`{name}_kernel`, `{name}_launch`, the `{name}::`
# namespace), the .cuh include, and the .so stem. Leave it baked in and a
# second MoE config emits a wrapper that includes the wrong header and defines
# entrypoints the driver will not find.
#
# The docstring occurrence that tripped up the driver batch was rewritten by
# hand first, so the count here is the payload count exactly.

SUBS = [
    ("gpt_oss_120b", "{cfg.name_clean}", 32),
    ("gpt-oss-120b", "{cfg.name}", 1),
]
