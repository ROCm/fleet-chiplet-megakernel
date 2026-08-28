# Step 5, batch 6: the two literals a full-body sweep turned up.
#
# Both are the same class of bug and neither breaks a build: a comment and a
# print that quietly describe a different model than the one being run.
#
#   router_tile_n's `# 16` is 128/8; change num_experts and the comment lies.
#   The phase-timing print hard-codes 36 twice -- once in the label, once in
#   the arithmetic. On a model with a different layer count it reports a
#   confidently wrong total, which is worse than reporting nothing.
#
# Everything else the sweep flagged was a false positive: `2` as a bf16 byte
# width, `2` as a two-element indptr, `8` inside an f-string field width.

SUBS = [
    ("            router_tile_n = NUM_EXPERTS // NUM_XCDS  # 16",
     "            router_tile_n = NUM_EXPERTS // NUM_XCDS  "
     "# {cfg.num_experts // cfg.num_xcds}", 1),

    ("            print(f\"    {{'TOTAL x 36 layers':62s}}: "
     "{{total_avg * 36 / 1000.0:8.3f}} ms\")",
     "            print(f\"    {{'TOTAL x {cfg.num_layers} layers':62s}}: "
     "{{total_avg * {cfg.num_layers} / 1000.0:8.3f}} ms\")", 1),
]
