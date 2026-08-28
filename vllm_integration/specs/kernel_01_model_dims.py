# Step 4, batch 1: scalar model dimensions.
#
# Substitute by MEANING, not by value. 2944 appears three times as three
# DIFFERENT config fields that merely coincide for GPT-OSS:
#   HIDDEN_SIZE           = padded_hidden_size          (2880 -> 2944)
#   INTERMEDIATE_SIZE     = padded_intermediate_size    (2880 -> 2944)
#   MOE_INTERMEDIATE_SIZE = padded_moe_intermediate_size(2880 -> 2944)
# A blind global replace of "2944" would fuse them and silently survive the
# gate for GPT-OSS while being wrong for every other model.
#
# Each `old` is a whole line, so a bare literal like 64 cannot be caught
# somewhere it does not belong.

SUBS = [
    ("static constexpr int NUM_LAYERS = 36;",
     "static constexpr int NUM_LAYERS = {cfg.num_layers};", 1),

    ("static constexpr int HIDDEN_SIZE = 2944;",
     "static constexpr int HIDDEN_SIZE = {cfg.padded_hidden_size};", 1),

    ("static constexpr int ACTUAL_HIDDEN_DIM = 2880;  // for RMSNorm mean",
     "static constexpr int ACTUAL_HIDDEN_DIM = {cfg.hidden_size};  "
     "// for RMSNorm mean", 1),

    ("static constexpr int INTERMEDIATE_SIZE = 2944;",
     "static constexpr int INTERMEDIATE_SIZE = {cfg.padded_intermediate_size};",
     1),

    ("static constexpr int VOCAB_SIZE = 201088;",
     "static constexpr int VOCAB_SIZE = {cfg.vocab_size};", 1),

    # The trailing comment says "256" but the real alignment is
    # output_per_wg * num_xcds = 512. Preserved verbatim: this file is a byte
    # target, and correcting a stale comment is a separate, deliberate change.
    ("static constexpr int PADDED_VOCAB_SIZE = 201216;  // next multiple of 256",
     "static constexpr int PADDED_VOCAB_SIZE = {cfg.padded_vocab_size};  "
     "// next multiple of 256", 1),
]
