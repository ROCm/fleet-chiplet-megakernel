"""How fleet_mk's MoE expert weights are laid out in memory, for the vLLM plugin.

The standalone driver (`demo_gpt_oss_120b.py`) grew four knobs describing an
expert buffer that fleet_mk did not necessarily write: a K row pitch, an N expert
pitch, a data/scale section split, and whether the two sections share one
allocation. Each was landed and measured there on fleet_mk's OWN slab, so that when
the buffer finally belongs to someone else, "someone else's bytes" is the only
variable left.

This module is that geometry for the plugin. It exists as its own file rather
than as constants inside `packing_moe.py` because the numbers are needed in two
places that must agree exactly -- the packer call (which decides how bytes are
arranged) and the pointer table (which decides where the kernel looks for them)
-- and a pitch that disagrees between those two does not fault. It reads real
scale bytes for the wrong rows and emits fluent garbage.

## The point of all four knobs: aliasing

vLLM keeps GPT-OSS's experts as two plain tensors per layer, `w13_weight` and
`w2_weight`, 1728 MiB/layer, 60.75 GiB across 36 layers. They survive
`process_weights_after_loading` at the same address with the same bytes (ROCm's
`StridedLayout` swizzle is identity; the shape change is a transposed *view*, so
a linear reader still sees `[E, rows, K/2]` row-major). Fleet MK packs its own copy
of exactly those bytes. Aliasing means pointing the kernel at vLLM's and not
packing ours -- but only if fleet_mk's addressing can be made to match a layout it
did not choose:

  * `k_stride`      -- vLLM rounds K to 3072, so its rows sit 1536 B apart while
                       the MFMA still reduces 2944. Pitch and reduction are
                       independent (`MPK_MOE_K_STRIDE`).
  * `n_stride`      -- vLLM rounds the output axis to 3072 rows too, while fleet_mk
                       computes 2944 (`MPK_MOE_N_STRIDE`).
  * `split_scales`  -- vLLM's data tensor obviously does not carry fleet_mk's scales
                       interleaved inside it (`MPK_MOE_SPLIT_SCALES`).
  * separate allocations -- and the scale slab cannot be at a fixed offset from a
                       buffer vLLM allocated.

## Why the scales stay fleet_mk's own

vLLM's TRITON MXFP4 backend **deletes** `w13_weight_scale` / `w2_weight_scale`
in `process_weights_after_loading`; the swizzled scales move inside the precision
configs. So there is nothing to alias on the scale side. Fleet MK keeps packing
scales -- but *only* scales, via `section="scales"`, because packing the data
section too would materialize the very 60 GiB copy the aliasing exists to
remove.

## Why split mode here always means two allocations

The driver has a `MOE_SPLIT_BUFFERS` knob because it could put the two sections
adjacent in one allocation and derive the scale base by arithmetic. That
arrangement cannot survive aliasing -- there is no offset from vLLM's tensor that
reaches fleet_mk's scales -- so the plugin does not implement it. Split here always
means two independent bases, which is also the only arrangement the aliased path
can use. The kernel is indifferent: it takes both bases as arguments and never
assumes a relationship between them, which is why none of this needs a flag
beyond `-DMPK_MOE_SPLIT_SCALES`.

## The three knobs that MUST match the .so

`k_stride`, `n_stride` and `split_scales` are compiled into the kernel
(`-DMPK_MOE_K_STRIDE`, `-DMPK_MOE_N_STRIDE`, `-DMPK_MOE_SPLIT_SCALES`). Nothing
checks agreement at runtime -- there is no channel to check it through -- so
`describe()` prints the layout unconditionally at setup, and the build line is
part of the layout's definition, not an afterthought.
"""

import os


def _env_int(name):
    return int(os.environ.get(name, "0") or "0")


class MoeLayout:
    """Expert-buffer geometry for one MoE model, derived from spec + env.

    Attributes used by the packer:
      k_stride_blocks, split_scales, w13_out_stride, w2_out_stride, alias
    Attributes used by the pointer table:
      split_scales, alias
    """

    def __init__(self, spec):
        S = self.spec = spec
        # GPT-OSS pads hidden and intermediate alike (both to padded_hidden_size),
        # which is what lets ONE n_stride knob cover both axes. Assert it rather
        # than assume it: a model where they differ needs two knobs, and would
        # otherwise silently get W2's pitch applied to W13's rows.
        ph = S.padded_hidden_size
        self.hidden = ph
        self.w13_out = 2 * ph
        self.w2_out = ph
        self.w13_opw = S.gemm["w13_output_per_wg"]
        self.w2_opw = S.gemm["w2_output_per_wg"]

        # ── K row pitch ──────────────────────────────────────────────────────
        self.k_stride = _env_int("FLEET_MK_MOE_K_STRIDE") or ph
        assert self.k_stride >= ph and self.k_stride % 32 == 0, (
            f"FLEET_MK_MOE_K_STRIDE={self.k_stride} must be a multiple of 32 and at "
            f"least the {ph}-wide reduction")
        self.k_stride_blocks = self.k_stride // 32

        # ── data/scale section split ─────────────────────────────────────────
        self.split_scales = os.environ.get("FLEET_MK_MOE_SPLIT_SCALES") == "1"

        # ── N expert pitch ───────────────────────────────────────────────────
        # In units of the intermediate/hidden row count (3072 for vLLM), not of
        # W13's doubled axis: W13 is gate+up interleaved, so its axis holds two
        # of them. The kernel's W13_N_STRIDE is the identical expression.
        self.n_stride = _env_int("FLEET_MK_MOE_N_STRIDE")
        assert not self.n_stride or self.split_scales, (
            "FLEET_MK_MOE_N_STRIDE requires FLEET_MK_MOE_SPLIT_SCALES=1 -- a foreign "
            "expert pitch implies a foreign buffer, which cannot also carry "
            "fleet_mk's per-workgroup interleaved scales")
        assert not self.n_stride or self.n_stride >= ph, (
            f"FLEET_MK_MOE_N_STRIDE={self.n_stride} is shorter than the {ph} rows "
            f"W2 computes -- experts would overlap")
        self.w13_out_stride = 2 * self.n_stride if self.n_stride else self.w13_out
        self.w2_out_stride = self.n_stride if self.n_stride else self.w2_out
        assert self.w13_out_stride % self.w13_opw == 0, \
            "W13 expert pitch is not a whole number of workgroups"
        assert self.w2_out_stride % self.w2_opw == 0, \
            "W2 expert pitch is not a whole number of workgroups"

        # ── alias vLLM's data tensors instead of packing our own ─────────────
        # Requires all three of the above to describe vLLM's buffer, since that
        # is the buffer the kernel would then be reading. The check is structural
        # (the knobs must be SET, and set consistently), not a comparison against
        # literal 3072s -- the padding rule belongs to vLLM, not to fleet_mk.
        self.alias = os.environ.get("FLEET_MK_MOE_ALIAS_VLLM") == "1"
        if self.alias:
            assert self.split_scales and self.n_stride and self.k_stride > ph, (
                "FLEET_MK_MOE_ALIAS_VLLM needs FLEET_MK_MOE_SPLIT_SCALES=1 plus "
                "FLEET_MK_MOE_K_STRIDE and FLEET_MK_MOE_N_STRIDE describing vLLM's "
                "padding (3072/3072 for GPT-OSS). Aliasing without them points "
                "the kernel at vLLM's memory while addressing it as fleet_mk's.")

    # ── packer arguments ─────────────────────────────────────────────────────
    def pack_kwargs(self, which):
        """Layout kwargs for `pack_mxfp4_workgroup`, for "w13" or "w2".

        `section` is deliberately NOT included: the caller decides whether it
        wants data, scales, or both, and under aliasing it asks for the two
        separately from the same kwargs.
        """
        assert which in ("w13", "w2"), which
        out_stride = self.w13_out_stride if which == "w13" else self.w2_out_stride
        computed = self.w13_out if which == "w13" else self.w2_out
        return dict(
            output_per_wg=self.w13_opw if which == "w13" else self.w2_opw,
            target_out_dim=computed,
            # W13 reduces over hidden, W2 over intermediate. They are the same
            # number here only because GPT-OSS pads both to padded_hidden_size;
            # `_moe_dims` in packing_moe.py makes the same identification, and a
            # model where they differ needs both this and the n_stride knob
            # split in two.
            target_num_blocks=self.hidden // 32,
            row_stride_blocks=self.k_stride_blocks,
            out_stride_rows=out_stride if self.n_stride else None,
            split_scales=self.split_scales,
        )

    def describe(self):
        """One line per knob, including the -D the .so must have been built with.

        Printed unconditionally. Three of these four knobs are compiled into the
        kernel and nothing verifies agreement at runtime, so the banner is the
        only place a mismatch is visible before it becomes wrong output.
        """
        ph = self.hidden
        lines = [
            f"MoE expert rows: stride={self.k_stride} values "
            f"({self.k_stride // 2} B), reduction={ph} values"
            + ("" if self.k_stride == ph else
               f"  <-- .so MUST be built -DMPK_MOE_K_STRIDE={self.k_stride}"),
            "MoE scales: "
            + ("SPLIT, own allocation  <-- .so MUST be built "
               "-DMPK_MOE_SPLIT_SCALES=1" if self.split_scales
               else "interleaved per workgroup"),
            f"MoE expert pitch: W13 {self.w13_out_stride} rows stored / "
            f"{self.w13_out} computed, W2 {self.w2_out_stride} / {self.w2_out}"
            + ("" if not self.n_stride else
               f"  <-- .so MUST be built -DMPK_MOE_N_STRIDE={self.n_stride}"),
            "MoE expert data: "
            + ("ALIASED from vLLM's w13_weight/w2_weight (not packed)"
               if self.alias else "packed by fleet_mk"),
        ]
        return "\n".join(lines)
