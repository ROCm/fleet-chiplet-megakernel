# Step 4, batch 7: the last places the kernel names a pointer slot by number.
#
# Two shared-memory array sizes and two bare mirage_out[] subscripts. These are
# the same off-by-one hazard as the counts in batch 5, one level down: if
# MIRAGE_OUT is ever reordered, `mirage_out[10]` keeps compiling and starts
# zeroing the wrong buffer. mirage_out_idx() resolves them by mirage's own
# name, so a reorder moves the subscript with it.
#
# The two indices used here are the ResAdd path's inputs, which runs on the
# last layer across all 240 workers -- getting either wrong is silent garbage,
# not a crash.

SUBS = [
    ("    __shared__ void *mirage_in[24];\n"
     "    __shared__ void *mirage_out[11];",
     "    __shared__ void *mirage_in[{len(MIRAGE_IN)}];\n"
     "    __shared__ void *mirage_out[{len(MIRAGE_OUT)}];", 1),

    ("                    float *ws_f32 = (float *)mirage_out[10];",
     "                    float *ws_f32 = (float *)"
     "mirage_out[{mirage_out_idx('moe_workspace_f32')}];", 1),
    ("                    const unsigned short *oproj = "
     "(const unsigned short *)mirage_out[5];",
     "                    const unsigned short *oproj = "
     "(const unsigned short *)mirage_out[{mirage_out_idx('attn_proj_out')}];",
     1),
]
