import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from titan_vllm.mxfp4_pack import pack_mxfp4_workgroup

E, OUT, NB, OPW = 4, 5760, 90, 128        # w13: 2*2880 rows, 2880/32 blocks
TGT_OUT, TGT_NB = 5888, 92                # titan pads to 2944
K_STRIDE_BLK, N_STRIDE_ROWS = 96, 6144    # the K+N config

blocks = torch.zeros(E, OUT, NB, 16, dtype=torch.uint8)
scales = torch.zeros(E, OUT, NB, dtype=torch.uint8)

for label, kstr, nstr in [("base", None, None), ("K only", K_STRIDE_BLK, None),
                          ("N only", None, N_STRIDE_ROWS), ("K+N", K_STRIDE_BLK, N_STRIDE_ROWS)]:
    t = pack_mxfp4_workgroup(blocks, scales, output_per_wg=OPW,
                             target_out_dim=TGT_OUT, target_num_blocks=TGT_NB,
                             row_stride_blocks=kstr, out_stride_rows=nstr,
                             split_scales=True)
    # kernel-side arithmetic (mirrors the .cuh)
    n_stride   = nstr if nstr else TGT_OUT
    wg_data    = OPW * ((kstr or TGT_NB) * 16)
    wg_scale   = OPW * TGT_NB
    exp_bytes  = (n_stride // OPW) * wg_data          # W13_EXPERT_BYTES (descriptor extent)
    exp_sc     = (TGT_OUT // OPW) * wg_scale          # W13_EXPERT_SCALE_BYTES
    need       = E * exp_bytes + E * exp_sc
    print(f"{label:8s} packed={t.numel():>12,}  kernel_needs={need:>12,}  "
          f"delta={t.numel()-need:>+12,}  extent=0x{exp_bytes:x}")
