# LDS budget for the MoE prefetch tiles at each K stride.
# Mirrors gang_moe_fused_mxfp4_mi300.cuh: TILE_DATA -> LPT -> TILE_DATA_PADDED -> TILE_BYTES.
NUM_WAVES, LDS_CAP = 4, 155*1024
K_REDUCE = 2944

def tiles(k_stride, tile_rows):
    tile_data = tile_rows * (k_stride // 2)
    n16 = tile_data // 16
    lpt = (n16 + 255) // 256
    padded = lpt * 256 * 16
    return lpt, padded + tile_rows * (K_REDUCE // 32)

for k in (2944, 3072):
    # FP4 activations: LDS_W2_OFF = round16(K/2 + K/32)
    w13_off = ((K_REDUCE//2 + K_REDUCE//32 + 15)//16)*16
    w2_off  = w13_off
    l13, b13 = tiles(k, 16)
    l2,  b2  = tiles(k, 16)
    tot13 = w13_off + b13*NUM_WAVES
    tot2  = w2_off  + b2*NUM_WAVES
    print(f"K={k}: W13 LPT={l13} TILE_BYTES={b13:>7,} total={tot13:>7,} "
          f"{'OK ' if tot13<=LDS_CAP else 'OVER'} | "
          f"W2 LPT={l2} TILE_BYTES={b2:>7,} total={tot2:>7,} "
          f"{'OK ' if tot2<=LDS_CAP else 'OVER'}  (cap {LDS_CAP:,})")
