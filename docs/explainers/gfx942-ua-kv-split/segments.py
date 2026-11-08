"""Launch shape of aiter's gfx942 unified-attention split-KV (3D) path, default config vs this PR.

Mirrors the triton path in aiter/ops/triton/attention/unified_attention.py (use_2d_kernel, total_num_q_blocks)
and compute_segment_params in aiter/ops/triton/utils/unified_attention_utils.py, without a GPU.
Shapes are one TP4 rank of Qwen3.5-397B-A17B: 8 query heads, 1 KV head, head_size 256, bf16 KV, page size 16.

    python segments.py                 # tables used in the explainer
    python segments.py --ctx 65536     # another context length
"""

import argparse


def cdiv(a, b):
    return -(-a // b)


def next_pow2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


# kv_split entries that the lookup lands on for D=256, bf16 KV
DEFAULT = {"verify": dict(MIN_SEGMENTS=8, SEGMENTS_PER_CU=16, MAX_SEGMENTS=128, SEGMENT_TILE_MAX=64),
           "draft": dict(MIN_SEGMENTS=8, SEGMENTS_PER_CU=16, MAX_SEGMENTS=128, SEGMENT_TILE_MAX=64)}
PR = {"verify": dict(MIN_SEGMENTS=8, SEGMENTS_PER_CU=4, MAX_SEGMENTS=512, SEGMENT_TILE_MIN=256, SEGMENT_TILE_MAX=256),
      "draft": dict(MIN_SEGMENTS=8, SEGMENTS_PER_CU=2, MAX_SEGMENTS=512, SEGMENT_TILE_MIN=256, SEGMENT_TILE_MAX=256)}

CUS = 304          # MI300X and MI325X
HQ, HKV = 8, 1     # query and KV heads per GPU
D = 256
PAGE = 16
TILE = 16          # kernel TILE_SIZE: next_pow2(page) clamped to TILE_SIZE_MIN/MAX (1..64), same in both configs
NUM_WARPS = 2      # attn_3d "any" on gfx942


def launch(cfg, bs, qlen, ctx):
    qpk = HQ // HKV
    block_m = 16 if qpk <= 16 else next_pow2(qpk)
    block_q = block_m // qpk
    tokens = bs * qlen
    q_blocks = bs if qlen == 1 else tokens // block_q + bs
    prgms_2d = q_blocks * HKV
    if ctx <= 512 or prgms_2d > CUS * 4:
        return None  # takes the 2D kernel, no split
    tile = max(cfg.get("SEGMENT_TILE_MIN", 1), min(cfg.get("SEGMENT_TILE_MAX", 1 << 30), next_pow2(PAGE)))
    limit = min(cfg["MAX_SEGMENTS"], cdiv(ctx, tile))
    share = cdiv(CUS * cfg["SEGMENTS_PER_CU"], max(1, prgms_2d))
    seg = next_pow2(max(min(cfg["MIN_SEGMENTS"], limit), min(limit, max(1, share))))
    tiles_per_seg = cdiv(ctx, seg * TILE)
    real_q_blocks = bs * cdiv(qlen, block_q)
    return dict(seg=seg, grid=q_blocks * HKV * seg, live=real_q_blocks * HKV * seg,
                tiles=tiles_per_seg, reduce_mb=tokens * HQ * seg * D * 4 / 1e6 if seg > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[65536, 131072, 196608])
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    a = ap.parse_args()
    for ctx in a.ctx:
        kv_mb = 2 * ctx * HKV * D * 2 / 1e6
        print(f"\ncontext {ctx:,} tokens: K+V per sequence {kv_mb:.0f} MB")
        print(f"{'call':6} {'bs':>3} | {'default: seg  live waves  tiles/seg  partials MB':>46} | {'PR: seg  live waves  tiles/seg  partials MB':>41}")
        for call, qlen in (("draft", 1), ("verify", 4)):
            for bs in a.bs:
                row = []
                for table in (DEFAULT, PR):
                    r = launch(table[call], bs, qlen, ctx)
                    row.append("2D kernel" if r is None else
                               f"{r['seg']:4} {r['live'] * NUM_WARPS:7} {r['tiles']:8} {r['reduce_mb']:10.1f}")
                print(f"{call:6} {bs:3} | {row[0]:>46} | {row[1]:>41}")


if __name__ == "__main__":
    main()
