import numpy as np
import config
from color import SHADED_LAB, N_BLOCKS, oklab_dist_batch as dist_batch, rgb_to_oklab as rgb_to_lab

_DH_RANGE  = np.arange(-config.MAX_STEP, config.MAX_STEP + 1)
_DH_SHADES = np.where(_DH_RANGE > 0, 2, np.where(_DH_RANGE < 0, 0, 1)).astype(np.int8)
_N_DH      = len(_DH_RANGE)


def solve_strip(
    target_pixels:  np.ndarray,
    height_penalty: float = 2.0,
):
    L = target_pixels.shape[0]
    target_labs = rgb_to_lab(target_pixels.astype(np.float32))

    # ── position 0: try all 3 shades ─────────────────────────────────────────
    all_first_errors = np.stack([
        dist_batch(SHADED_LAB[:, s, :], target_labs[0]) for s in range(3)
    ])

    flat_errors = all_first_errors.ravel()
    top_flat    = np.argsort(flat_errors)[:config.BEAM_WIDTH]
    top_shades  = (top_flat // N_BLOCKS).astype(np.int8)
    top_b       = (top_flat  % N_BLOCKS).astype(np.int32)
    n           = len(top_b)

    beam_costs   = flat_errors[top_flat].astype(np.float32)
    beam_heights = np.full(n, config.MAX_HEIGHT // 2, dtype=np.int32)
    beam_paths   = np.zeros((n, L), dtype=np.int32)
    beam_hpaths  = np.full((n, L), config.MAX_HEIGHT // 2, dtype=np.int16)
    beam_paths[:, 0]  = top_b
    beam_hpaths[:, 0] = config.MAX_HEIGHT // 2
    beam_shades0      = top_shades.copy()

    # ── positions 1..L-1 ─────────────────────────────────────────────────────
    for i in range(1, L):
        current_lab  = target_labs[i]
        err_by_shade = np.stack([
            dist_batch(SHADED_LAB[:, s, :], current_lab) for s in range(3)
        ])                                                # (3, N_BLOCKS)

        new_heights = beam_heights[:, None] + _DH_RANGE[None, :]   # (n, D)
        valid       = (new_heights >= 0) & (new_heights <= config.MAX_HEIGHT)

        shade_costs = err_by_shade[_DH_SHADES]                      # (D, N_BLOCKS)

        # 1. penalize any height change (flat cost per step, not per unit)
        #    this reduces how often changes happen without biasing direction
        change_costs = (_DH_COSTS > 0).astype(np.float32) * height_penalty

        # 2. penalize proximity to boundaries (encourages using full range)
        #    normalize to [0,1]: 0 at center, 1 at boundary
        margin = np.minimum(new_heights, config.MAX_HEIGHT - new_heights).astype(np.float32)
        boundary_costs = (1.0 - margin / (config.MAX_HEIGHT / 2)) * height_penalty * 0.2

        cand = (beam_costs[:, None, None]
                + shade_costs[None, :, :]
                + change_costs[None, :, None]
                + boundary_costs[:, :, None])

        cand[~valid] = np.inf

        flat_costs   = cand.ravel()
        flat_heights = new_heights[:, :, None].repeat(N_BLOCKS, axis=2).ravel()
        flat_pb      = np.repeat(np.arange(n), _N_DH * N_BLOCKS)
        flat_bk      = np.tile(np.arange(N_BLOCKS), n * _N_DH)

        K    = min(config.BEAM_WIDTH, int(np.sum(valid)) * N_BLOCKS)
        topk = np.argpartition(flat_costs, K - 1)[:K]
        topk = topk[np.argsort(flat_costs[topk])]

        pb = flat_pb[topk]
        bk = flat_bk[topk]
        nh = flat_heights[topk].astype(np.int32)

        nb                = len(topk)
        new_paths         = np.zeros((nb, L), dtype=np.int32)
        new_hpaths        = np.zeros((nb, L), dtype=np.int16)
        new_paths[:, :i]  = beam_paths[pb, :i]
        new_hpaths[:, :i] = beam_hpaths[pb, :i]
        new_paths[:, i]   = bk
        new_hpaths[:, i]  = nh

        beam_costs   = flat_costs[topk]
        beam_heights = nh
        beam_paths   = new_paths
        beam_hpaths  = new_hpaths
        beam_shades0 = beam_shades0[pb]

    path        = list(zip(beam_paths[0].tolist(), beam_hpaths[0].astype(int).tolist()))
    first_shade = int(beam_shades0[0])
    return path, float(beam_costs[0]), first_shade
