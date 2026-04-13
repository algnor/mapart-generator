import numpy as np
from color import SHADED_LAB, SHADED_RGB, N_BLOCKS, cie94_batch, rgb_to_lab
# solver.py - change top import to:
import config


def solve_strip(
    target_pixels:  np.ndarray,
    height_penalty: float = 2.0,
    dither_strength:         float = 0.5
):
    """
    target_pixels : (L, 3) float32 — one strip along Z axis
    Shade rule    : current HIGHER than previous → brighter (shade 2)
                    current LOWER                → darker   (shade 0)
                    same                         → normal   (shade 1)
    Returns       : [(block_idx, height), ...], cost
    """
    L       = target_pixels.shape[0]
    targets = target_pixels.astype(np.float32).copy()

    # first element — no predecessor, shade 1
    shade      = 1
    target_lab = rgb_to_lab(targets[0:1])[0]
    errors     = cie94_batch(SHADED_LAB[:, shade, :], target_lab)
    top_b      = np.argsort(errors)[:config.BEAM_WIDTH]
    n          = len(top_b)

    beam_costs    = errors[top_b].astype(np.float32)
    beam_heights  = np.zeros(n, dtype=np.int32)
    beam_paths    = np.zeros((n, L), dtype=np.int32)
    beam_hpaths   = np.zeros((n, L), dtype=np.int8)
    beam_rendered = SHADED_RGB[top_b, shade, :].astype(np.float32)
    beam_paths[:, 0] = top_b

    for i in range(1, L):
        if dither_strength > 0:
            residual      = targets[i-1] - beam_rendered[0]
            targets[i]    = np.clip(targets[i]   + residual * dither_strength,       0, 255)
            if i + 1 < L:
                targets[i+1] = np.clip(targets[i+1] + residual * dither_strength * 0.5, 0, 255)

        target_lab   = rgb_to_lab(targets[i:i+1])[0]
        err_by_shade = np.stack([
            cie94_batch(SHADED_LAB[:, s, :], target_lab) for s in range(3)
        ])  # (3, N_BLOCKS)

        all_costs    = []
        all_heights  = []
        all_rendered = []
        all_pb       = []
        all_bk       = []

        for di, dh in enumerate((-1, 0, 1)):
            shade = di
            new_h = beam_heights + dh
            valid = (new_h >= 0) & (new_h <= config.MAX_HEIGHT_DIFF)
            vi    = np.where(valid)[0]
            if not vi.size:
                continue

            e      = err_by_shade[shade]
            h_cost = height_penalty * abs(dh)
            cand   = beam_costs[vi, None] + e[None, :] + h_cost

            V  = len(vi)
            pb = np.repeat(vi, N_BLOCKS)
            bk = np.tile(np.arange(N_BLOCKS), V)

            all_costs.append(cand.ravel())
            all_heights.append(np.repeat(new_h[vi], N_BLOCKS))
            all_rendered.append(SHADED_RGB[bk, shade, :])
            all_pb.append(pb)
            all_bk.append(bk)

        merged_costs    = np.concatenate(all_costs)
        merged_heights  = np.concatenate(all_heights)
        merged_rendered = np.concatenate(all_rendered, axis=0)
        merged_pb       = np.concatenate(all_pb)
        merged_bk       = np.concatenate(all_bk)

        K    = min(config.BEAM_WIDTH, len(merged_costs))
        topk = np.argpartition(merged_costs, K-1)[:K]
        topk = topk[np.argsort(merged_costs[topk])]

        nb = len(topk)
        pb = merged_pb[topk]
        bk = merged_bk[topk]
        nh = merged_heights[topk].astype(np.int32)

        new_paths         = np.zeros((nb, L), dtype=np.int32)
        new_hpaths        = np.zeros((nb, L), dtype=np.int8)
        new_paths[:, :i]  = beam_paths[pb, :i]
        new_hpaths[:, :i] = beam_hpaths[pb, :i]
        new_paths[:, i]   = bk
        new_hpaths[:, i]  = nh

        beam_costs    = merged_costs[topk]
        beam_heights  = nh
        beam_paths    = new_paths
        beam_hpaths   = new_hpaths
        beam_rendered = merged_rendered[topk]

    path = list(zip(beam_paths[0].tolist(),
                    beam_hpaths[0].astype(int).tolist()))
    return path, float(beam_costs[0])
