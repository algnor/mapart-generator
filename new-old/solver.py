import numpy as np
import config
from color import SHADED_LAB, SHADED_RGB, N_BLOCKS, cie94_batch, rgb_to_lab


def _shade_for_dh(dh: int) -> int:
    if dh > 0: return 2
    if dh < 0: return 0
    return 1


def solve_strip(
    target_pixels:  np.ndarray,
    height_penalty: float = 2.0,
    dither_strength: float = 0.3,
):
    """
    target_pixels  : (L, 3) float32 RGB
    Returns        : [(block_idx, height), ...], cost
    """
    L = target_pixels.shape[0]

    # convert full strip to LAB once
    target_labs = rgb_to_lab(target_pixels.astype(np.float32))  # (L, 3)
    dither_err  = np.zeros(3, dtype=np.float32)

    # ── first position ───────────────────────────────────────────────────────
    shade       = 1
    current_lab = target_labs[0] + dither_err
    errors      = cie94_batch(SHADED_LAB[:, shade, :], current_lab)
    top_b       = np.argsort(errors)[:config.BEAM_WIDTH]
    n           = len(top_b)

    beam_costs   = errors[top_b].astype(np.float32)
    beam_heights = np.zeros(n, dtype=np.int32)
    beam_paths   = np.zeros((n, L), dtype=np.int32)
    beam_hpaths  = np.zeros((n, L), dtype=np.int8)
    beam_paths[:, 0] = top_b

    if dither_strength > 0:
        best_lab   = SHADED_LAB[top_b[0], shade]
        dither_err = (current_lab - best_lab) * dither_strength

    # ── subsequent positions ─────────────────────────────────────────────────
    for i in range(1, L):
        current_lab  = target_labs[i] + dither_err
        err_by_shade = np.stack([
            cie94_batch(SHADED_LAB[:, s, :], current_lab) for s in range(3)
        ])  # (3, N_BLOCKS)

        all_costs   = []
        all_heights = []
        all_pb      = []
        all_bk      = []

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
            all_pb.append(pb)
            all_bk.append(bk)

        merged_costs   = np.concatenate(all_costs)
        merged_heights = np.concatenate(all_heights)
        merged_pb      = np.concatenate(all_pb)
        merged_bk      = np.concatenate(all_bk)

        K    = min(config.BEAM_WIDTH, len(merged_costs))
        topk = np.argpartition(merged_costs, K - 1)[:K]
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

        beam_costs   = merged_costs[topk]
        beam_heights = nh
        beam_paths   = new_paths
        beam_hpaths  = new_hpaths

        if dither_strength > 0:
            best_block  = int(bk[0])
            best_shade  = _shade_for_dh(int(nh[0]) - int(new_hpaths[0, i - 1]))
            best_lab    = SHADED_LAB[best_block, best_shade]
            dither_err  = (current_lab - best_lab) * dither_strength

    path = list(zip(beam_paths[0].tolist(),
                    beam_hpaths[0].astype(int).tolist()))
    return path, float(beam_costs[0])