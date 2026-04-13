import numpy as np
import config
from color import SHADED_LAB, N_BLOCKS, cie94_batch, cie94_batch_multi, rgb_to_lab

_DH_RANGE  = np.arange(-config.MAX_STEP, config.MAX_STEP + 1)
_DH_SHADES = np.where(_DH_RANGE > 0, 2, np.where(_DH_RANGE < 0, 0, 1)).astype(np.int8)
_DH_COSTS  = np.abs(_DH_RANGE).astype(np.float32)
_N_DH      = len(_DH_RANGE)


def solve_strip(
    target_pixels:   np.ndarray,   # (L, 3) float32 RGB, index 0 = Z=0 in schematic
    height_penalty:  float = 2.0,
    dither_strength: float = 0.3,
    per_beam_dither: bool  = True,
):
    """
    Solves one vertical strip top-to-bottom (index 0 = map north = schematic Z=0).
    Returns:
        path        : [(block_idx, height), ...] length L, index 0 = Z=0
        cost        : float
        first_shade : int (0/1/2) — shade that Z=0 should render with,
                      used to compute the sacrificial block height in export
    """
    L = target_pixels.shape[0]
    target_labs = rgb_to_lab(target_pixels.astype(np.float32))

    # ── position 0: try all 3 shades ─────────────────────────────────────────
    current_lab      = target_labs[0]
    all_first_errors = np.stack([
        cie94_batch(SHADED_LAB[:, s, :], current_lab) for s in range(3)
    ])                                                   # (3, N_BLOCKS)

    flat_errors = all_first_errors.ravel()               # (3*N_BLOCKS,)
    top_flat    = np.argsort(flat_errors)[:config.BEAM_WIDTH]
    top_shades  = (top_flat // N_BLOCKS).astype(np.int8)
    top_b       = (top_flat  % N_BLOCKS).astype(np.int32)
    n           = len(top_b)

    beam_costs   = flat_errors[top_flat].astype(np.float32)
    beam_heights = np.zeros(n, dtype=np.int32)
    beam_paths   = np.zeros((n, L), dtype=np.int32)
    beam_hpaths  = np.zeros((n, L), dtype=np.int16)
    beam_paths[:, 0]  = top_b
    beam_hpaths[:, 0] = 0
    beam_shades0      = top_shades.copy()                # shade at position 0, per beam

    if per_beam_dither and dither_strength > 0:
        chosen_labs = SHADED_LAB[top_b, top_shades, :]
        beam_dither = (current_lab - chosen_labs) * dither_strength  # (n, 3)
        dither_err  = np.zeros(3, dtype=np.float32)
    else:
        beam_dither = np.zeros((n, 3), dtype=np.float32)
        best_lab    = SHADED_LAB[top_b[0], top_shades[0]]
        dither_err  = (current_lab - best_lab) * dither_strength if dither_strength > 0 else np.zeros(3, dtype=np.float32)

    # ── positions 1..L-1 ─────────────────────────────────────────────────────
    for i in range(1, L):

        if per_beam_dither and dither_strength > 0:
            current_labs = target_labs[i] + beam_dither             # (n, 3)
            err_by_shade = np.stack([
                cie94_batch_multi(SHADED_LAB[:, s, :], current_labs)
                for s in range(3)
            ])                                                       # (3, n, N_BLOCKS)
        else:
            current_lab  = target_labs[i] + dither_err
            err_by_shade = np.stack([
                cie94_batch(SHADED_LAB[:, s, :], current_lab)
                for s in range(3)
            ])                                                       # (3, N_BLOCKS)

        if not (per_beam_dither and dither_strength > 0):
            # ── fast vectorised path ─────────────────────────────────────────
            new_heights = beam_heights[:, None] + _DH_RANGE[None, :]  # (n, D)
            valid       = (new_heights >= 0) & (new_heights <= config.MAX_HEIGHT)

            shade_costs = err_by_shade[_DH_SHADES]                    # (D, N_BLOCKS)
            h_costs     = _DH_COSTS * height_penalty                  # (D,)

            cand = (beam_costs[:, None, None]
                    + shade_costs[None, :, :]
                    + h_costs[None, :, None])                          # (n, D, N_BLOCKS)
            cand[~valid] = np.inf

            flat_costs   = cand.ravel()
            flat_heights = np.broadcast_to(
                new_heights[:, :, None], (n, _N_DH, N_BLOCKS)
            ).ravel()
            flat_pb    = np.repeat(np.arange(n), _N_DH * N_BLOCKS)
            flat_bk    = np.tile(np.arange(N_BLOCKS), n * _N_DH)
            flat_shade = np.tile(np.repeat(_DH_SHADES, N_BLOCKS), n)

            K    = min(config.BEAM_WIDTH, int(np.sum(valid)) * N_BLOCKS)
            topk = np.argpartition(flat_costs, K - 1)[:K]
            topk = topk[np.argsort(flat_costs[topk])]

            pb = flat_pb[topk]
            bk = flat_bk[topk]
            nh = flat_heights[topk].astype(np.int32)
            sk = flat_shade[topk]

        else:
            # ── per-beam path ─────────────────────────────────────────────────
            all_costs   = []
            all_heights = []
            all_pb      = []
            all_bk      = []
            all_shade   = []

            for dh in range(-config.MAX_STEP, config.MAX_STEP + 1):
                shade = 2 if dh > 0 else (0 if dh < 0 else 1)
                new_h = beam_heights + dh
                valid = (new_h >= 0) & (new_h <= config.MAX_HEIGHT)
                vi    = np.where(valid)[0]
                if not vi.size:
                    continue

                e    = err_by_shade[shade, vi, :]
                cand = beam_costs[vi, None] + e + height_penalty * abs(dh)

                V = len(vi)
                all_costs.append(cand.ravel())
                all_heights.append(np.repeat(new_h[vi], N_BLOCKS))
                all_pb.append(np.repeat(vi, N_BLOCKS))
                all_bk.append(np.tile(np.arange(N_BLOCKS), V))
                all_shade.append(np.full(V * N_BLOCKS, shade, dtype=np.int8))

            flat_costs     = np.concatenate(all_costs)
            merged_heights = np.concatenate(all_heights)
            merged_pb      = np.concatenate(all_pb)
            merged_bk      = np.concatenate(all_bk)
            merged_shade   = np.concatenate(all_shade)

            K    = min(config.BEAM_WIDTH, len(flat_costs))
            topk = np.argpartition(flat_costs, K - 1)[:K]
            topk = topk[np.argsort(flat_costs[topk])]

            pb = merged_pb[topk]
            bk = merged_bk[topk]
            nh = merged_heights[topk].astype(np.int32)
            sk = merged_shade[topk]

        # ── update beam state ─────────────────────────────────────────────────
        nb               = len(topk)
        new_paths        = np.zeros((nb, L), dtype=np.int32)
        new_hpaths       = np.zeros((nb, L), dtype=np.int16)
        new_paths[:, :i]  = beam_paths[pb, :i]
        new_hpaths[:, :i] = beam_hpaths[pb, :i]
        new_paths[:, i]   = bk
        new_hpaths[:, i]  = nh

        beam_costs   = flat_costs[topk]
        beam_heights = nh
        beam_paths   = new_paths
        beam_hpaths  = new_hpaths
        beam_shades0 = beam_shades0[pb]

        if dither_strength > 0:
            if per_beam_dither:
                chosen_labs = SHADED_LAB[bk, sk, :]
                beam_dither = (current_labs[pb] - chosen_labs) * dither_strength
            else:
                best_lab   = SHADED_LAB[int(bk[0]), int(sk[0])]
                dither_err = (current_lab - best_lab) * dither_strength
        elif not per_beam_dither:
            dither_err = np.zeros(3, dtype=np.float32)

    path        = list(zip(beam_paths[0].tolist(), beam_hpaths[0].astype(int).tolist()))
    first_shade = int(beam_shades0[0])
    return path, float(beam_costs[0]), first_shade
