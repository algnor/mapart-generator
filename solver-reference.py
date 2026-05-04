from color import SHADED_LAB, rgb_to_oklab, oklab_dist_batch_multi
import config
import numpy as np

def solve_strip(strip: np.ndarray, height_penalty: float = 1.0):
    """
    strip: (N, 3) float32 RGB, N = map_rows * 128
    Returns: path [(block_idx, height), ...], total_cost, first_shade
    """
    N = strip.shape[0]
    n_blocks = SHADED_LAB.shape[0]  # (N_BLOCKS, 3, 3)
    beam_width = config.BEAM_WIDTH
    max_height = config.MAX_HEIGHT
    max_step   = config.MAX_STEP

    # Precompute target OKLab for each row
    target_lab = rgb_to_oklab(strip)  # (N, 3)

    # State: (height, beam_entry) -> (block_idx, shade, cost, parent)
    # Beam: list of (cost, height, block_idx, shade, parent_idx)

    # For row 0, we don't have a previous height, so shade is determined freely.
    # We'll enumerate all possible first heights (0..max_height) and all shades.

    # Initialize beam for row 0
    # Compute distances for row 0: target vs all (block, shade) combos
    t0 = target_lab[0]  # (3,)
    t0_rep = np.tile(t0, (n_blocks * 3, 1))  # (n_blocks*3, 3)
    palette = SHADED_LAB.reshape(n_blocks * 3, 3)  # (n_blocks*3, 3)

    from color import oklab_dist_batch
    dists = oklab_dist_batch(t0_rep, palette)  # (n_blocks*3,)
    dists = dists.reshape(n_blocks, 3)  # (n_blocks, shade)

    # beam entries: list of [cost, height, block_idx, shade_idx, back_ptr]
    # back_ptr is index into previous beam
    beam = []
    for h in range(max_height + 1):
        for bi in range(n_blocks):
            for si in range(3):
                c = dists[bi, si]
                beam.append([c, h, bi, si, -1])

    # Keep only beam_width best
    beam.sort(key=lambda x: x[0])
    beam = beam[:beam_width]

    # Store all beam states per row for backtracking
    all_beams = [beam]

    for row in range(1, N):
        t = target_lab[row]  # (3,)
        # For each candidate in new beam, we extend from previous beam
        # Transition: new_h can differ from prev_h by at most max_step
        # shade is determined by height comparison:
        #   new_h > prev_h -> shade 2 (bright)
        #   new_h < prev_h -> shade 0 (dark)
        #   new_h == prev_h -> shade 1 (medium)

        prev_beam = all_beams[-1]

        # Collect all (new_h, block, shade) candidates with best cost
        # candidate_map: (new_h, bi, si) -> (best_cost, parent_idx)
        # Since shade is fixed by height transition, si is determined.
        
        # Build list of all transitions
        # For efficiency, vectorize over blocks
        
        candidates = {}  # key=(new_h, bi) -> [cost, parent_idx]

        for pi, (prev_cost, prev_h, prev_bi, prev_si, _) in enumerate(prev_beam):
            for dh in range(-max_step, max_step + 1):
                new_h = prev_h + dh
                if new_h < 0 or new_h > max_height:
                    continue
                # shade determined by height change
                if new_h > prev_h:
                    si = 2
                elif new_h < prev_h:
                    si = 0
                else:
                    si = 1

                h_cost = abs(dh) * height_penalty

                # dist for all blocks at this shade
                # we'll compute per block
                key = (new_h, si)
                if key not in candidates:
                    # compute block dists for this shade
                    palette_si = SHADED_LAB[:, si, :]  # (n_blocks, 3)
                    t_rep = np.tile(t, (n_blocks, 1))
                    block_dists = oklab_dist_batch(t_rep, palette_si)  # (n_blocks,)
                    candidates[key] = {
                        'block_dists': block_dists,
                        'best': {}  # bi -> (cost, parent_idx)
                    }

                block_dists = candidates[key]['block_dists']
                best_map = candidates[key]['best']

                for bi in range(n_blocks):
                    total = prev_cost + h_cost + block_dists[bi]
                    if bi not in best_map or total < best_map[bi][0]:
                        best_map[bi] = (total, pi)

        # Flatten candidates into beam
        new_beam = []
        for (new_h, si), data in candidates.items():
            for bi, (cost, pi) in data['best'].items():
                new_beam.append([cost, new_h, bi, si, pi])

        new_beam.sort(key=lambda x: x[0])
        new_beam = new_beam[:beam_width]
        all_beams.append(new_beam)

    # Backtrack from best end state
    best = all_beams[-1][0]
    total_cost = best[0]

    path = []
    row = N - 1
    idx = 0
    while row >= 0:
        entry = all_beams[row][idx]
        path.append((entry[2], entry[1]))  # (block_idx, height)
        idx = entry[4]
        row -= 1

    path.reverse()
    first_shade = all_beams[0][0][3]  # shade of first row in best path... need to trace

    # Re-trace to get first shade correctly
    # Retrace again properly
    path = []
    row = N - 1
    idx = 0
    shades = []
    while row >= 0:
        entry = all_beams[row][idx]
        path.append((entry[2], entry[1]))
        shades.append(entry[3])
        idx = entry[4]
        row -= 1

    path.reverse()
    shades.reverse()
    first_shade = shades[0]

    return path, total_cost, first_shade