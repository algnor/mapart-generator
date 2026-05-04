import numpy as np
import numba as nb
from color import SHADED_LAB, rgb_to_oklab
import config

@nb.njit(cache=True, parallel=False)
def _dp_core(block_dists, trans_nh, trans_si, trans_hc, trans_count,
             BW, MAX_H, n_blocks):
    """
    block_dists: (N, 3, n_blocks) float32
    trans_nh/si/hc: (MAX_H+1, max_trans) int32/int32/float32
    trans_count:    (MAX_H+1,) int32  — valid entries per prev_h
    """
    N = block_dists.shape[0]
    n_hs = MAX_H + 1

    # Beam arrays
    beam_cost = np.full(BW, np.inf, dtype=np.float32)
    beam_h    = np.zeros(BW, dtype=np.int32)
    beam_b    = np.zeros(BW, dtype=np.int32)
    beam_s    = np.zeros(BW, dtype=np.int32)
    beam_par  = np.full(BW, -1, dtype=np.int32)

    # Row 0: free shade, all heights
    cur_bw = 0
    tmp_cost = np.full(n_hs * 3 * n_blocks, np.inf, dtype=np.float32)
    tmp_h    = np.zeros(n_hs * 3 * n_blocks, dtype=np.int32)
    tmp_b    = np.zeros(n_hs * 3 * n_blocks, dtype=np.int32)
    tmp_s    = np.zeros(n_hs * 3 * n_blocks, dtype=np.int32)
    tmp_par  = np.full(n_hs * 3 * n_blocks, -1, dtype=np.int32)

    idx = 0
    for h in range(n_hs):
        for s in range(3):
            for b in range(n_blocks):
                tmp_cost[idx] = block_dists[0, s, b]
                tmp_h[idx] = h
                tmp_b[idx] = b
                tmp_s[idx] = s
                idx += 1

    # partial sort: pick BW best
    cur_bw = min(BW, idx)
    # insertion-style top-k (BW is small ~48)
    for i in range(idx):
        if tmp_cost[i] < beam_cost[cur_bw - 1]:
            beam_cost[cur_bw - 1] = tmp_cost[i]
            beam_h   [cur_bw - 1] = tmp_h[i]
            beam_b   [cur_bw - 1] = tmp_b[i]
            beam_s   [cur_bw - 1] = tmp_s[i]
            beam_par [cur_bw - 1] = -1
            # bubble up
            j = cur_bw - 1
            while j > 0 and beam_cost[j] < beam_cost[j-1]:
                beam_cost[j], beam_cost[j-1] = beam_cost[j-1], beam_cost[j]
                beam_h   [j], beam_h   [j-1] = beam_h   [j-1], beam_h   [j]
                beam_b   [j], beam_b   [j-1] = beam_b   [j-1], beam_b   [j]
                beam_s   [j], beam_s   [j-1] = beam_s   [j-1], beam_s   [j]
                beam_par [j], beam_par [j-1] = beam_par [j-1], beam_par [j]
                j -= 1

    # Backtrack storage: (N, BW) for b, h, s, par
    all_b   = np.zeros((N, BW), dtype=np.int32)
    all_h   = np.zeros((N, BW), dtype=np.int32)
    all_s   = np.zeros((N, BW), dtype=np.int32)
    all_par = np.full((N, BW), -1, dtype=np.int32)
    all_bw  = np.zeros(N, dtype=np.int32)

    all_b  [0, :cur_bw] = beam_b  [:cur_bw]
    all_h  [0, :cur_bw] = beam_h  [:cur_bw]
    all_s  [0, :cur_bw] = beam_s  [:cur_bw]
    all_par[0, :cur_bw] = beam_par[:cur_bw]
    all_bw [0]          = cur_bw

    # Dense accumulator for transitions
    acc_cost = np.full((n_hs, 3, n_blocks), np.inf, dtype=np.float32)
    acc_par  = np.zeros((n_hs, 3, n_blocks), dtype=np.int32)

    max_trans = trans_nh.shape[1]

    for row in range(1, N):
        bd = block_dists[row]  # (3, n_blocks)

        # Reset accumulator
        for h in range(n_hs):
            for s in range(3):
                for b in range(n_blocks):
                    acc_cost[h, s, b] = np.inf

        # Fill accumulator
        for pi in range(cur_bw):
            pc   = beam_cost[pi]
            ph   = beam_h[pi]
            nt   = trans_count[ph]
            for ti in range(nt):
                nh = trans_nh[ph, ti]
                si = trans_si[ph, ti]
                hc = trans_hc[ph, ti]
                base = pc + hc
                for b in range(n_blocks):
                    c = base + bd[si, b]
                    if c < acc_cost[nh, si, b]:
                        acc_cost[nh, si, b] = c
                        acc_par [nh, si, b] = pi

        # Flatten & top-k into new beam
        new_cost = np.full(BW, np.inf, dtype=np.float32)
        new_h    = np.zeros(BW, dtype=np.int32)
        new_b    = np.zeros(BW, dtype=np.int32)
        new_s    = np.zeros(BW, dtype=np.int32)
        new_par  = np.zeros(BW, dtype=np.int32)
        new_bw   = 0

        for h in range(n_hs):
            for s in range(3):
                for b in range(n_blocks):
                    c = acc_cost[h, s, b]
                    if c == np.inf:
                        continue
                    if new_bw < BW:
                        # insert into sorted beam
                        pos = new_bw
                        new_cost[pos] = c
                        new_h   [pos] = h
                        new_b   [pos] = b
                        new_s   [pos] = s
                        new_par [pos] = acc_par[h, s, b]
                        new_bw += 1
                        # bubble up
                        j = pos
                        while j > 0 and new_cost[j] < new_cost[j-1]:
                            new_cost[j], new_cost[j-1] = new_cost[j-1], new_cost[j]
                            new_h   [j], new_h   [j-1] = new_h   [j-1], new_h   [j]
                            new_b   [j], new_b   [j-1] = new_b   [j-1], new_b   [j]
                            new_s   [j], new_s   [j-1] = new_s   [j-1], new_s   [j]
                            new_par [j], new_par [j-1] = new_par [j-1], new_par [j]
                            j -= 1
                    elif c < new_cost[new_bw - 1]:
                        new_cost[new_bw-1] = c
                        new_h   [new_bw-1] = h
                        new_b   [new_bw-1] = b
                        new_s   [new_bw-1] = s
                        new_par [new_bw-1] = acc_par[h, s, b]
                        j = new_bw - 1
                        while j > 0 and new_cost[j] < new_cost[j-1]:
                            new_cost[j], new_cost[j-1] = new_cost[j-1], new_cost[j]
                            new_h   [j], new_h   [j-1] = new_h   [j-1], new_h   [j]
                            new_b   [j], new_b   [j-1] = new_b   [j-1], new_b   [j]
                            new_s   [j], new_s   [j-1] = new_s   [j-1], new_s   [j]
                            new_par [j], new_par [j-1] = new_par [j-1], new_par [j]
                            j -= 1

        beam_cost = new_cost
        beam_h    = new_h
        beam_b    = new_b
        beam_s    = new_s
        beam_par  = new_par
        cur_bw    = new_bw

        all_b  [row, :cur_bw] = beam_b  [:cur_bw]
        all_h  [row, :cur_bw] = beam_h  [:cur_bw]
        all_s  [row, :cur_bw] = beam_s  [:cur_bw]
        all_par[row, :cur_bw] = beam_par[:cur_bw]
        all_bw [row]          = cur_bw

    return all_b, all_h, all_s, all_par, all_bw


def _build_trans_tables(MAX_H, MAX_S, height_penalty):
    max_trans = 2 * MAX_S + 1
    trans_nh    = np.zeros((MAX_H + 1, max_trans), dtype=np.int32)
    trans_si    = np.zeros((MAX_H + 1, max_trans), dtype=np.int32)
    trans_hc    = np.zeros((MAX_H + 1, max_trans), dtype=np.float32)
    trans_count = np.zeros(MAX_H + 1,              dtype=np.int32)
    for ph in range(MAX_H + 1):
        ti = 0
        for dh in range(-MAX_S, MAX_S + 1):
            nh = ph + dh
            if nh < 0 or nh > MAX_H:
                continue
            trans_nh[ph, ti] = nh
            trans_si[ph, ti] = 2 if nh > ph else (0 if nh < ph else 1)
            trans_hc[ph, ti] = abs(dh) * height_penalty
            ti += 1
        trans_count[ph] = ti
    return trans_nh, trans_si, trans_hc, trans_count


def solve_strip(strip: np.ndarray, height_penalty: float = 1.0):
    N        = strip.shape[0]
    n_blocks = SHADED_LAB.shape[0]
    BW       = config.BEAM_WIDTH
    MAX_H    = config.MAX_HEIGHT
    MAX_S    = config.MAX_STEP

    target_lab = rgb_to_oklab(strip)  # (N, 3)

    # (N, 3, n_blocks)
    t_exp = target_lab[:, None, None, :]
    pal   = SHADED_LAB.transpose(1, 0, 2)[None, :, :, :]  # (1, 3, n_blocks, 3)
    diff  = t_exp - pal
    diff[:, :, :, 1] *= config.CHROMA_WEIGHT
    diff[:, :, :, 2] *= config.CHROMA_WEIGHT
    block_dists = np.sqrt(np.einsum('nsbi,nsbi->nsb', diff, diff)).astype(np.float32)

    trans_nh, trans_si, trans_hc, trans_count = _build_trans_tables(MAX_H, MAX_S, height_penalty)

    all_b, all_h, all_s, all_par, all_bw = _dp_core(
        block_dists, trans_nh, trans_si, trans_hc, trans_count, BW, MAX_H, n_blocks
    )

    # Backtrack
    path   = []
    shades = []
    idx    = 0
    for row in range(N - 1, -1, -1):
        path.append((int(all_b[row, idx]), int(all_h[row, idx])))
        shades.append(int(all_s[row, idx]))
        idx = int(all_par[row, idx])

    path.reverse()
    shades.reverse()
    return path, float(block_dists[0, 0, 0]), shades[0]
