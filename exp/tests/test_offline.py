import numpy as np

from ragsgo_exp.offline import subseq_dtw, subseq_dtw_slope


def _seq(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.random((12, n))
    return x / np.linalg.norm(x, axis=0, keepdims=True)


def test_finds_reference_inside_longer_recording():
    ref = _seq(100, 0)
    live = np.concatenate([_seq(40, 1), ref, _seq(30, 2)], axis=1)
    wp, cost = subseq_dtw(ref, live)
    assert wp[0, 0] == 0 and wp[-1, 0] == 99
    assert abs(wp[0, 1] - 40) <= 1 and abs(wp[-1, 1] - 139) <= 1
    assert cost < 0.05


def test_handles_slower_playing():
    ref = _seq(100, 0)
    idx = np.round(np.arange(0, 100, 0.8)).astype(int)
    idx = idx[idx < 100]  # live is 1.25x longer than ref
    live = np.concatenate([_seq(20, 3), ref[:, idx], _seq(20, 4)], axis=1)
    wp, _ = subseq_dtw(ref, live)
    assert abs(wp[0, 1] - 20) <= 1
    assert abs(wp[-1, 1] - (20 + len(idx) - 1)) <= 1


def test_path_moves_by_unit_steps_forward():
    wp, _ = subseq_dtw(_seq(50, 0), _seq(80, 1))
    steps = np.diff(wp, axis=0)
    assert np.all(steps >= 0) and np.all(steps <= 1) and np.all(steps.sum(axis=1) >= 1)


def test_slope_constrained_path_uses_only_allowed_steps():
    wp, _ = subseq_dtw_slope(_seq(50, 0), _seq(120, 1))
    steps = {tuple(int(v) for v in s) for s in np.diff(wp, axis=0)}
    assert steps <= {(1, 1), (1, 2), (2, 1)}
    assert wp[0, 0] == 0 and wp[-1, 0] == 49


def test_slope_constrained_finds_copy_played_slower():
    ref = _seq(100, 0)
    idx = np.round(np.arange(0, 100, 0.8)).astype(int)
    idx = idx[idx < 100]
    live = np.concatenate([_seq(20, 3), ref[:, idx], _seq(20, 4)], axis=1)
    wp, cost = subseq_dtw_slope(ref, live)
    assert abs(wp[0, 1] - 20) <= 1
    assert abs(wp[-1, 1] - (20 + len(idx) - 1)) <= 1
    assert cost < 0.05
