import numpy as np

from ragsgo_exp.online import online_align


def _smooth_random_sequence(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((12, n))
    kernel = np.ones(3) / 3  # neighbouring frames look alike, as in music
    x = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, x)
    return x / np.linalg.norm(x, axis=0, keepdims=True)


def _played_at(ref, rate, seed=1):
    """Live frame i shows ref frame round(i * rate), plus a little noise."""
    rng = np.random.default_rng(seed)
    n_live = int((ref.shape[1] - 1) / rate)
    idx = np.round(np.arange(n_live) * rate).astype(int)
    live = ref[:, idx] + 0.05 * rng.random((12, n_live))
    live /= np.linalg.norm(live, axis=0, keepdims=True)
    return live, idx


def test_tracks_slower_and_faster_playing():
    ref = _smooth_random_sequence(300)
    for rate in (0.8, 1.25):
        live, true_idx = _played_at(ref, rate)
        est = online_align(live, ref)
        err = np.abs(est - true_idx)
        assert np.percentile(err, 95) <= 2, (rate, float(np.percentile(err, 95)))


def test_is_causal():
    ref = _smooth_random_sequence(300)
    live, _ = _played_at(ref, 1.0)
    rng = np.random.default_rng(5)
    live2 = live.copy()
    tail = rng.random((12, live.shape[1] - 150))
    live2[:, 150:] = tail / np.linalg.norm(tail, axis=0, keepdims=True)
    est1 = online_align(live, ref)
    est2 = online_align(live2, ref)
    np.testing.assert_array_equal(est1[:150], est2[:150])


def test_silent_frames_do_not_move_the_estimate_much():
    ref = _smooth_random_sequence(300)
    live, true_idx = _played_at(ref, 1.0)
    silent = np.zeros(live.shape[1], dtype=bool)
    silent[100:110] = True  # one second of silence in the middle
    est = online_align(live, ref, silent=silent)
    after = slice(130, 280)
    assert np.percentile(np.abs(est[after] - true_idx[after]), 95) <= 2
