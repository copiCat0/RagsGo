import numpy as np

from ragsgo_exp import features as F


def _sine(freq, dur=2.0, sr=F.SR):
    t = np.arange(int(dur * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_chroma_peak_is_A_for_440hz():
    feats = F.chroma_features(_sine(440.0), causal=True)
    mean_chroma = feats.chroma[:, ~feats.silent].mean(axis=1)
    assert int(np.argmax(mean_chroma)) == 9  # C=0, ..., A=9


def test_causal_hpss_ignores_future_frames():
    rng = np.random.default_rng(0)
    S = rng.random((64, 200)).astype(np.float32)
    S2 = S.copy()
    S2[:, 120:] = rng.random((64, 80))
    H1 = F.harmonic_part(S, causal=True)
    H2 = F.harmonic_part(S2, causal=True)
    np.testing.assert_allclose(H1[:, :120], H2[:, :120])


def test_centered_hpss_does_use_future_frames():
    # Guards the test above: the centred filter must react to later frames.
    rng = np.random.default_rng(0)
    S = rng.random((64, 200)).astype(np.float32)
    S2 = S.copy()
    S2[:, 120:] = rng.random((64, 80))
    H1 = F.harmonic_part(S, causal=False)
    H2 = F.harmonic_part(S2, causal=False)
    assert not np.allclose(H1[:, :120], H2[:, :120])


def _chord(midis, amps=None, dur=2.0, sr=F.SR):
    t = np.arange(int(dur * sr)) / sr
    amps = amps or [1.0] * len(midis)
    y = sum(a * np.sin(2 * np.pi * 440.0 * 2 ** ((m - 69) / 12) * t) for m, a in zip(midis, amps))
    return (0.3 * y / len(midis)).astype(np.float32)


def _mean_chroma(y):
    f = F.chroma_features(y, causal=False, hpss=False)
    v = f.chroma[:, ~f.silent].mean(axis=1)
    return v / np.linalg.norm(v)


def test_same_chord_in_another_voicing_is_similar():
    a = _mean_chroma(_chord([60, 64, 67]))                          # C major, C4 E4 G4
    b = _mean_chroma(_chord([48, 64, 67, 72], [1.0, 0.5, 0.8, 0.6]))  # C major, other octaves and levels
    assert float(a @ b) > 0.8


def test_unrelated_chords_are_dissimilar():
    c_major = _mean_chroma(_chord([60, 64, 67]))
    fs_major = _mean_chroma(_chord([66, 70, 73]))  # F# major: no shared pitch class
    assert float(c_major @ fs_major) < 0.3


def test_pooled_chroma_is_unit_norm_at_about_10hz():
    feats = F.chroma_features(_sine(261.63, dur=3.0), causal=True)
    n = feats.chroma.shape[1]
    assert 25 <= n <= 31
    norms = np.linalg.norm(feats.chroma[:, ~feats.silent], axis=0)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_frame_chroma_pooled_matches_chroma_features():
    from ragsgo_exp import beats as B
    rng = np.random.default_rng(0)
    t = np.arange(int(6 * F.SR)) / F.SR
    y = (np.sin(2 * np.pi * 440 * t) * (t < 3) + np.sin(2 * np.pi * 392 * t) * (t >= 3)
         + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    feats = F.chroma_features(y, causal=True, hpss=True)
    fc, ft, fe = F.frame_chroma(y, causal=True, hpss=True)
    K = feats.chroma.shape[1]
    pooled = B.pool_between(fc, ft, np.arange(K + 1) / F.FEAT_RATE)
    has = np.bincount(np.floor(ft * F.FEAT_RATE).astype(int), minlength=K)[:K] > 0
    assert np.allclose(pooled[:, has], feats.chroma[:, has], atol=1e-5)


def test_frame_chroma_with_running_level_is_fully_causal():
    rng = np.random.default_rng(3)
    t = np.arange(int(8 * F.SR)) / F.SR
    y = (0.1 * np.sin(2 * np.pi * 440 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    y2 = y.copy()
    y2[int(5 * F.SR):] *= 10.0           # 5초 뒤에 갑자기 아주 커진다
    a, ft, _ = F.frame_chroma(y, causal=True, hpss=True, level="running")
    b, _, _ = F.frame_chroma(y2, causal=True, hpss=True, level="running")
    ready = ft + F.N_FFT / 2 / F.SR      # 프레임 끝 시각
    early = ready < 4.9
    assert np.allclose(a[:, early], b[:, early])
