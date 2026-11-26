import numpy as np

from ragsgo_exp import augment as A

SR = 22050


def tone(f, dur, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def peak_freq(y):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.fft.rfftfreq(len(y), 1 / SR)[np.argmax(spec)]


def db(x):
    return 10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20)


def test_time_stretch_changes_duration_and_keeps_pitch():
    y = tone(440, 2.0)
    out = A.time_stretch(y, rate=1.25)
    assert abs(len(out) / SR - 1.6) < 0.02
    assert abs(peak_freq(out[2000:-2000]) - 440) < 5


def test_phone_filter_cuts_deep_bass_and_keeps_mids():
    low = A.phone_mic(tone(50, 2.0))
    mid = A.phone_mic(tone(1000, 2.0))
    assert db(low[4000:]) < db(tone(50, 2.0)) - 20
    assert abs(db(mid[4000:]) - db(tone(1000, 2.0))) < 1.5


def test_compressor_narrows_level_difference():
    y = np.concatenate([tone(440, 2.0, 0.3), tone(440, 2.0, 0.03)])     # 20 dB 차이
    out = A.compress(y)
    diff = db(out[int(0.5 * SR):int(2 * SR)]) - db(out[int(2.5 * SR):])
    assert diff < 12


def test_remove_drums_keeps_tone_and_cuts_hits():
    rng = np.random.default_rng(0)
    y = tone(440, 4.0, 0.1)
    hits = np.zeros_like(y)
    n = int(0.04 * SR)
    env = np.exp(-np.arange(n) / (0.006 * SR))
    for t in np.arange(0.5, 3.5, 0.5):
        i = int(t * SR)
        hits[i:i + n] += (0.5 * rng.standard_normal(n) * env).astype(np.float32)
    out = A.remove_drums(y + hits)
    m = slice(int(0.3 * SR), int(3.7 * SR))
    tone_part = np.fft.rfft(out[m])
    ref_part = np.fft.rfft(y[m])
    k = int(440 * (m.stop - m.start) / SR)
    assert abs(np.abs(tone_part[k]) / np.abs(ref_part[k]) - 1) < 0.25   # 음은 남는다
    resid = out[m] - y[m]
    assert db(resid) < db(hits[m]) - 10                                     # 타격은 10 dB 넘게 줄어든다
