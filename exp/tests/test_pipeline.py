import numpy as np

from ragsgo_exp import beats as B
from ragsgo_exp import features as F
from ragsgo_exp.pipeline import reference_beat_chroma, run_pipeline

SR = F.SR
BEAT = 0.6


def synth_song(n_beats, seed=0):
    """박마다 화음이 바뀌는 합성 곡 (사인파 3화음 + 박마다 드럼 타격). 반환: 소리, 박 시각."""
    rng = np.random.default_rng(seed)
    roots = rng.integers(0, 12, n_beats)
    minor = rng.random(n_beats) < 0.5
    t_beats = 0.5 + np.arange(n_beats) * BEAT
    y = np.zeros(int((t_beats[-1] + 1.5) * SR), dtype=np.float32)
    n = int(BEAT * SR)
    tt = np.arange(n) / SR
    for k in range(n_beats):
        tones = [roots[k], roots[k] + (3 if minor[k] else 4), roots[k] + 7]
        seg = np.zeros(n)
        for pc in tones:
            for octave in (3, 4, 5):
                f = 440.0 * 2 ** ((pc + 12 * octave - 57) / 12)
                seg += np.sin(2 * np.pi * f * tt)
        i = int(t_beats[k] * SR)
        y[i:i + n] += (0.05 * seg * np.minimum(1, tt / 0.01)).astype(np.float32)
    hit_n = int(0.04 * SR)
    env = np.exp(-np.arange(hit_n) / (0.006 * SR))
    for t in t_beats:
        i = int(t * SR)
        y[i:i + hit_n] += (0.5 * rng.standard_normal(hit_n) * env).astype(np.float32)
    return y, t_beats


def silence(dur, seed=1):
    return (0.0005 * np.random.default_rng(seed).standard_normal(int(dur * SR))).astype(np.float32)


def numbers_ok(out, true_time_to_number):
    ok = []
    for t, num in zip(out["beats"], out["number"]):
        want = true_time_to_number(t)
        if want is not None:
            ok.append(num == want)
    return np.array(ok)


def test_pipeline_counts_beats_of_known_song():
    ref_audio, ref_beats = synth_song(64)
    ref_bs = reference_beat_chroma(ref_audio, ref_beats)
    live = np.concatenate([silence(1.0), ref_audio * 0.7]) + silence(1.0 + len(ref_audio) / SR, seed=2)
    out = run_pipeline(live, ref_bs, bpm=60 / BEAT)
    true_t = ref_beats + 1.0

    def want(t):
        j = int(np.argmin(np.abs(true_t - t)))
        return j + 1 if abs(true_t[j] - t) < 0.1 else None
    ok = numbers_ok(out, want)
    assert len(ok) > 55 and ok[4:].mean() > 0.95


def test_pipeline_finds_position_after_stop_and_restart():
    ref_audio, ref_beats = synth_song(96, seed=3)
    ref_bs = reference_beat_chroma(ref_audio, ref_beats)
    stop_beat, restart_beat = 60, 24
    a_end = int((ref_beats[stop_beat] - 0.02) * SR)
    b_start = int((ref_beats[restart_beat] - 0.02) * SR)
    seg_a = ref_audio[:a_end]
    seg_b = ref_audio[b_start:]
    pause = silence(3.0, seed=4)
    live = np.concatenate([silence(1.0, seed=5), seg_a, pause, seg_b])
    out = run_pipeline(live, ref_bs, bpm=60 / BEAT)
    t_b0 = 1.0 + a_end / SR + 3.0 - 0.02  # 재시작 구간에서 restart_beat번 박(0부터)이 울리는 시각

    def want(t):
        if t < 1.0 + a_end / SR:
            tb = ref_beats + 1.0
            j = int(np.argmin(np.abs(tb - t)))
            return j + 1 if abs(tb[j] - t) < 0.1 else None
        tb = t_b0 + (ref_beats[restart_beat:] - ref_beats[restart_beat])
        j = int(np.argmin(np.abs(tb - t)))
        return restart_beat + j + 1 if abs(tb[j] - t) < 0.1 else None
    beats_after = out["beats"] > t_b0 + 16 * BEAT  # 다시 시작하고 4마디(16박) 뒤부터
    ok_after = numbers_ok({"beats": out["beats"][beats_after], "number": out["number"][beats_after]}, want)
    assert len(ok_after) > 30 and ok_after.mean() > 0.95
    assert len(out["resumes"]) == 2  # 곡 시작과 재시작


def test_resume_hint_relocks_within_a_bar():
    """다시 칠 마디를 탭으로 알려 주면, 연주를 다시 시작한 순간을 그 마디 첫 박으로 보고 바로 맞춘다."""
    ref_audio, ref_beats = synth_song(96, seed=6)
    ref_bs = reference_beat_chroma(ref_audio, ref_beats)
    stop_beat, restart_beat = 60, 24
    a_end = int((ref_beats[stop_beat] - 0.02) * SR)
    b_start = int((ref_beats[restart_beat] - 0.02) * SR)
    live = np.concatenate([silence(1.0, seed=7), ref_audio[:a_end], silence(3.0, seed=8), ref_audio[b_start:]])
    out = run_pipeline(live, ref_bs, bpm=60 / BEAT, resume_hint=lambda i: restart_beat)
    t_b0 = 1.0 + a_end / SR + 3.0 - 0.02
    after = (out["beats"] > t_b0 + 4 * BEAT) & (out["beats"] < t_b0 + 30 * BEAT)  # 다시 시작하고 1마디 뒤부터
    tb = t_b0 + (ref_beats[restart_beat:] - ref_beats[restart_beat])
    ok = []
    for t, num in zip(out["beats"][after], out["number"][after]):
        j = int(np.argmin(np.abs(tb - t)))
        if abs(tb[j] - t) < 0.1:
            ok.append(num == restart_beat + j + 1)
    assert len(ok) > 20 and np.mean(ok) > 0.95
