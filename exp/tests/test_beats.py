import numpy as np

from ragsgo_exp import beats as B

SR = 22050


def drum_hits(times, dur, amp=0.5, seed=0):
    """짧은 잡음 타격(드럼 비슷한 소리)을 주어진 시각에 넣는다."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(dur * SR), dtype=np.float32)
    n = int(0.04 * SR)
    env = np.exp(-np.arange(n) / (0.006 * SR))
    for t, a in zip(times, np.broadcast_to(amp, len(times))):
        i = int(round(t * SR))
        if 0 <= i and i + n <= len(y):
            y[i:i + n] += (a * rng.standard_normal(n) * env).astype(np.float32)
    return y


def beat_times_with_drift(bpm0, bpm1, dur, start=0.5):
    """템포가 bpm0에서 bpm1로 서서히 바뀌는 박 시각."""
    t, out = start, []
    while t < dur - 0.5:
        out.append(t)
        frac = t / dur
        t += 60.0 / (bpm0 + (bpm1 - bpm0) * frac)
    return np.array(out)


def f_measure(est, ref, tol=0.07):
    est, ref = np.asarray(est), np.asarray(ref)
    used = np.zeros(len(ref), bool)
    hit = 0
    for e in est:
        d = np.abs(ref - e)
        d[used] = np.inf
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= tol:
            used[j] = True
            hit += 1
    p = hit / max(len(est), 1)
    r = hit / max(len(ref), 1)
    return 0.0 if hit == 0 else 2 * p * r / (p + r)


def test_onset_strength_peaks_line_up_with_hits():
    times = np.arange(0.5, 9.5, 0.7)
    odf, t = B.onset_strength(drum_hits(times, 10.0))
    for x in times:
        win = (t > x - 0.1) & (t < x + 0.1)
        peak_t = t[win][np.argmax(odf[win])]
        assert abs(peak_t - x) <= 1.5 / B.ODF_FPS, (x, peak_t)


def test_onset_strength_is_causal():
    y = drum_hits(np.arange(0.5, 9.5, 0.7), 10.0)
    y2 = y.copy()
    y2[int(6.0 * SR):] = 0.3 * np.random.default_rng(1).standard_normal(len(y2) - int(6.0 * SR))
    a, t = B.onset_strength(y)
    b, _ = B.onset_strength(y2)
    ready = B.frame_ready_time(np.arange(len(a)))  # 프레임을 계산할 수 있게 되는 시각
    early = ready < 6.0
    assert np.allclose(a[early], b[early])


def test_tempo_estimate_finds_true_tempo():
    times = beat_times_with_drift(85, 85, 30.0)
    odf, _ = B.onset_strength(drum_hits(times, 30.0))
    assert abs(B.estimate_tempo(odf, prior_bpm=85) - 85) < 1.0
    assert abs(B.estimate_tempo(odf, prior_bpm=80) - 85) < 1.0


def test_dp_beats_follow_quarter_notes_not_weaker_eighths():
    q = beat_times_with_drift(85, 85, 30.0)
    jitter = np.random.default_rng(2).normal(0, 0.008, len(q))
    eighths = (q[:-1] + q[1:]) / 2
    y = drum_hits(q + jitter, 30.0, amp=0.6) + drum_hits(eighths, 30.0, amp=0.15, seed=3)
    odf, t = B.onset_strength(y)
    est = B.dp_beats(odf, bpm=85)
    assert f_measure(est, q + jitter, tol=0.03) > 0.97


def test_causal_tracker_locks_and_follows_drift():
    true = beat_times_with_drift(84, 90, 60.0)
    rng = np.random.default_rng(4)
    keep = rng.random(len(true)) > 0.05  # 박 5%는 소리가 없다
    played = true[keep] + rng.normal(0, 0.01, keep.sum())
    y = drum_hits(played, 60.0)
    odf, _ = B.onset_strength(y)
    tr = B.BeatTracker(bpm=85)
    out = tr.run(odf)
    est = out.beat_times
    assert f_measure(est[est > 5.0], true[true > 5.0]) > 0.95


def test_causal_tracker_predicts_next_beat():
    true = beat_times_with_drift(85, 88, 40.0)
    odf, _ = B.onset_strength(drum_hits(true, 40.0))
    out = B.BeatTracker(bpm=85).run(odf)
    errs = []
    for k in range(len(out.frame_time)):
        now = out.ready_time[k]
        if now < 8.0:
            continue
        nxt = true[true > now]
        if len(nxt) == 0:
            break
        if nxt[0] - now < 0.1:  # 박 바로 앞 프레임은 뺀다 (경계에서 "다음 박"이 바뀐다)
            continue
        errs.append(out.next_beat[k] - nxt[0])
    errs = np.abs(np.array(errs))
    assert np.percentile(errs, 95) < 0.05


def test_beat_sync_pooling_averages_frames_between_beats():
    fc = np.zeros((12, 100))
    ft = np.arange(100) * 0.1
    fc[0, :50] = 1.0   # 0~5초는 C
    fc[7, 50:] = 1.0   # 5~10초는 G
    pooled = B.pool_between(fc, ft, np.array([0.0, 2.5, 5.0, 7.5, 10.0]))
    assert pooled.shape == (12, 4)
    assert np.argmax(pooled[:, 0]) == 0 and np.argmax(pooled[:, 3]) == 7
    assert np.allclose(np.linalg.norm(pooled, axis=0), 1.0)


def test_music_start_ignores_count_in_clicks():
    rng = np.random.default_rng(7)
    y = 0.0005 * rng.standard_normal(int(8 * SR)).astype(np.float32)        # 조용한 방
    y += drum_hits(np.array([1.0, 1.7, 2.4, 3.1]), 8.0, amp=0.05)            # 스틱 카운트 4번 (짧은 딸깍)
    t = np.arange(len(y)) / SR
    band = (t >= 3.8) * 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
    y = (y + band).astype(np.float32)                                         # 3.8초부터 밴드 연주
    start = B.detect_music_start(y)
    assert abs(start - 3.8) < 0.1


def test_music_start_is_none_for_silence():
    rng = np.random.default_rng(8)
    y = (0.0005 * rng.standard_normal(int(5 * SR))).astype(np.float32)
    assert B.detect_music_start(y) is None


def band_like(dur, start, stop, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    on = (t >= start) & (t < stop)
    return (on * amp * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))).astype(np.float32)


def test_music_activity_finds_stop_and_restart():
    rng = np.random.default_rng(9)
    dur = 20.0
    y = (0.0005 * rng.standard_normal(int(dur * SR))).astype(np.float32)
    y += band_like(dur, 1.0, 8.0) + band_like(dur, 12.0, 20.0)                     # 8초에 멈추고 12초에 다시
    y += drum_hits(np.array([9.5, 10.2, 10.9, 11.6]), dur, amp=0.05)               # 멈춘 동안 스틱 카운트
    times, on = B.music_activity(y)
    def state(t):
        return on[np.searchsorted(times, t)]
    assert state(5.0) and state(15.0)
    assert not state(9.8) and not state(11.0)
    # 멈춤은 1초 안에, 재개는 0.4초 안에 알아챈다
    off_at = times[(times > 8.0) & ~on][0]
    on_again = times[(times > 11.0) & on][0]
    assert off_at < 9.0 and on_again < 12.4


def test_music_activity_stays_on_through_quiet_passage():
    rng = np.random.default_rng(10)
    dur = 12.0
    y = (0.0005 * rng.standard_normal(int(dur * SR))).astype(np.float32)
    y += band_like(dur, 1.0, 12.0) * np.where(np.arange(int(dur * SR)) / SR < 6.0, 1.0, 0.25).astype(np.float32)
    times, on = B.music_activity(y)
    assert on[(times > 2.0)].all()   # 12 dB 작아진 구간은 멈춤이 아니다


def test_music_start_is_backdated_to_a_quiet_intro_in_a_noisy_room():
    rng = np.random.default_rng(13)
    dur = 10.0
    y = (0.02 * rng.standard_normal(int(dur * SR))).astype(np.float32)          # 시끄러운 방 (-34 dB)
    y += band_like(dur, 2.0, 3.5, amp=0.045)                                      # 조용한 인트로 (잡음보다 약 4 dB 큼)
    y += band_like(dur, 3.5, 10.0, amp=0.3)                                       # 본 연주
    times, on, starts = B.music_activity(y, rise_db=8.0, return_starts=True)
    assert len(starts) == 1
    assert abs(starts[0] - 2.0) < 0.15


def test_tracker_reports_beat_strength_low_when_drums_stop():
    true = beat_times_with_drift(85, 85, 40.0)
    amp = np.where(true < 20.0, 0.5, 0.0)                     # 20초 뒤로는 타격이 없다
    rng = np.random.default_rng(14)
    y = drum_hits(true, 40.0, amp=amp) + (0.002 * rng.standard_normal(int(40.0 * SR))).astype(np.float32)
    odf, _ = B.onset_strength(y)
    out = B.BeatTracker(bpm=85).run(odf)
    s = out.beat_strength
    early = (out.beat_times > 5) & (out.beat_times < 19)
    late = out.beat_times > 25
    assert np.median(s[early]) > 0.5
    assert np.median(s[late]) < 0.2
