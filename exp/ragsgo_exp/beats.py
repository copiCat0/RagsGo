"""Phase 1-b 박 추적 도구.

- onset_strength: 로그 스펙트럼 플럭스. 프레임 t는 샘플 [t*HOP, t*HOP+N_FFT)만 쓰므로 인과적이다.
- estimate_tempo: 자기상관에 사전 템포 가중을 곱해 템포를 고른다.
- dp_beats: 오프라인 동적계획법 박 추적 (Ellis 2007). 원곡 박 격자와 정답 초안을 만드는 데 쓴다.
- BeatTracker: 인과 박 추적. (박 안 위상, 박 간격)을 상태로 두는 HMM의 전방 필터다.
  bar pointer 모델(Whiteley 2006, Krebs 2015)을 박 하나 길이로 줄인 것이다. 미래 프레임은 쓰지 않는다.
- pool_between: 두 시각 사이 프레임의 평균 (박 동기 크로마)
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import get_window

SR = 22050
ODF_NFFT = 1024
ODF_HOP = 256
ODF_FPS = SR / ODF_HOP          # 초당 86.13 프레임 (11.6 ms)
ODF_FMAX = 8000.0
ODF_GAMMA = 100.0
# 프레임 t를 계산할 수 있는 시각은 창의 끝이다. 로그 압축 때문에 온셋이 창 끝에 막 들어온 첫 1~2 프레임에서
# 플럭스가 가장 크다. 합성 타격 소리로 재 보니 봉우리 프레임의 끝은 실제 온셋보다 14~24 ms 뒤였다.
# 그래서 사건 시각 = 창 끝 - 1.5홉(17 ms)으로 둔다 (test_onset_strength_peaks_line_up_with_hits).
ODF_EVENT_BEFORE_READY = 1.5 * ODF_HOP / SR


def frame_ready_time(t):
    return (np.asarray(t) * ODF_HOP + ODF_NFFT) / SR


def frame_event_time(t):
    return frame_ready_time(t) - ODF_EVENT_BEFORE_READY


def onset_strength(y: np.ndarray):
    """반환: (odf, 각 프레임의 사건 시각). odf[t]는 프레임 t와 t-1의 로그 크기 차이(양수만)의 합."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < ODF_NFFT:
        y = np.pad(y, (0, ODF_NFFT - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, ODF_NFFT)[::ODF_HOP]
    win = get_window("hann", ODF_NFFT).astype(np.float32)
    nb = int(ODF_FMAX * ODF_NFFT / SR) + 1
    L = np.empty((frames.shape[0], nb), dtype=np.float32)
    for s in range(0, frames.shape[0], 4096):
        X = np.abs(np.fft.rfft(frames[s:s + 4096] * win, axis=1)[:, :nb]) / win.sum()
        L[s:s + 4096] = np.log1p(ODF_GAMMA * X)
    prev = maximum_filter1d(L[:-1], size=3, axis=1)  # 이웃 bin 최댓값과 비교해 비브라토 오검출을 줄인다
    odf = np.zeros(frames.shape[0], dtype=np.float64)
    odf[1:] = np.maximum(L[1:] - prev, 0).sum(axis=1)
    return odf, frame_event_time(np.arange(len(odf)))


def estimate_tempo(odf: np.ndarray, prior_bpm: float, fps: float = ODF_FPS, prior_octaves: float = 0.5,
                   bpm_range=(40.0, 240.0)) -> float:
    """자기상관 봉우리 중 사전 템포 근처를 고른다. 사전 가중은 로그 템포 축의 정규분포다."""
    x = odf - odf.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    lags = np.arange(n, dtype=np.float64)
    lo = int(np.floor(60 * fps / bpm_range[1]))
    hi = min(int(np.ceil(60 * fps / bpm_range[0])), n - 2)
    L = lags[lo:hi + 1]
    bpm = 60 * fps / L
    w = np.exp(-0.5 * (np.log2(bpm / prior_bpm) / prior_octaves) ** 2)
    score = np.maximum(ac[lo:hi + 1], 0) * w
    k = int(np.argmax(score)) + lo
    # 포물선 보간으로 프레임보다 잘게
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    denom = a - 2 * b + c
    off = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return float(60 * fps / (k + np.clip(off, -0.5, 0.5)))


def dp_beats(odf: np.ndarray, bpm: float, fps: float = ODF_FPS, tightness: float = 100.0,
             trim: bool = True) -> np.ndarray:
    """오프라인 동적계획법 박 추적 (Ellis 2007). 반환: 박 사건 시각 (초)."""
    period = 60.0 * fps / bpm
    o = odf / (odf.std() + 1e-12)
    half = int(round(period))
    win = np.exp(-0.5 * (np.arange(-half, half + 1) * 32.0 / period) ** 2)
    local = np.convolve(o, win, mode="same")
    n = len(local)
    offs = np.arange(int(round(period / 2)), int(round(2 * period)) + 1)
    txwt = -tightness * np.log(offs / period) ** 2
    cum = np.zeros(n)
    back = np.full(n, -1)
    for t in range(n):
        cand = t - offs
        ok = cand >= 0
        if ok.any():
            sc = cum[cand[ok]] + txwt[ok]
            j = int(np.argmax(sc))
            if sc[j] > 0:
                cum[t] = local[t] + sc[j]
                back[t] = cand[ok][j]
                continue
        cum[t] = local[t]
    # 마지막 박: 누적 점수의 국소 최댓값 중 중앙값의 절반을 넘는 마지막 것
    peaks = np.where((cum[1:-1] > cum[:-2]) & (cum[1:-1] >= cum[2:]))[0] + 1
    if len(peaks) == 0:
        return np.array([])
    good = peaks[cum[peaks] > 0.5 * np.median(cum[peaks])]
    t = int(good[-1]) if len(good) else int(peaks[-1])
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    beats = np.array(beats[::-1])
    if trim and len(beats) > 2:
        th = 0.5 * np.sqrt(np.mean(local[beats] ** 2))
        strong = np.where(local[beats] >= th)[0]
        beats = beats[strong[0]:strong[-1] + 1]
    return frame_event_time(beats)


@dataclass
class TrackerOutput:
    frame_time: np.ndarray          # 프레임 사건 시각
    ready_time: np.ndarray          # 프레임을 계산할 수 있게 된 시각
    phase: np.ndarray               # 사후 분포의 원형 평균 위상 [0, 1)
    period: np.ndarray              # 사후 평균 박 간격 (초)
    next_beat: np.ndarray           # 그 프레임을 받은 시각(ready_time) 뒤로 처음 올 박의 예측 시각 (초)
    confidence: np.ndarray          # 위상 집중도 R (0~1)
    beat_times: np.ndarray          # 검출한 박의 사건 시각
    beat_report_times: np.ndarray   # 그 박을 알게 된 시각
    beat_periods: np.ndarray        # 그 박에서의 박 간격 추정 (초)
    beat_strength: np.ndarray = None  # 박 세기 0~1: 박 자리 온셋이 바로 앞 반 박 자리보다 얼마나 큰가 (인과적)


class BeatTracker:
    """인과 박 추적 HMM. 상태 = (박 간격 p 프레임, 박 안 위상 φ = 0..p-1).

    - 매 프레임 위상이 1 오른다. 위상이 박 끝에서 0으로 돌아갈 때만 박 간격이 이웃 값으로 바뀔 수 있다.
    - 관측: 위상이 박 앞부분(1/λ)이면 온셋 세기 o, 아니면 (1-o)/(λ-1) (madmom의 관측 모델과 같은 꼴).
    - o는 온셋 세기를 과거 평균으로 나눠 0~1로 자른 값이다 (인과적).
    """

    def __init__(self, bpm: float, fps: float = ODF_FPS, tempo_tol: float = 0.15, p_tempo: float = 0.02,
                 lam: float = 16.0, norm_tau_s: float = 3.0, norm_gain: float = 4.0):
        self.fps = fps
        p0 = 60.0 * fps / bpm
        self.periods = np.arange(int(np.floor(p0 * (1 - tempo_tol))), int(np.ceil(p0 * (1 + tempo_tol))) + 1)
        K = len(self.periods)
        self.start = np.concatenate([[0], np.cumsum(self.periods)[:-1]]).astype(int)
        self.end = self.start + self.periods - 1
        self.n_states = int(self.periods.sum())
        self.phi = np.concatenate([np.arange(p) for p in self.periods]).astype(np.float64)
        self.per = np.concatenate([np.full(p, p) for p in self.periods]).astype(np.float64)
        self.is_beat = self.phi < np.maximum(1.0, self.per / lam)
        self.lam = lam
        T = np.zeros((K, K))
        for k in range(K):
            T[k, k] = 1.0 - 2 * p_tempo
            if k > 0:
                T[k, k - 1] = p_tempo
            if k < K - 1:
                T[k, k + 1] = p_tempo
        self.T = T / T.sum(axis=1, keepdims=True)  # 행: 이전 템포, 열: 다음 템포
        self.ang = np.exp(2j * np.pi * self.phi / self.per)
        self.norm_alpha = 1.0 - np.exp(-1.0 / (fps * norm_tau_s))
        self.norm_gain = norm_gain

    def run(self, odf: np.ndarray) -> TrackerOutput:
        n = len(odf)
        a = np.full(self.n_states, 1.0 / self.n_states)
        mean = None
        phase = np.zeros(n)
        period = np.zeros(n)
        conf = np.zeros(n)
        ft = frame_event_time(np.arange(n))
        rt = frame_ready_time(np.arange(n))
        beats, reports, beat_per, strength = [], [], [], []
        prev_theta = None
        for t in range(n):
            x = float(odf[t])
            mean = x if mean is None else mean + self.norm_alpha * (x - mean)
            o = np.clip(x / (self.norm_gain * mean + 1e-12), 1e-3, 1 - 1e-3)
            new = np.empty_like(a)
            new[1:] = a[:-1]
            new[self.start] = a[self.end] @ self.T
            new *= np.where(self.is_beat, o, (1.0 - o) / (self.lam - 1.0))
            a = new / new.sum()
            z = a @ self.ang
            theta = (np.angle(z) / (2 * np.pi)) % 1.0
            P = float(a @ self.per)
            phase[t], period[t], conf[t] = theta, P / self.fps, abs(z)
            if prev_theta is not None and theta < prev_theta - 0.5:  # 위상이 한 바퀴 돌았다 = 박
                bt_ev = ft[t] - theta * P / self.fps
                beats.append(bt_ev)
                reports.append(rt[t])
                beat_per.append(P / self.fps)
                # 박 세기: 박 자리 온셋과 바로 앞 반 박 자리 온셋을 비교한다. 추적기는 박을 실제 소리보다
                # 조금 일찍 내므로 박 자리는 사건 프레임 -2 ~ +3까지 본다. 그래서 이 값은 박 보고보다
                # 약 35 ms 늦게 정해진다 (넘김 확인에서는 지난 박들의 세기만 쓴다, turns.py).
                # 드럼이 박을 치면 박 자리가 훨씬 크다. 드럼이 빠지면 둘이 비슷해진다. 0.5 → 0, 0.8 → 1.
                k_ev = int(round(t - theta * P))
                on_ = float(np.max(odf[max(0, k_ev - 2):min(len(odf), k_ev + 4)]))
                k_half = int(round(k_ev - P / 2))
                off_ = float(np.max(odf[max(0, k_half - 2):max(1, k_half + 4)]))
                sal = on_ / (on_ + off_ + 1e-12)
                strength.append(float(np.clip((sal - 0.5) / 0.3, 0.0, 1.0)))
            prev_theta = theta
        next_beat = ft + (1.0 - phase) * period
        late = next_beat <= rt  # 예측한 박이 이미 지났으면 다음 박으로 넘긴다
        next_beat[late] += (np.floor((rt[late] - next_beat[late]) / period[late]) + 1) * period[late]
        return TrackerOutput(ft, rt, phase, period, next_beat, conf, np.array(beats), np.array(reports),
                             np.array(beat_per), np.array(strength))


def pool_between(fc: np.ndarray, ft: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """edges[i] <= 시각 < edges[i+1]인 프레임을 평균하고, 평평한 성분을 빼고, L2 정규화한다. 모양 (12, len(edges)-1)."""
    idx = np.searchsorted(ft, edges)
    out = np.zeros((fc.shape[0], len(edges) - 1))
    for i in range(len(edges) - 1):
        a, b = idx[i], idx[i + 1]
        if b <= a:
            j = min(max(a, 0), fc.shape[1] - 1)
            out[:, i] = fc[:, j]
        else:
            out[:, i] = fc[:, a:b].mean(axis=1)
    out -= out.mean(axis=0, keepdims=True)
    return out / np.maximum(np.linalg.norm(out, axis=0), 1e-12)


def detect_music_start(y: np.ndarray, sr: int = SR, floor_s: float = 0.5, rise_db: float = 20.0,
                       sustain_s: float = 0.25, win_s: float = 0.05):
    """시작 버튼 뒤 밴드가 실제로 연주를 시작한 시각 (초, y 기준). 없으면 None.

    처음 floor_s초의 음량 중앙값을 바닥으로 보고, 바닥보다 rise_db 이상 큰 소리가 sustain_s초 이상
    이어지는 첫 지점을 고른다. 스틱 카운트 같은 짧은 딸깍 소리는 이어지지 않으므로 고르지 않는다.
    알게 되는 시각은 시작 + sustain_s다 (인과적).
    """
    w = int(win_s * sr)
    n = len(y) // w
    if n == 0:
        return None
    rms = np.sqrt(np.mean(np.asarray(y[:n * w], dtype=np.float64).reshape(n, w) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-12)
    floor = np.median(db[:max(1, int(floor_s / win_s))])
    loud = db > floor + rise_db
    need = max(1, int(round(sustain_s / win_s)))
    run = 0
    for i, v in enumerate(loud):
        run = run + 1 if v else 0
        if run >= need:
            return (i - need + 1) * win_s
    return None


MUSIC_WIN_S = 0.05
MUSIC_ON_SUSTAIN_S = 0.25


def music_activity(y: np.ndarray, sr: int = SR, floor_s: float = 0.5, rise_db: float = 20.0,
                   sustain_on_s: float = MUSIC_ON_SUSTAIN_S, drop_db: float = 20.0, sustain_off_s: float = 0.6,
                   back_db: float = 12.0, level_tau_s: float = 5.0, return_starts: bool = False,
                   back_rise_db: float = 3.0, max_back_s: float = 2.0):
    """밴드가 연주 중인지(ON) 멈췄는지(OFF)를 50 ms 칸마다 인과적으로 판단한다.

    반환: (times, on). times[i]는 칸 i가 끝나는 시각(그 판단을 알게 되는 시각), on[i]는 그때의 상태.
    - OFF → ON: 바닥 + rise_db 이상이면서 직전 연주 음량 - back_db 이상인 소리가 sustain_on_s 이어질 때.
      스틱 카운트처럼 짧은 소리나, 연주보다 훨씬 작은 말소리로는 켜지지 않는다.
    - ON → OFF: 최근 sustain_off_s 동안의 평균 에너지가 연주 음량(지수 평균)보다 drop_db 이상 작을 때.
      곡 안의 조용한 대목(10 dB 안팎 작아짐)은 멈춤으로 보지 않는다.
    return_starts=True면 연주가 실제로 시작된 시각 목록도 낸다. ON이 된 지점에서 거꾸로 가며,
    바닥(멈춘 동안의 음량 중앙값)보다 back_rise_db 이상 큰 칸이 이어지는 데까지(최대 max_back_s) 앞당긴다.
    잡음이 큰 방에서 조용한 인트로로 시작해도 첫 박을 놓치지 않게 하려는 것이다.
    """
    w = int(MUSIC_WIN_S * sr)
    n = len(y) // w
    p = np.mean(np.asarray(y[:n * w], dtype=np.float64).reshape(n, w) ** 2, axis=1) + 1e-12
    db = 10 * np.log10(p)
    floor = np.median(db[:max(1, int(floor_s / MUSIC_WIN_S))])
    need_on = max(1, int(round(sustain_on_s / MUSIC_WIN_S)))
    k_off = max(1, int(round(sustain_off_s / MUSIC_WIN_S)))
    a = 1.0 - np.exp(-MUSIC_WIN_S / level_tau_s)
    on = np.zeros(n, dtype=bool)
    starts = []
    off_db = []
    max_back = int(round(max_back_s / MUSIC_WIN_S))
    state, run, level = False, 0, None
    for i in range(n):
        if not state:
            off_db.append(db[i])
            th = floor + rise_db if level is None else max(floor + rise_db, level - back_db)
            run = run + 1 if db[i] >= th else 0
            if run >= need_on:
                state = True
                i0 = i - need_on + 1
                seg = db[i0:i + 1]
                level = float(np.mean(seg)) if level is None else level
                base = float(np.median(off_db[:max(1, len(off_db) - need_on)])) if len(off_db) > need_on else floor
                j = i0
                while j - 1 >= max(0, i0 - max_back) and db[j - 1] >= base + back_rise_db:
                    j -= 1
                starts.append(j * MUSIC_WIN_S)
                off_db = []
        else:
            level = level + a * (db[i] - level)
            recent = 10 * np.log10(np.mean(p[max(0, i - k_off + 1):i + 1]))
            if i >= k_off and recent < level - drop_db:
                state, run = False, 0
        on[i] = state
    times = (np.arange(n) + 1) * MUSIC_WIN_S
    if return_starts:
        return times, on, np.array(starts)
    return times, on


def resume_times(times: np.ndarray, on: np.ndarray, sustain_on_s: float = MUSIC_ON_SUSTAIN_S):
    """OFF → ON으로 바뀐 칸에서 연주가 실제로 다시 시작된 시각(이어진 소리의 시작)을 낸다."""
    idx = np.where(on[1:] & ~on[:-1])[0] + 1
    out = list(times[idx] - sustain_on_s - MUSIC_WIN_S)
    if len(on) and on[0]:
        out = [0.0] + out
    return np.array(out)
