"""Phase 1-a 특징: STFT → (인과) HPSS → 크로마 12차원 → 10 Hz.

librosa 없이 numpy/scipy로 구현한다. 나중에 C++로 옮기기도 쉽다.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import get_window, lfilter

SR = 22050
N_FFT = 4096
HOP = 512
FEAT_RATE = 10.0            # 묶은 뒤 초당 프레임 수
HPSS_TIME_WIN = 21          # 프레임 (약 0.49초)
HPSS_FREQ_WIN = 17          # 주파수 bin
FMIN, FMAX = 60.0, 5000.0
LOG_GAMMA = 100.0
CHROMA_SIGMA = 0.35         # 음이름 가중의 폭 (반음). 0.5면 옆 음으로 13%가 번진다
SILENCE_DB_BELOW_P95 = 40.0
STFT_CHUNK = 1024           # 메모리를 아끼려고 프레임을 나눠 계산한다


@dataclass
class Features:
    chroma: np.ndarray      # (12, K), 열마다 L2 정규화
    silent: np.ndarray      # (K,) bool
    times: np.ndarray       # (K,) 각 0.1초 칸의 가운데 시각 (초)
    energy_db: np.ndarray   # (K,)


def n_bins() -> int:
    return int(np.ceil(FMAX * N_FFT / SR)) + 1


def magnitude(y: np.ndarray) -> np.ndarray:
    """|STFT|, 모양 (bins, frames). center=False라서 프레임 t는 샘플 [t*HOP, t*HOP+N_FFT)만 쓴다."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP]
    win = get_window("hann", N_FFT).astype(np.float32)
    nb = n_bins()
    out = np.empty((nb, frames.shape[0]), dtype=np.float32)
    for s in range(0, frames.shape[0], STFT_CHUNK):
        spec = np.fft.rfft(frames[s:s + STFT_CHUNK] * win, axis=1)
        out[:, s:s + STFT_CHUNK] = np.abs(spec[:, :nb]).T
    return out


def frame_times(n_frames: int) -> np.ndarray:
    """STFT 프레임 가운데 시각 (초)."""
    return (np.arange(n_frames) * HOP + N_FFT / 2) / SR


def harmonic_part(S: np.ndarray, causal: bool = True) -> np.ndarray:
    """부드러운 마스크 HPSS의 화성 성분.

    causal=True면 시간축 중앙값 창이 [t-20, t] 프레임만 덮는다 (과거와 현재만 씀).
    주파수축 중앙값은 같은 프레임 안에서만 계산하므로 원래 인과적이다.
    """
    if causal:
        H = median_filter(S, size=(1, HPSS_TIME_WIN), origin=(0, HPSS_TIME_WIN // 2), mode="nearest")
    else:
        H = median_filter(S, size=(1, HPSS_TIME_WIN), mode="nearest")
    P = median_filter(S, size=(HPSS_FREQ_WIN, 1), mode="nearest")
    H2 = H.astype(np.float64) ** 2
    P2 = P.astype(np.float64) ** 2
    return (S * (H2 / (H2 + P2 + 1e-20))).astype(np.float32)


def chroma_filter() -> np.ndarray:
    """(12, bins). 각 bin의 에너지를 가까운 음이름에 나눠 준다. C=0, ..., A=9.

    880 Hz 근처를 중심으로 옥타브 가중을 준다 (librosa 기본값과 같은 생각: 중심 A5, 폭 2옥타브).
    """
    freqs = np.arange(n_bins()) * SR / N_FFT
    W = np.zeros((12, len(freqs)))
    valid = (freqs >= FMIN) & (freqs <= FMAX)
    p = 69 + 12 * np.log2(freqs[valid] / 440.0)
    octave_w = np.exp(-0.5 * ((np.log2(freqs[valid] / 27.5) - 5.0) / 2.0) ** 2)
    for c in range(12):
        d = (p - c + 6) % 12 - 6  # 음이름 c까지의 거리 (반음, -6~6)
        W[c, valid] = np.exp(-0.5 * (d / CHROMA_SIGMA) ** 2) * octave_w
    return W


RUNNING_LEVEL_TAU_S = 5.0
RUNNING_LEVEL_GAIN = 3.0    # 지수 평균의 3배를 상위 5% 음량 근사로 쓴다


def frame_chroma(y: np.ndarray, causal: bool = True, hpss: bool = True, level: str = "global"):
    """STFT 프레임(초당 약 43개)마다 로그 압축 크로마. 반환: (12, T) 크로마, (T,) 프레임 시각, (T,) 프레임 에너지.

    아직 평평한 성분을 빼거나 정규화하지 않은 값이다. 묶는 쪽(10 Hz 칸, 박 사이)에서 한다.
    level: 로그 압축 기준 음량. "global"은 녹음 전체 상위 5%(미래 소리를 씀, 원곡 쪽용),
    "running"은 지나간 소리의 지수 평균 x 3 (인과적, 마이크 쪽용).
    """
    S = magnitude(y)
    Sh = harmonic_part(S, causal=causal) if hpss else S
    energy = chroma_filter() @ (Sh.astype(np.float64) ** 2)              # (12, T)
    # 녹음 전체 음량(상위 5%)을 기준으로 로그 압축한다. 프레임마다 정규화하면 벡터가 평평해져
    # 서로 다른 화음도 비슷하게 보인다 (v1에서 확인한 문제). 진단용 기준이며 실제 앱은 이동 추정이 필요하다.
    total = energy.sum(axis=0)
    if level == "running":
        a = 1.0 - np.exp(-HOP / SR / RUNNING_LEVEL_TAU_S)
        ema = lfilter([a], [1.0, a - 1.0], total, zi=[(1.0 - a) * total[0]])[0]
        lev = RUNNING_LEVEL_GAIN * ema + 1e-20
    else:
        lev = np.percentile(total, 95) + 1e-20
    fc = np.log1p(LOG_GAMMA * energy / lev)
    fe = (S.astype(np.float64) ** 2).sum(axis=0)
    return fc, frame_times(S.shape[1]), fe


def chroma_features(y: np.ndarray, causal: bool = True, hpss: bool = True) -> Features:
    frame_chroma_, ft, frame_energy = frame_chroma(y, causal=causal, hpss=hpss)

    bins = np.floor(ft * FEAT_RATE).astype(int)
    K = int(bins.max()) + 1
    counts = np.bincount(bins, minlength=K).astype(np.float64)
    pooled = np.stack([np.bincount(bins, weights=frame_chroma_[c], minlength=K) for c in range(12)])
    pooled /= np.maximum(counts, 1)
    pooled_energy = np.bincount(bins, weights=frame_energy, minlength=K) / np.maximum(counts, 1)

    pooled -= pooled.mean(axis=0, keepdims=True)  # 평평한 성분을 뺀다: 코사인 유사도가 상관계수가 된다
    chroma = pooled / np.maximum(np.linalg.norm(pooled, axis=0), 1e-12)
    energy_db = 10 * np.log10(pooled_energy + 1e-20)
    # 무음 기준은 녹음 전체의 상위 5% 음량 대비 40 dB 아래 (진단용. 실제 앱은 고정 기준이 필요)
    silent = (counts == 0) | (energy_db < np.percentile(energy_db, 95) - SILENCE_DB_BELOW_P95)
    times = (np.arange(K) + 0.5) / FEAT_RATE
    return Features(chroma.astype(np.float32), silent, times, energy_db.astype(np.float32))
