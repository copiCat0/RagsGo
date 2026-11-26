"""Phase 1-b 견고성 시험용 소리 변형 (흉내일 뿐 실제 기기 녹음을 대신하지 못한다, R8).

- time_stretch: 위상 보코더로 음높이는 두고 빠르기만 바꾼다 (밴드가 원곡보다 빠르거나 느린 경우).
- phone_mic: 휴대폰 내장 마이크 흉내. 150 Hz 아래와 7 kHz 위를 깎는다.
- compress: 자동 음량 조절(AGC) 흉내. 빠른 압축기.
- add_noise: 방 잡음 흉내 (분홍색에 가까운 잡음, 정해진 SNR).
"""
import numpy as np
from scipy.signal import butter, get_window, lfilter, sosfilt

SR = 22050


def time_stretch(y: np.ndarray, rate: float, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """rate > 1이면 빨라진다(길이 1/rate). 위상 보코더 (프레임 사이 위상 진행을 보존)."""
    y = np.asarray(y, dtype=np.float64)
    win = get_window("hann", n_fft)
    pad = np.concatenate([np.zeros(n_fft), y, np.zeros(n_fft)])
    frames = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop] * win
    X = np.fft.rfft(frames, axis=1)
    n = X.shape[0]
    steps = np.arange(0, n - 1, rate)
    omega = 2 * np.pi * hop * np.arange(X.shape[1]) / n_fft
    phase = np.angle(X[0])
    out_frames = np.empty((len(steps), X.shape[1]), dtype=np.complex128)
    for i, s in enumerate(steps):
        k = int(s)
        frac = s - k
        mag = (1 - frac) * np.abs(X[k]) + frac * np.abs(X[k + 1])
        out_frames[i] = mag * np.exp(1j * phase)
        dphi = np.angle(X[k + 1]) - np.angle(X[k]) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))
        phase = phase + omega + dphi
    out = np.zeros(len(steps) * hop + n_fft)
    norm = np.zeros_like(out)
    for i, f in enumerate(np.fft.irfft(out_frames, n=n_fft, axis=1)):
        out[i * hop:i * hop + n_fft] += f * win
        norm[i * hop:i * hop + n_fft] += win ** 2
    out /= np.maximum(norm, 1e-8)
    target = int(round(len(y) / rate))
    return out[n_fft:n_fft + target].astype(np.float32)


def phone_mic(y: np.ndarray, lo: float = 150.0, hi: float = 7000.0) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    return sosfilt(sos, np.asarray(y, dtype=np.float64)).astype(np.float32)


def compress(y: np.ndarray, threshold_db: float = -30.0, ratio: float = 4.0, attack_s: float = 0.01,
             release_s: float = 0.3, makeup_db: float = 10.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    env = np.empty_like(y)
    a_att = np.exp(-1.0 / (attack_s * SR))
    a_rel = np.exp(-1.0 / (release_s * SR))
    e = 0.0
    x = np.abs(y)
    for i in range(len(y)):  # 소리 길이만큼 도는 단순 루프 (시험용)
        a = a_att if x[i] > e else a_rel
        e = a * e + (1 - a) * x[i]
        env[i] = e
    lvl = 20 * np.log10(env + 1e-9)
    over = np.maximum(lvl - threshold_db, 0.0)
    gain_db = -over * (1 - 1 / ratio) + makeup_db
    out = y * 10 ** (gain_db / 20)
    peak = np.abs(out).max()
    return (out / peak * 0.99 if peak > 0.99 else out).astype(np.float32)


def add_noise(y: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(len(y))
    pink = lfilter([1.0], [1.0, -0.95], w)  # 저역이 많은 잡음
    p_sig = np.mean(np.asarray(y, dtype=np.float64) ** 2)
    pink *= np.sqrt(p_sig / (np.mean(pink ** 2) * 10 ** (snr_db / 10)))
    return (y + pink).astype(np.float32)


def remove_drums(y: np.ndarray, n_fft: int = 2048, hop: int = 512, time_win: int = 31, freq_win: int = 31,
                 power: float = 2.0) -> np.ndarray:
    """비인과 HPSS로 타악 성분을 빼서 '드럼 없이 치는 구간'을 흉내 낸다 (R13 시험용).

    시간축 중앙값(화성)과 주파수축 중앙값(타악)으로 부드러운 마스크를 만들어 화성 성분만 되돌린다.
    """
    from scipy.ndimage import median_filter
    y = np.asarray(y, dtype=np.float64)
    win = get_window("hann", n_fft)
    pad = np.concatenate([np.zeros(n_fft), y, np.zeros(n_fft)])
    frames = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop] * win
    X = np.fft.rfft(frames, axis=1)
    S = np.abs(X)
    H = median_filter(S, size=(time_win, 1), mode="nearest")
    P = median_filter(S, size=(1, freq_win), mode="nearest")
    mask = H ** power / (H ** power + P ** power + 1e-20)
    frames_out = np.fft.irfft(X * mask, n=n_fft, axis=1)
    out = np.zeros(len(pad))
    norm = np.zeros(len(pad))
    for i, f in enumerate(frames_out):
        out[i * hop:i * hop + n_fft] += f * win
        norm[i * hop:i * hop + n_fft] += win ** 2
    out /= np.maximum(norm, 1e-8)
    return out[n_fft:n_fft + len(y)].astype(np.float32)
