"""Phase 1-b 인과 처리 전체 (소리 배열 하나를 받아 박마다 원곡 박 번호를 낸다).

순서: 온셋 → 박 추적(HMM) → 연주 중/멈춤 판단 → 박 사이 크로마(인과 HPSS) → 박 단위 정렬.
- 멈춘 동안 알게 된 박은 버린다. 멈춤 직후 첫 박(재개 박)에서는 어느 위치로든 거의 벌점 없이 뛸 수 있다.
- 멈춤을 사이에 둔 박 구간의 크로마는 쓰지 않는다 (valid=False).
- 곡 맨 처음 재개(시작 버튼 뒤 첫 연주)는 start_known이면 곡 첫 박으로 본다.
시각은 모두 입력 소리 기준 초다.
"""
import numpy as np

from . import beats as B
from . import features as F
from .beat_align import beat_online_align

STFT_HALF_S = F.N_FFT / 2 / F.SR  # 박 직전 프레임이 끝나려면 박 뒤로 이만큼 더 들어야 한다 (약 93 ms)


def reference_beat_chroma(y_ref: np.ndarray, grid: np.ndarray, hpss: bool = False) -> np.ndarray:
    """원곡(드럼 뺀 소리)의 박 사이 크로마 (12, len(grid)). 원곡 쪽이라 비인과 처리다."""
    fc, ft, _ = F.frame_chroma(y_ref, causal=False, hpss=hpss)
    period = np.median(np.diff(grid))
    return B.pool_between(fc, ft, np.concatenate([grid, [grid[-1] + period]]))


def run_pipeline(y: np.ndarray, ref_bs: np.ndarray, bpm: float, p_stay: float = 4.0, p_skip: float = 4.0,
                 p_jump: float = 30.0, p_jump_resume: float = 0.5, start_known: bool = True, resume_cost=None,
                 level: str = "running", activity_rise_db: float = 20.0, resume_hint=None):
    """resume_cost: 함수 f(멈춘 위치(원곡 박, 0부터), 원곡 박 수, 몇 번째 재시작) -> (N,) 위치별 뛰기 벌점.
    주면 재개 박에서 p_jump_resume 대신 쓴다 (구간 탭, 단락 시작 가중 등). None이면 어디로든 p_jump_resume.
    activity_rise_db: 연주 시작으로 볼 음량 상승 (방 잡음 바닥 기준). 잡음이 큰 방에서는 낮춰야 한다.
    resume_hint: 함수 f(몇 번째 재시작) -> 다시 칠 마디 첫 박의 원곡 박 번호(0부터) 또는 None.
    사용자가 다시 칠 마디를 탭한 경우다. 연주를 다시 시작한 순간을 그 박으로 보고 위치를 새로 시작한다
    (한 마디 어긋남(±4박)은 작은 비용으로 허용)."""
    odf, _ = B.onset_strength(y)
    tr = B.BeatTracker(bpm=bpm).run(odf)
    times, on, resumes = B.music_activity(y, rise_db=activity_rise_db, return_starts=True)
    fc, ft, _ = F.frame_chroma(y, causal=True, hpss=True, level=level)

    bt, rep, per, stg = tr.beat_times, tr.beat_report_times, tr.beat_periods, tr.beat_strength
    # 박을 알게 된 시각에 '연주 중'이었던 박만 쓴다
    idx = np.clip(np.searchsorted(times, rep), 0, len(on) - 1)
    keep = on[idx]
    if len(resumes):
        keep &= bt >= resumes[0] - 0.15
    bt, rep, per, stg = bt[keep], rep[keep], per[keep], stg[keep]
    n = len(bt)
    empty = {"beats": bt, "ready": rep, "period": per, "strength": stg, "number": np.zeros(0, int),
             "margin": np.zeros(0),
             "resume_beat": np.zeros(0, bool), "resumes": resumes, "all_beats": tr.beat_times,
             "all_reports": tr.beat_report_times, "activity": (times, on)}
    if n < 2:
        return empty

    # 재개 박: 각 재개 시각 뒤 첫 박
    resume_beat = np.zeros(n, dtype=bool)
    resume_time_of = {}
    for r in resumes:
        j = int(np.searchsorted(bt, r - 0.15))
        if j < n:
            resume_beat[j] = True
            resume_time_of[j] = r

    live_bs = B.pool_between(fc, ft, bt)          # 열 c = 박 c ~ 박 c+1
    valid = ~resume_beat[1:]                       # 끝 박이 재개 박이면 그 구간은 멈춤을 사이에 둔다
    pj = np.full(n - 1, p_jump)
    pj[resume_beat[:-1]] = p_jump_resume           # 재개 박에서 시작하는 첫 구간
    first_is_start = resume_beat[0]
    start_index = 0
    if start_known and first_is_start:
        # 곡 시작: 연주가 시작된 시각이 곡 첫 박이다. 박 추적기가 첫 박을 놓쳤으면 그만큼 번호를 민다.
        start_index = max(0, int(round((bt[0] - resumes[0]) / per[0])))
        pj[0] = p_jump
        start_window = 1
    else:
        start_window = ref_bs.shape[1]
    jump_fn = None
    resume_cols = [c for c in np.where(resume_beat[:-1])[0] if not (c == 0 and first_is_start)]
    order = {int(c): i + 1 for i, c in enumerate(resume_cols)}
    reset_at = {}
    if resume_hint is not None:
        N = ref_bs.shape[1]
        for c, i in order.items():
            h = resume_hint(i)
            if h is None:
                continue
            target = int(h) + max(0, int(round((bt[c] - resume_time_of[c]) / per[c])))
            v = np.full(N, 1e9)
            for d, cost in ((0, 0.0), (-4, 1.0), (4, 1.0)):
                if 0 <= target + d < N:
                    v[target + d] = cost
            reset_at[c] = v
    if resume_cost is not None:

        def jump_fn(k, est_so_far):
            if k not in order or len(est_so_far) < 2:
                return None
            return resume_cost(int(est_so_far[-2]), ref_bs.shape[1], order[k])
    est, margin = beat_online_align(live_bs, ref_bs, p_stay=p_stay, p_skip=p_skip, start_window=start_window,
                                    valid=valid, p_jump=pj, start_index=start_index, jump_cost_at=jump_fn,
                                    reset_at=reset_at or None)
    number = np.empty(n, dtype=np.int64)
    number[1:] = est + 2
    number[0] = number[1] - 1
    mg = np.concatenate([[0.0], margin])
    ready = np.maximum(rep, bt + STFT_HALF_S)
    return {"beats": bt, "ready": ready, "period": per, "strength": stg, "number": number, "margin": mg,
            "resume_beat": resume_beat, "resumes": resumes, "all_beats": tr.beat_times,
            "all_reports": tr.beat_report_times, "activity": (times, on)}
