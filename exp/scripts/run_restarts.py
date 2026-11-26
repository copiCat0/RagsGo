"""Phase 1-b: 재시작(R6) 합성 시험. 띵바츄 합주 녹음을 잘라 붙여 '멈춤 → 스틱 카운트 → 앞 구간부터 다시'를 만든다.

- 멈춤: 녹음 앞부분의 조용한 방 소리(0~0.9초)를 이어 붙인다 (2.5초).
- 카운트: 녹음 앞부분의 실제 스틱 카운트(0.6~3.8초, 딸깍 4번)를 그대로 쓴다.
- 다시 칠 위치: 8마디 단락 시작(11, 19, 27, ... 마디). 이 곡은 8마디 화음 고리가 곡 내내 반복되므로
  재탐색에는 가장 어려운 경우다.
- 방식: free = 멈춤 뒤 어디로든 뛸 수 있음(자동), tap = 사용자가 다시 칠 마디를 탭함(그 마디 첫 박에서 새로 시작).
  v1(restarts_tac_v1_free_prior_hintwindow.json)에서 본 '단락 시작 가중'과 '탭 ±2박 창'은 결과가 나빠 뺐다.
- 판정: 다시 친 첫 박부터, 박 번호가 8박 연속 맞기 시작한 박까지 걸린 마디 수 (기획 기준 4마디 이내).
  잘못된 고리(32박 차이)에 붙으면 '다른 고리'로 따로 센다.
정답은 임시 정답(gt_tac.csv)에서 그대로 옮긴다.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "scripts"))
from ragsgo_exp import features as F  # noqa: E402
from ragsgo_exp.dataio import RESULTS, load_wav  # noqa: E402
from ragsgo_exp.pipeline import run_pipeline  # noqa: E402
import run_beats as RB  # noqa: E402

SR = F.SR
PAUSE_S = 2.5
PHRASE_STARTS = [11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91]


def build(y, gt, stop_bar, stop_beat_in_bar, restart_bar, tail_bars=24):
    """반환: 합성 소리, 정답 (시각, 박 번호) 배열, 재개 첫 박 시각."""
    t = gt["live_sec"]
    first = t[0]
    pre = y[int((first - RB.PRE_ROLL) * SR):int((first - 0.02) * SR)]       # 시작 버튼 ~ 곡 첫 박 직전
    stop_idx = (stop_bar - 1) * 4 + (stop_beat_in_bar - 1)                    # 이 박 직전에서 멈춘다 (0부터)
    a_end = t[stop_idx] - 0.02
    seg_a = y[int((first - 0.02) * SR):int(a_end * SR)].copy()
    fade = int(0.05 * SR)
    seg_a[-fade:] *= np.linspace(1, 0, fade)
    room = y[0:int(0.9 * SR)]
    pause = np.tile(room, int(np.ceil(PAUSE_S / 0.9)))[:int(PAUSE_S * SR)]
    count = y[int(0.6 * SR):int(3.8 * SR)]
    r_idx = (restart_bar - 1) * 4
    end_idx = min(len(t) - 1, r_idx + tail_bars * 4)
    seg_b = y[int((t[r_idx] - 0.02) * SR):int(t[end_idx] * SR)]
    audio = np.concatenate([pre, seg_a, pause, count, seg_b]).astype(np.float32)

    off_a = len(pre) / SR - (first - 0.02)                    # 원래 녹음 시각 → 합성 시각 (구간 A)
    beats_a = [(t[i] + off_a, i + 1) for i in range(stop_idx)]
    start_b = (len(pre) + len(seg_a) + len(pause) + len(count)) / SR
    off_b = start_b - (t[r_idx] - 0.02)
    beats_b = [(t[i] + off_b, i + 1) for i in range(r_idx, end_idx)]
    return audio, np.array(beats_a + beats_b), t[r_idx] + off_b


def score(out, truth, resume_t, beat_s):
    bt, num = out["beats"], out["number"]
    tt, tn = truth[:, 0], truth[:, 1].astype(int)

    def want(x):
        j = int(np.argmin(np.abs(tt - x)))
        return tn[j] if abs(tt[j] - x) < 0.5 * beat_s else None

    before = [(n == want(x)) for x, n in zip(bt, num) if x < resume_t - 3.0 and want(x) is not None]
    after = [(x, n, want(x)) for x, n in zip(bt, num) if x >= resume_t - 0.1 and want(x) is not None]
    ok = np.array([n == w for _, n, w in after])
    lock = None
    for k in range(len(ok) - 7):
        if ok[k:k + 8].all():
            lock = k
            break
    other_loop = None
    if lock is None and len(after):
        diffs = np.array([n - w for _, n, w in after[-16:]])
        other_loop = int(np.median(diffs))
    return {"before_correct": round(float(np.mean(before)), 3) if before else None,
            "relock_bars": None if lock is None else round(lock / 4, 2),
            "after_lock_correct": None if lock is None else round(float(ok[lock:].mean()), 3),
            "stuck_offset_beats": other_loop}


def prior_cost(phrase_starts_beat0, back_bars=32):
    """단락 시작 가중: 멈춘 곳보다 앞 back_bars마디 안의 단락 시작 0.3, 마디 첫 박 1.0, 그 밖 4.0."""
    def f(stop, n_ref, _i):
        v = np.full(n_ref, 4.0)
        lo = max(0, stop - back_bars * 4)
        idx = np.arange(lo, min(stop + 1, n_ref))
        v[idx[idx % 4 == 0]] = 1.0
        for p in phrase_starts_beat0:
            if lo <= p <= stop:
                v[p] = 0.3
        return v
    return f


def hint_cost(restart_beat0, tol=2):
    """사용자가 다시 칠 마디를 탭한 경우: 그 마디 첫 박 ±tol박만 허용."""
    def f(_stop, n_ref, _i):
        v = np.full(n_ref, 1e9)
        v[max(0, restart_beat0 - tol):restart_beat0 + tol + 1] = 0.0
        return v
    return f


def main():
    y = load_wav("tac_live")
    gt = RB.load_gt("tac")
    bpm = json.loads((RESULTS / "gt_tac_summary.json").read_text())["linear_bpm"]
    beat_s = 60 / bpm
    ref_bs = RB.ref_beat_chroma("tac", gt["orig_sec"])
    rows = []
    rng = np.random.default_rng(0)
    for restart_bar in PHRASE_STARTS[:-1]:
        for back in (8, 16):
            stop_bar = restart_bar + back + int(rng.integers(0, 4))
            if stop_bar > 96:
                continue
            stop_beat = int(rng.integers(1, 5))
            audio, truth, resume_t = build(y, gt, stop_bar, stop_beat, restart_bar)
            phrase0 = [(b - 1) * 4 for b in PHRASE_STARTS]
            r0 = (restart_bar - 1) * 4
            for mode, kw in (("free", {}), ("tap", {"resume_hint": lambda i, r0=r0: r0})):
                out = run_pipeline(audio, ref_bs, bpm, p_stay=4.0, p_skip=4.0, p_jump=30.0, p_jump_resume=0.5, **kw)
                s = score(out, truth, resume_t, beat_s)
                row = {"stop_bar": stop_bar, "stop_beat": stop_beat, "restart_bar": restart_bar,
                       "mode": mode, "n_resumes_detected": int(len(out["resumes"])), **s}
                rows.append(row)
                print(row, flush=True)
    summary = {}
    for pj in ("free", "tap"):
        rs = [r for r in rows if r["mode"] == pj]
        locks = [r["relock_bars"] for r in rs if r["relock_bars"] is not None]
        summary[str(pj)] = {"n": len(rs), "relocked": len(locks),
                            "relock_within_4_bars": int(sum(1 for v in locks if v <= 4)),
                            "relock_bars_median": None if not locks else float(np.median(locks)),
                            "relock_bars_max": None if not locks else float(np.max(locks)),
                            "stuck_cases": [(r["restart_bar"], r["stuck_offset_beats"]) for r in rs
                                            if r["relock_bars"] is None]}
    out = {"note": "합성 재시작 시험 (띵바츄, 임시 정답 기준)", "summary": summary, "rows": rows}
    (RESULTS / "restarts_tac.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
