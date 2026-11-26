"""Phase 1-b: 박 추적 + 박 단위 정렬을 임시 정답과 비교한다 (띵바츄).

- 박 추적기: 인과 HMM (beats.BeatTracker). 사전 템포 = 원곡 템포.
- 박 단위 정렬: 박 사이 크로마를 원곡 박 사이 크로마와 맞춘다 (beat_align).
- 비교용: Phase 1-a의 프레임 단위 온라인 정렬기도 같은 임시 정답으로 다시 잰다.
결과는 임시 정답 기준이다. 사용자가 클릭 확인을 마치면 다시 잰다.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from ragsgo_exp import beats as B  # noqa: E402
from ragsgo_exp import features as F  # noqa: E402
from ragsgo_exp.beat_align import beat_online_align  # noqa: E402
from ragsgo_exp.dataio import RESULTS, SONGS, load_features, load_wav  # noqa: E402
from ragsgo_exp.online import online_align  # noqa: E402
from ragsgo_exp.pipeline import run_pipeline  # noqa: E402

PRE_ROLL = 1.0  # 곡 첫 박 1초 전에 시작 버튼을 누른다고 본다


def load_gt(song):
    rows = list(csv.DictReader(open(RESULTS / f"gt_{song}.csv")))
    g = {k: np.array([float(r[k]) for r in rows]) for k in ("beat", "bar", "beat_in_bar", "page", "orig_sec",
                                                              "live_sec")}
    return g


def f_measure(est, ref, tol=0.07):
    est, ref = np.asarray(est), np.asarray(ref)
    used = np.zeros(len(ref), bool)
    hit = 0
    for e in est:
        d = np.abs(ref - e)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            hit += 1
    p, r = hit / max(len(est), 1), hit / max(len(ref), 1)
    return (0.0 if hit == 0 else 2 * p * r / (p + r)), p, r


STFT_HALF_S = F.N_FFT / 2 / F.SR  # 박 직전 프레임이 끝나려면 박 뒤로 이만큼 더 들어야 한다 (약 93 ms)


def track_song(song, start, bpm, ref_bs, p_stay=4.0, p_skip=4.0, start_window=1, p_jump=30.0):
    """시작 버튼(start, 녹음 기준 초)부터 인과 처리 (ragsgo_exp.pipeline). 시각은 녹음 기준 초로 바꿔 낸다.

    start_window=1이면 곡을 처음부터 친다고 아는 경우(연주 시작 = 곡 첫 박), 아니면 시작 위치를 모르는 경우.
    """
    y = load_wav(f"{song}_live")
    yc = y[int(start * F.SR):]
    out = run_pipeline(yc, ref_bs, bpm, p_stay=p_stay, p_skip=p_skip, p_jump=p_jump,
                       start_known=(int(start_window) == 1))
    res = dict(out)
    for k in ("beats", "ready", "all_beats", "all_reports", "resumes"):
        res[k] = out[k] + start
    res["music_start"] = float(out["resumes"][0] + start) if len(out["resumes"]) else None
    return res


def ref_beat_chroma(song, grid):
    y = load_wav(f"{song}_orig_nodrums")
    fc, ft, _ = F.frame_chroma(y, causal=False, hpss=False)
    period = np.median(np.diff(grid))
    return B.pool_between(fc, ft, np.concatenate([grid, [grid[-1] + period]]))


def evaluate_numbers(bt, num_est, gt_live, gt_beat, beat_s):
    """추적 박마다 정답 박 번호(가장 가까운 정답 박, 반 박 안)와 비교한다."""
    j = np.searchsorted(gt_live, bt)
    j = np.clip(j, 1, len(gt_live) - 1)
    near = np.where(np.abs(gt_live[j - 1] - bt) < np.abs(gt_live[j] - bt), j - 1, j)
    dist = np.abs(gt_live[near] - bt)
    in_song = (bt >= gt_live[0] - 0.5 * beat_s) & (bt <= gt_live[-1] + 0.5 * beat_s)
    matched = in_song & (dist < 0.5 * beat_s)
    err = num_est[matched] - gt_beat[near[matched]]
    return {
        "tracked_beats_in_song": int(in_song.sum()),
        "matched_to_gt_within_half_beat": int(matched.sum()),
        "beat_number_correct": round(float(np.mean(err == 0)), 3),
        "within_1_beat": round(float(np.mean(np.abs(err) <= 1)), 3),
        "within_half_bar": round(float(np.mean(np.abs(err) <= 2)), 3),
        "error_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(err, return_counts=True))},
    }, matched, near


def main(song="tac", p_stay=4.0, p_skip=4.0, start_window=1):
    p_stay, p_skip = float(p_stay), float(p_skip)
    meta = SONGS[song]
    gt = load_gt(song)
    summ = json.loads((RESULTS / f"gt_{song}_summary.json").read_text())
    bpm = summ["linear_bpm"]
    beat_s = 60.0 / bpm
    start = gt["live_sec"][0] - PRE_ROLL
    res = {"note": "임시 정답(사람 확인 전) 기준", "song": song, "start_sec": round(start, 3),
           "prior_bpm": bpm, "p_stay": p_stay, "p_skip": p_skip,
           "start_window": int(start_window)}

    grid = gt["orig_sec"]
    ref_bs = ref_beat_chroma(song, grid)
    tr = track_song(song, start, bpm, ref_bs, p_stay=p_stay, p_skip=p_skip, start_window=int(start_window))
    res["music_start_sec"] = None if tr["music_start"] is None else round(tr["music_start"], 3)

    # 1) 박 추적 (추적기가 낸 모든 박)
    bt_all, rep_all = tr["all_beats"], tr["all_reports"]
    song_gt = gt["live_sec"]
    sel = (bt_all > song_gt[0] - 0.35) & (bt_all < song_gt[-1] + 0.35)
    f, p, r = f_measure(bt_all[sel], song_gt)
    lag = rep_all - bt_all
    res["tracker"] = {"f_measure_70ms": round(f, 3), "precision": round(p, 3), "recall": round(r, 3),
                      "n_tracked_in_song": int(sel.sum()), "n_gt": len(song_gt),
                      "report_lag_ms_median": round(float(np.median(lag) * 1000), 1),
                      "report_lag_ms_p95": round(float(np.percentile(lag, 95) * 1000), 1)}
    hit = np.array([np.min(np.abs(song_gt - b)) <= 0.07 for b in bt_all])
    run = np.convolve(hit.astype(int), np.ones(4, int), mode="valid")
    first = int(np.argmax(run == 4)) if np.any(run == 4) else -1
    res["tracker"]["first_4_hits_at_sec_after_song_start"] = round(float(bt_all[first] - song_gt[0]), 2) \
        if first >= 0 else None
    d = np.array([b - song_gt[np.argmin(np.abs(song_gt - b))] for b in bt_all[sel]])
    d70 = d[np.abs(d) <= 0.07]
    res["tracker"]["timing_error_ms_of_hits"] = {"median": round(float(np.median(d70) * 1000), 1),
                                                 "p95_abs": round(float(np.percentile(np.abs(d70), 95) * 1000), 1)}

    # 2) 박 단위 정렬 (곡 시작 감지 뒤 박만)
    bt, rep, per, num_est, margin = tr["beats"], tr["ready"], tr["period"], tr["number"], tr["margin"]
    ev, matched, near = evaluate_numbers(bt, num_est, song_gt, gt["beat"], beat_s)
    res["beat_align"] = ev

    # 3) 비교: Phase 1-a 프레임 단위 온라인 정렬기를 같은 임시 정답으로
    live10 = load_features(song, "live", causal=True, hpss=True)
    ref10 = load_features(song, "orig_nodrums", causal=False, hpss=False)
    b10 = int(start * F.FEAT_RATE)
    e10 = int((song_gt[-1] + 0.5) * F.FEAT_RATE)
    est10 = online_align(live10.chroma[:, b10:e10], ref10.chroma, silent=live10.silent[b10:e10],
                         p_stay=1.0, p_skip=1.0, start_window=50)
    t10 = (np.arange(b10, e10) + 0.5) / F.FEAT_RATE
    ok10 = (t10 >= song_gt[0]) & (t10 <= song_gt[-1])
    true_orig = np.interp(t10[ok10], song_gt, grid)
    err10 = ((est10[ok10] + 0.5) / F.FEAT_RATE - true_orig) / beat_s
    a = np.abs(err10)
    res["frame_aligner_phase1a_vs_gt"] = {"median_abs_beats": round(float(np.median(a)), 3),
                                          "within_half_beat": round(float(np.mean(a < 0.5)), 3),
                                          "within_half_bar": round(float(np.mean(a < 2)), 3),
                                          "p95_abs_beats": round(float(np.percentile(a, 95)), 2)}
    # 오프라인 경로도 같은 정답으로
    m = np.load(RESULTS / f"offline_{song}_map.npz")
    off = (np.interp(t10[ok10] * F.FEAT_RATE - 0.5, m["live_frames"], m["ref_of_live"]) + 0.5) / F.FEAT_RATE
    ao = np.abs(off - true_orig) / beat_s
    res["offline_path_phase1a_vs_gt"] = {"median_abs_beats": round(float(np.median(ao)), 3),
                                         "within_half_beat": round(float(np.mean(ao < 0.5)), 3),
                                         "within_half_bar": round(float(np.mean(ao < 2)), 3)}

    tag = f"{song}_s{p_stay}_k{p_skip}_w{int(start_window)}"
    with open(RESULTS / f"beats_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["beat_time", "report_time", "period", "est_beat_no", "margin", "gt_beat_no", "gt_time"])
        for k in range(len(bt)):
            g = int(gt["beat"][near[k]]) if matched[k] else ""
            gtt = round(float(song_gt[near[k]]), 4) if matched[k] else ""
            w.writerow([round(float(bt[k]), 4), round(float(rep[k]), 4), round(float(per[k]), 4), int(num_est[k]),
                        round(float(margin[k]), 3), g, gtt])
    (RESULTS / f"beats_{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:])
