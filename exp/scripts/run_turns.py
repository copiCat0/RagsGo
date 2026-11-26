"""Phase 1-b: 넘김 예측 절차를 띵바츄 합주 녹음에 돌려 본다 (임시 정답 기준).

- 실제 페이지 끝 5곳(15, 35, 55, 75, 95마디 시작)과, 모든 마디선(4~98마디 시작)을 넘김 후보로 본 경우를 잰다.
- 판정 (U11): |넘긴 시각 - 정답 시각| <= 1박이면 성공. "직접"은 자동 넘김 포기로 따로 센다.
- 조건: 시작 창(곡 시작 위치를 얼마나 아는가), 정렬 벌점, 확신 여유 기준.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "scripts"))
from ragsgo_exp.beat_align import beat_online_align  # noqa: E402
from ragsgo_exp.dataio import RESULTS, SONGS  # noqa: E402
from ragsgo_exp.turns import BeatEvent, simulate_turn  # noqa: E402
import run_beats as RB  # noqa: E402


def pipeline(song, start_window, p_stay, p_skip):
    gt = RB.load_gt(song)
    summ = json.loads((RESULTS / f"gt_{song}_summary.json").read_text())
    bpm = summ["linear_bpm"]
    start = gt["live_sec"][0] - RB.PRE_ROLL
    ref_bs = RB.ref_beat_chroma(song, gt["orig_sec"])
    tr = RB.track_song(song, start, bpm, ref_bs, p_stay=p_stay, p_skip=p_skip, start_window=start_window)
    events = [BeatEvent(float(tr["beats"][k]), float(tr["ready"][k]), float(tr["period"][k]),
                        int(tr["number"][k]), float(tr["margin"][k]), float(tr["strength"][k]))
              for k in range(len(tr["beats"]))]
    return gt, bpm, events


def evaluate(gt, bpm, events, bars, beats_per_bar, min_margin, min_strength=0.0):
    beat_s = 60.0 / bpm
    out = []
    for bar in bars:
        turn_beat = (bar - 1) * beats_per_bar + 1
        true_t = gt["live_sec"][turn_beat - 1]
        r = simulate_turn(events, turn_beat, beats_per_bar, min_margin=min_margin, min_strength=min_strength)
        err = None if r.fire_time is None else (r.fire_time - true_t) / beat_s
        if r.status == "auto":
            verdict = "ok" if abs(err) <= 1.0 else ("early" if err < 0 else "late")
        else:
            verdict = r.status
        out.append({"bar": bar, "turn_beat": turn_beat, "true_sec": round(float(true_t), 3),
                    "status": r.status, "verdict": verdict,
                    "error_beats": None if err is None else round(float(err), 3), "reason": r.reason})
    return out


def tally(rows):
    v = [r["verdict"] for r in rows]
    errs = np.array([r["error_beats"] for r in rows if r["verdict"] == "ok"])
    return {"n": len(rows), "ok": v.count("ok"), "early": v.count("early"), "late": v.count("late"),
            "manual": v.count("manual"), "missed": v.count("missed"),
            "ok_timing_error_beats": None if len(errs) == 0 else {
                "median": round(float(np.median(errs)), 3), "p95_abs": round(float(np.percentile(np.abs(errs), 95)), 3),
                "max_abs": round(float(np.abs(errs).max()), 3)}}


def main(song="tac"):
    meta = SONGS[song]
    bpb = meta["beats_per_bar"]
    page_turn_bars = [b + 1 for b in meta["page_last_bars"][:-1]]
    all_bars = list(range(4, meta["bars"] + 1))
    results = {"note": "임시 정답(사람 확인 전) 기준. 출력 지연 0으로 가정", "song": song, "conditions": []}
    for start_window in (1, 10000):  # 1: 곡을 처음부터 친다고 앎, 10000: 시작 위치 모름
        for pen in (1.0, 2.0, 4.0):
            gt, bpm, events = pipeline(song, start_window, pen, pen)
            for min_margin, min_strength in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.4)):
                pages = evaluate(gt, bpm, events, page_turn_bars, bpb, min_margin, min_strength)
                allb = evaluate(gt, bpm, events, all_bars, bpb, min_margin, min_strength)
                cond = {"start_window_beats": start_window, "penalty": pen, "min_margin": min_margin,
                        "min_strength": min_strength,
                        "page_turns": tally(pages), "all_bar_lines": tally(allb),
                        "page_turn_detail": pages,
                        "all_bar_failures": [r for r in allb if r["verdict"] != "ok"]}
                results["conditions"].append(cond)
                print(start_window, pen, min_margin, min_strength, "pages", tally(pages), "| all", tally(allb), flush=True)
    (RESULTS / f"turns_{song}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
