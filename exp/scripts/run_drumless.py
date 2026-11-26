"""Phase 1-b R13 시험: 드럼이 빠진 구간에서 박 번호를 유지하는가 (띵바츄, 임시 정답 기준, 흉내).

- whole: 곡 전체에서 타악 성분을 뺌 (비인과 HPSS). 박 추적이 화성 악기의 온셋에만 기댄다.
- bars35-50: 35~50마디(16마디, 약 45초)만 타악 성분을 뺌. 드럼이 쉬는 브리지나 조용한 대목 흉내.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "scripts"))
from ragsgo_exp import augment as A  # noqa: E402
from ragsgo_exp import features as F  # noqa: E402
from ragsgo_exp.dataio import RESULTS, SONGS, load_wav  # noqa: E402
from ragsgo_exp.pipeline import run_pipeline  # noqa: E402
from ragsgo_exp.turns import BeatEvent  # noqa: E402
import run_beats as RB  # noqa: E402
import run_turns as RT  # noqa: E402


def main(song="tac"):
    meta = SONGS[song]
    gt = RB.load_gt(song)
    bpm = json.loads((RESULTS / f"gt_{song}_summary.json").read_text())["linear_bpm"]
    start = gt["live_sec"][0] - RB.PRE_ROLL
    y = load_wav(f"{song}_live")[int(start * F.SR):]
    ref_bs = RB.ref_beat_chroma(song, gt["orig_sec"])
    nod = A.remove_drums(y)
    a = int((gt["live_sec"][(35 - 1) * 4] - start) * F.SR)
    b = int((gt["live_sec"][(51 - 1) * 4] - start) * F.SR)
    part = y.copy()
    fade = int(0.2 * F.SR)
    w = np.ones(b - a)
    w[:fade] = np.linspace(0, 1, fade)
    w[-fade:] = np.linspace(1, 0, fade)
    part[a:b] = y[a:b] * (1 - w) + nod[a:b] * w
    rows = []
    g = dict(gt)
    g["live_sec"] = gt["live_sec"] - start
    for name, yy in (("whole", nod), ("bars35-50", part)):
        out = run_pipeline(yy, ref_bs, bpm, p_stay=4.0, p_skip=4.0, p_jump=30.0)
        sel = (out["all_beats"] > g["live_sec"][0] - 0.35) & (out["all_beats"] < g["live_sec"][-1] + 0.35)
        f, _, _ = RB.f_measure(out["all_beats"][sel], g["live_sec"])
        ev, matched, near = RB.evaluate_numbers(out["beats"], out["number"], g["live_sec"], g["beat"], 60 / bpm)
        events = [BeatEvent(float(out["beats"][k]), float(out["ready"][k]), float(out["period"][k]),
                            int(out["number"][k]), float(out["margin"][k]), float(out["strength"][k]))
                  for k in range(len(out["beats"]))]
        pages = RT.evaluate(g, bpm, events, [x + 1 for x in meta["page_last_bars"][:-1]], meta["beats_per_bar"], 0.0,
                            0.4)
        allb = RT.evaluate(g, bpm, events, list(range(4, meta["bars"] + 1)), meta["beats_per_bar"], 0.0, 0.4)
        allb0 = RT.evaluate(g, bpm, events, list(range(4, meta["bars"] + 1)), meta["beats_per_bar"], 0.0, 0.0)
        # 드럼 없는 구간 안팎의 박 추적 정확도
        in_sec = (g["live_sec"] >= (a / F.SR)) & (g["live_sec"] < (b / F.SR))
        f_in, _, _ = RB.f_measure(out["all_beats"][(out["all_beats"] >= a / F.SR) & (out["all_beats"] < b / F.SR)],
                                  g["live_sec"][in_sec])
        row = {"condition": name, "tracker_f70_song": round(f, 3), "tracker_f70_bars35_50": round(f_in, 3),
               "beat_number_correct": ev["beat_number_correct"], "error_counts": ev["error_counts"],
               "pages_gate0.4": RT.tally(pages), "all_bar_lines_gate0.4": RT.tally(allb),
               "all_bar_lines_no_gate": RT.tally(allb0),
               "failures": [r for r in allb if r["verdict"] != "ok"]}
        rows.append(row)
        print(name, {k: v for k, v in row.items() if k != "failures"}, flush=True)
        for r in row["failures"]:
            print("   ", r["bar"], r["verdict"], r["error_beats"], r["reason"])
    (RESULTS / f"drumless_{song}.json").write_text(json.dumps(
        {"note": "드럼 없는 구간 흉내 (비인과 HPSS로 타악 성분 제거), 임시 정답 기준", "rows": rows}, indent=2,
        ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
