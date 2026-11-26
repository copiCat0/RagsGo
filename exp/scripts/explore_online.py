"""온라인 정렬기 진단용 탐색: 특징(인과/가운데 기준)과 템포 변화 벌점을 바꿔 가며 오프라인 경로와 비교한다.

결과는 진단용이고 정답이 아니다. 오프라인 경로가 깔끔한 띵바츄(tac)를 주로 본다.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from ragsgo_exp import features as F  # noqa: E402
from ragsgo_exp.dataio import RESULTS, SONGS, load_features  # noqa: E402
from ragsgo_exp.online import online_align  # noqa: E402

RATE = F.FEAT_RATE


def evaluate(song, causal, penalty):
    meta = SONGS[song]
    live = load_features(song, "live", causal=causal, hpss=True)
    ref = load_features(song, "orig_nodrums", causal=False, hpss=False)
    m = np.load(RESULTS / f"offline_{song}_map.npz")
    ref_c = np.roll(ref.chroma, int(m["key_shift"]), axis=0)
    s, e = int(m["live_start"]), int(m["live_end"])
    begin = max(0, s - int(RATE))
    est = online_align(live.chroma[:, begin:e + 1], ref_c, silent=live.silent[begin:e + 1],
                       p_stay=penalty, p_skip=penalty, start_window=int(5 * RATE))
    idx = np.arange(begin, e + 1)
    ok = idx >= s
    err = (est[ok] - np.interp(idx[ok], m["live_frames"], m["ref_of_live"])) / RATE
    beat = 60.0 / meta["bpm"]
    a = np.abs(err)
    return {
        "median_abs_sec": round(float(np.median(a)), 2),
        "p95_abs_sec": round(float(np.percentile(a, 95)), 2),
        "bias_sec": round(float(np.median(err)), 2),
        "within_half_beat": round(float(np.mean(a < 0.5 * beat)), 3),
        "within_half_bar": round(float(np.mean(a < 2 * beat)), 3),
        "jumps_over_5s": int(np.sum(np.abs(np.diff(est)) > 5 * RATE)),
    }


def main():
    rows = []
    for song in SONGS:
        for causal in (False, True):
            for penalty in (0.05, 0.2, 0.5, 1.0, 2.0):
                r = {"song": song, "live_features": "causal" if causal else "centered", "penalty": penalty}
                r.update(evaluate(song, causal, penalty))
                rows.append(r)
                print(r)
    (RESULTS / "explore_online.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
