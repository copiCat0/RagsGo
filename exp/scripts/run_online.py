"""Task 6: 온라인 진단. 결과는 진단용이고 정답이 아니다.

합주 녹음을 인과 처리(미래 프레임 안 씀)로 원곡에 맞춰 가며,
Task 4의 오프라인 경로와 얼마나 다른지 잰다.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from ragsgo_exp import features as F  # noqa: E402
from ragsgo_exp.dataio import RESULTS, SONGS, load_features  # noqa: E402
from ragsgo_exp.online import online_align  # noqa: E402

RATE = F.FEAT_RATE
# 템포 변화 벌점. explore_online.py에서 띵바츄로 골랐다 (0.05는 위치가 자주 튐). 같은 곡으로 고른 값이라 과적합 위험이 있다.
TEMPO_PENALTY = 1.0


def stats(err_sec: np.ndarray, beat: float, bar: float) -> dict:
    a = np.abs(err_sec)
    return {
        "median_abs_sec": round(float(np.median(a)), 3),
        "p90_abs_sec": round(float(np.percentile(a, 90)), 3),
        "p95_abs_sec": round(float(np.percentile(a, 95)), 3),
        "max_abs_sec": round(float(a.max()), 3),
        "median_abs_beats": round(float(np.median(a) / beat), 3),
        "p95_abs_beats": round(float(np.percentile(a, 95) / beat), 3),
        "within_half_beat": round(float(np.mean(a < 0.5 * beat)), 3),
        "within_half_bar": round(float(np.mean(a < 0.5 * bar)), 3),
    }


def main() -> None:
    summary = {"note": "진단용. 비교 대상은 오프라인 DTW 경로이며 정답이 아니다.", "songs": {}}
    for song, meta in SONGS.items():
        live = load_features(song, "live", causal=True, hpss=True)
        ref = load_features(song, "orig_nodrums", causal=False, hpss=False)
        m = np.load(RESULTS / f"offline_{song}_map.npz")
        live_start, live_end = int(m["live_start"]), int(m["live_end"])

        shift = int(m["key_shift"])  # 오프라인에서 찾은 반음 이동 (R9). 앱에서는 곡 준비 때 정한다
        ref_c = np.roll(ref.chroma, shift, axis=0)
        begin = max(0, live_start - int(1.0 * RATE))  # 곡 시작 1초 전에 시작 버튼
        end = min(live.chroma.shape[1], live_end + 1)
        est = online_align(live.chroma[:, begin:end], ref_c, silent=live.silent[begin:end],
                           p_stay=TEMPO_PENALTY, p_skip=TEMPO_PENALTY, start_window=int(5 * RATE))

        idx = np.arange(begin, end)
        ok = (idx >= live_start) & (idx <= live_end)
        ref_off = np.interp(idx[ok], m["live_frames"], m["ref_of_live"])
        err = (est[ok] - ref_off) / RATE  # 원곡 기준 초. 음수면 온라인이 뒤처짐
        beat = 60.0 / meta["bpm"]
        bar = beat * meta["beats_per_bar"]
        bias = float(np.median(err))
        jumps = int(np.sum(np.abs(np.diff(est)) > 5 * RATE))

        summary["songs"][song] = {
            "title": meta["title"],
            "key_shift_semitones": shift,
            "tempo_penalty": TEMPO_PENALTY,
            "beat_sec": round(beat, 3),
            "frames_compared": int(ok.sum()),
            "raw": stats(err, beat, bar),
            "median_signed_error_sec": round(bias, 3),
            "after_removing_constant_lag": stats(err - bias, beat, bar),
            "jumps_over_5s": jumps,
        }

        t_live = (idx[ok] + 0.5) / RATE
        with open(RESULTS / f"online_{song}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rehearsal_sec", "online_original_sec", "offline_original_sec", "diff_sec", "diff_beats"])
            for t, e_on, e_off, d in zip(t_live, est[ok], ref_off, err):
                w.writerow([round(t, 2), round((e_on + 0.5) / RATE, 2), round((e_off + 0.5) / RATE, 2),
                            round(d, 3), round(d / beat, 3)])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_live, err / beat, lw=0.8)
        for y, style in ((0.5, ":"), (-0.5, ":"), (2, "--"), (-2, "--")):
            ax.axhline(y, ls=style, lw=0.8, color="gray")
        ax.set_ylim(-8, 8)
        ax.set_xlabel("rehearsal recording (s)")
        ax.set_ylabel("online - offline (beats)")
        ax.set_title(f"{song}: online vs offline (diagnostic). dotted = +-0.5 beat, dashed = +-0.5 bar")
        fig.tight_layout()
        fig.savefig(RESULTS / f"online_{song}.png", dpi=110)
        plt.close(fig)

    (RESULTS / "online_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
