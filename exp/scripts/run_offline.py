"""Task 4: 오프라인 진단 정렬. 결과는 진단용이고 정답이 아니다.

원곡(드럼 뺀 것, 앞뒤 무음 제거) 전체를 합주 녹음 안에서 찾는다.
기울기 제한 subsequence DTW (국소 템포 비율 0.5~2배, 코사인 거리)를 쓴다.
- 키·튜닝 차이: 원곡 크로마를 -2~+2 반음 돌려 보고 가장 잘 맞는 것을 쓴다 (R9)
- 같은 곡 확인: 맞는 짝과 틀린 짝의 비용 비교 (U16)
- 합주 녹음에서 곡의 시작과 끝, 평균 템포 비율, 10초 구간별 기울기와 비용 (R11)
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
from ragsgo_exp.offline import subseq_dtw_slope  # noqa: E402

RATE = F.FEAT_RATE
SHIFTS = range(-2, 3)


def sec(k):
    return (np.asarray(k) + 0.5) / RATE  # 특징 프레임 k의 가운데 시각


def nonsilent_range(f: "F.Features"):
    idx = np.where(~f.silent)[0]
    return int(idx[0]), int(idx[-1]) + 1


def best_shift(ref_chroma, live_chroma):
    costs = {k: subseq_dtw_slope(np.roll(ref_chroma, k, axis=0), live_chroma)[1] for k in SHIFTS}
    return min(costs, key=costs.get), {str(k): round(v, 4) for k, v in costs.items()}


def analyse(wp, ref_c, live_c, ref_lo):
    r, l = wp[:, 0], wp[:, 1]
    cell_cost = 1.0 - np.sum(ref_c[:, r] * live_c[:, l], axis=0)
    ref_frames = np.arange(r[0], r[-1] + 1)
    live_of_ref = np.interp(ref_frames, r, l)
    live_frames = np.arange(l[0], l[-1] + 1)
    ref_of_live = np.interp(live_frames, l, r)

    block = int(10 * RATE)
    blocks = []
    for k in range(0, len(ref_frames) - block + 1, block):
        a, b = ref_frames[k], ref_frames[k + block - 1]
        in_block = (r >= a) & (r <= b)
        blocks.append({
            "original_from_sec": round(float(sec(a + ref_lo)), 1),
            "slope": round(float((live_of_ref[k + block - 1] - live_of_ref[k]) / (block - 1)), 3),
            "mean_cost": round(float(cell_cost[in_block].mean()), 3),
        })
    ref_len = (r[-1] - r[0] + 1) / RATE
    live_len = (l[-1] - l[0] + 1) / RATE
    info = {
        "original_used_from_sec": round(float(sec(r[0] + ref_lo)), 2),
        "original_used_to_sec": round(float(sec(r[-1] + ref_lo)), 2),
        "live_song_start_sec": round(float(sec(l[0])), 2),
        "live_song_end_sec": round(float(sec(l[-1])), 2),
        "original_len_sec": round(float(ref_len), 2),
        "live_song_len_sec": round(float(live_len), 2),
        "tempo_ratio_live_over_original": round(float(live_len / ref_len), 3),
        "mean_path_cost": round(float(cell_cost.mean()), 4),
        "blocks_10s": blocks,
    }
    mapping = {"live_frames": live_frames, "ref_of_live": ref_of_live + ref_lo,
               "ref_frames": ref_frames + ref_lo, "live_of_ref": live_of_ref}
    return info, mapping


def plot(song, wp, ref_lo, info):
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 10))
    a1.plot(sec(wp[:, 0] + ref_lo), sec(wp[:, 1]), lw=1)
    x0, y0 = sec(wp[0, 0] + ref_lo), sec(wp[0, 1])
    xs = np.array([x0, x0 + info["original_len_sec"]])
    a1.plot(xs, y0 + (xs - x0), "--", lw=0.8, label="slope 1 (same tempo)")
    a1.set_xlabel("original (s)")
    a1.set_ylabel("rehearsal recording (s)")
    a1.set_title(f"{song}: offline path (diagnostic, not ground truth)")
    a1.legend()
    xs = [b["original_from_sec"] for b in info["blocks_10s"]]
    a2.step(xs, [b["slope"] for b in info["blocks_10s"]], where="post")
    a2.axhline(1.0, ls="--", lw=0.8)
    a2.set_ylabel("local slope")
    a3.step(xs, [b["mean_cost"] for b in info["blocks_10s"]], where="post")
    a3.set_ylabel("mean cosine distance")
    a3.set_xlabel("original (s)")
    fig.tight_layout()
    fig.savefig(RESULTS / f"offline_{song}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    live = {s: load_features(s, "live", causal=False, hpss=True) for s in SONGS}
    ref_full = {s: load_features(s, "orig_nodrums", causal=False, hpss=False) for s in SONGS}
    ref_lo = {s: nonsilent_range(ref_full[s])[0] for s in SONGS}
    ref = {s: ref_full[s].chroma[:, slice(*nonsilent_range(ref_full[s]))] for s in SONGS}

    summary = {"note": "진단용. 오프라인 DTW 결과이며 정답이 아니다.", "pair_cost_best_shift": {},
               "key_shift_cost": {}, "songs": {}}

    for rs in SONGS:  # 같은 곡 확인 (짝마다 가장 잘 맞는 반음 이동으로)
        for ls in SONGS:
            k, costs = best_shift(ref[rs], live[ls].chroma)
            summary["pair_cost_best_shift"][f"original={rs} rehearsal={ls}"] = {"shift": k, "cost": costs[str(k)]}

    for song in SONGS:
        k, costs = best_shift(ref[song], live[song].chroma)
        summary["key_shift_cost"][song] = {"best": k, "costs": costs}
        ref_c = np.roll(ref[song], k, axis=0)
        wp, _ = subseq_dtw_slope(ref_c, live[song].chroma)
        info, m = analyse(wp, ref_c, live[song].chroma, ref_lo[song])
        info["key_shift_semitones"] = k
        summary["songs"][song] = info
        np.savez(RESULTS / f"offline_{song}_map.npz", key_shift=k,
                 live_start=wp[0, 1], live_end=wp[-1, 1], **m)
        with open(RESULTS / f"offline_{song}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["original_sec", "rehearsal_sec"])
            for r_i, l_i in wp:
                w.writerow([round(float(sec(r_i + ref_lo[song])), 2), round(float(sec(l_i)), 2)])
        plot(song, wp, ref_lo[song], info)

    (RESULTS / "offline_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
