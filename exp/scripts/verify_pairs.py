"""discover_pairs.py의 후보 짝을 더 엄격하게 확인한다 (추정, 사용자 확인 필요).

평균 비용만으로는 조성과 화음 분포가 비슷한 다른 곡도 낮게 나온다. 그래서 두 가지를 더 본다.
1. 뒤집기 시험: 원곡을 시간 역순으로 뒤집어도 화음 분포는 같다. 같은 곡이면 순서가 맞아야 하므로
   뒤집은 쪽 비용이 분명히 커진다. 다른 곡이면 둘이 비슷하다.
2. 경로 직선성: 같은 곡을 같은 템포로 쳤다면 10초 구간별 기울기가 1 근처에 모인다.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "scripts"))
from ragsgo_exp.dataio import RESULTS  # noqa: E402
from ragsgo_exp.offline import subseq_dtw_slope  # noqa: E402
import discover_pairs as dp  # noqa: E402

RATE = 5.0  # discover_pairs 특징은 5 Hz

CANDIDATES = [
    ("tac(TaU)", "1030:ThinkAboutChu_02"),      # 양성 대조 (확인된 짝)
    ("tac(TaU)", "1030:HiBully_01"),            # 음성 대조
    ("n103(악보폴더)", "합주녹음:103"),
    ("n103(악보폴더)", "1030:해초_01"),
    ("n103(악보폴더)", "1030:멋진헛간_02"),
    ("자처", "합주녹음:크립2"),
    ("자처", "합주녹음:Vlad3"),
    ("히카리:02. Rewrite", "1030:멋진헛간_02"),
    ("히카리:anime-ost_naruto-shippuden-opening-16-silhouette-by-kana-boon", "1030:해초_01"),
    ("히카리:唱- Ado", "1030:BeforeYouGo_02"),
    ("히카리:神っぽいな - peenokioP ", "합주녹음:Vlad3"),
]


def load(kind, key):
    keys = list(dp.ORIGINALS) if kind == "orig" else list(dp.RECORDINGS)
    i = keys.index(key)
    z = np.load(dp.DATA / f"disc_{kind if kind == 'orig' else 'rec'}{i:02d}.npz")
    return dp.trim(z["chroma"], z["silent"])


def run_dir(ref, live):
    wp, cost = subseq_dtw_slope(ref, live)
    r, l = wp[:, 0], wp[:, 1]
    block = int(10 * RATE)
    slopes = []
    ref_frames = np.arange(r[0], r[-1] + 1)
    live_of_ref = np.interp(ref_frames, r, l)
    for k in range(0, len(ref_frames) - block + 1, block):
        slopes.append((live_of_ref[k + block - 1] - live_of_ref[k]) / (block - 1))
    slopes = np.array(slopes)
    return cost, slopes


def check(ok, rk):
    o, rec = load("orig", ok), load("rec", rk)
    best = None
    for k in dp.SHIFTS:
        r = np.roll(o, k, axis=0)
        for direction, (a, b) in (("orig_in_rec", (r, rec)), ("rec_in_orig", (rec, r))):
            if b.shape[1] < 0.55 * a.shape[1]:
                continue
            cost, slopes = run_dir(a, b)
            if best is None or cost < best["cost"]:
                best = {"shift": k, "direction": direction, "cost": cost, "slopes": slopes, "a": a, "b": b}
    rev_cost, _ = run_dir(best["a"][:, ::-1], best["b"]) if best["direction"] == "orig_in_rec" \
        else run_dir(best["a"], best["b"][:, ::-1])
    s = best["slopes"]
    return {
        "original": ok, "recording": rk, "shift": best["shift"], "direction": best["direction"],
        "cost": round(best["cost"], 4), "reversed_cost": round(rev_cost, 4),
        "reversal_gap": round(rev_cost - best["cost"], 4),
        "slope_median": round(float(np.median(s)), 3) if len(s) else None,
        "frac_blocks_slope_0.8_1.25": round(float(np.mean((s > 0.8) & (s < 1.25))), 2) if len(s) else None,
    }


def main():
    rows = []
    for ok, rk in CANDIDATES:
        row = check(ok, rk)
        rows.append(row)
        print(row, flush=True)
    (RESULTS / "verify_pairs.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def main_all():
    """모든 짝에 대해, discover_pairs에서 고른 반음 이동과 방향으로 뒤집기 시험만 한다."""
    d = json.loads((RESULTS / "discover_pairs.json").read_text())
    rows = []
    for p in d["pairs"]:
        o = np.roll(load("orig", p["original"]), p["shift"], axis=0)
        rec = load("rec", p["recording"])
        if p["direction"] == "orig_in_rec":
            fwd, slopes = run_dir(o, rec)
            rev, _ = run_dir(o[:, ::-1], rec)
        else:
            fwd, slopes = run_dir(rec, o)
            rev, _ = run_dir(rec, o[:, ::-1])
        rows.append({**p, "reversed_cost": round(rev, 4), "reversal_gap": round(rev - fwd, 4),
                     "frac_blocks_slope_0.8_1.25": round(float(np.mean((slopes > 0.8) & (slopes < 1.25))), 2)
                     if len(slopes) else None})
    rows.sort(key=lambda r: -r["reversal_gap"])
    (RESULTS / "verify_pairs_all.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    for r in rows[:15]:
        print(r["reversal_gap"], r["original"], "|", r["recording"], r["cost"], r["shift"], r["direction"],
              r["frac_blocks_slope_0.8_1.25"])


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "all":
    main_all()
