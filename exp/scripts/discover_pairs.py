"""Phase 1-b 곡 늘리기: 디스크에 있는 원곡 음원과 합주 녹음을 전부 짝지어 같은 곡을 찾는다.

결과는 추정이다. 맞는 짝으로 보이면 사용자가 들어서 확인한다.
- 원곡: 분리 트랙(드럼 뺀 합)과 히카리 폴더 mp3
- 합주 녹음: 1030 영상, 뮤즈벙 영상, 합주녹음 폴더
- 비용: 기울기 제한 subsequence DTW의 프레임당 평균 코사인 거리, 반음 이동 -2~+2 중 최솟값.
  짧은 녹음(곡 일부)도 찾도록 "원곡 안에서 녹음 찾기" 방향도 계산해서 작은 쪽을 쓴다.
- 판정 보조: 원곡마다 다른 녹음들의 비용 분포에서 z 점수를 낸다.

특징만 exp/data/에 캐시한다. 소리는 저장하지 않는다.
"""
import json
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from ragsgo_exp import features as F  # noqa: E402
from ragsgo_exp.dataio import DATA, RESULTS  # noqa: E402
from ragsgo_exp.offline import subseq_dtw_slope  # noqa: E402

HOBBY = Path("/Users/iwonjun/Documents/03_개인/50_취미/합주")
REC = Path("/Users/iwonjun/Documents/03_개인/60_사이드프로젝트/RagsGO/합주녹음")
STEM_DIR = HOBBY / "1030" / "자처,띵바츄"
SHIFTS = range(-2, 3)


def nfc_glob(folder: Path, prefix: str, suffix: str):
    """macOS 파일 이름은 NFD라서 한글 패턴 glob이 안 맞는다. NFC로 바꿔 비교한다."""
    return sorted(p for p in folder.iterdir()
                  if unicodedata.normalize("NFC", p.name).startswith(prefix) and p.name.endswith(suffix))

ORIGINALS = {
    "tac(TaU)": {"stems": nfc_glob(STEM_DIR, "TaU_", ".wav")},
    "자처": {"stems": nfc_glob(STEM_DIR, "자처_", ".wav")},
    "n103(악보폴더)": {"stems": [HOBBY / "1030" / "악보" / f"{n}.wav" for n in ("bass", "drums", "other", "vocals")]},
}
for p in sorted((HOBBY / "히카리").glob("*.mp3")):
    ORIGINALS[f"히카리:{unicodedata.normalize('NFC', p.stem)}"] = {"file": p}

RECORDINGS = {}
for name in ("BeforeYouGo_01", "BeforeYouGo_02", "FarAway_01", "HiBully_01", "ThinkAboutChu_02",
             "라쿠나벙-존", "멋진헛간_02", "해초_01"):
    RECORDINGS[f"1030:{name}"] = HOBBY / "1030" / f"{name}.mp4"
for p in sorted((HOBBY / "벙" / "뮤즈벙").glob("*.mp4")):
    RECORDINGS[f"뮤즈벙:{unicodedata.normalize('NFC', p.stem)}"] = p
for p in sorted(REC.glob("*.m4a")):
    RECORDINGS[f"합주녹음:{unicodedata.normalize('NFC', p.stem)}"] = p


def decode(path: Path) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", str(F.SR), "-f", "f32le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def to_5hz(chroma: np.ndarray, silent: np.ndarray):
    k = chroma.shape[1] // 2
    c = chroma[:, :2 * k].reshape(12, k, 2).mean(axis=2)
    c = c / np.maximum(np.linalg.norm(c, axis=0), 1e-12)
    s = silent[:2 * k].reshape(k, 2).all(axis=1)
    return c.astype(np.float32), s


def features_for(key: str, spec, cache_tag: str):
    cache = DATA / f"disc_{cache_tag}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["chroma"], z["silent"], float(z["dur"])
    if isinstance(spec, Path):
        y = decode(spec)
        f = F.chroma_features(y, causal=False, hpss=True)
    elif "stems" in spec:
        if not spec["stems"]:
            raise FileNotFoundError(f"{key}: 분리 트랙을 찾지 못함")
        stems = {p.stem: decode(p) for p in spec["stems"]}
        n = min(len(v) for v in stems.values())
        y = np.sum([v[:n] for k, v in stems.items() if "drums" not in k], axis=0)
        f = F.chroma_features(y, causal=False, hpss=False)
    else:
        y = decode(spec["file"])
        f = F.chroma_features(y, causal=False, hpss=True)
    c, s = to_5hz(f.chroma, f.silent)
    dur = len(y) / F.SR
    np.savez(cache, chroma=c, silent=s, dur=dur)
    return c, s, dur


def trim(c, s):
    idx = np.where(~s)[0]
    if len(idx) == 0:
        return c
    return c[:, idx[0]:idx[-1] + 1]


def pair_cost(ref, live):
    """반음 이동별로 두 방향을 계산해 가장 작은 비용과 그때의 이동, 방향을 낸다."""
    best = (np.inf, 0, "")
    n_ref, n_live = ref.shape[1], live.shape[1]
    for k in SHIFTS:
        r = np.roll(ref, k, axis=0)
        if n_live >= 0.55 * n_ref:  # 원곡 전체가 녹음 안에 들어갈 수 있다
            c = subseq_dtw_slope(r, live)[1]
            if c < best[0]:
                best = (c, k, "orig_in_rec")
        if n_ref >= 0.55 * n_live:  # 녹음 전체가 원곡 안에 들어갈 수 있다 (곡 일부만 녹음한 경우)
            c = subseq_dtw_slope(live, r)[1]
            if c < best[0]:
                best = (c, k, "rec_in_orig")
    return best


def main():
    t0 = time.time()
    orig = {}
    for i, (k, spec) in enumerate(ORIGINALS.items()):
        c, s, dur = features_for(k, spec, f"orig{i:02d}")
        orig[k] = (trim(c, s), dur)
        print(f"[{time.time() - t0:6.0f}s] original {k}: {dur:.1f}s", flush=True)
    rec = {}
    for i, (k, p) in enumerate(RECORDINGS.items()):
        c, s, dur = features_for(k, p, f"rec{i:02d}")
        rec[k] = (trim(c, s), dur)
        print(f"[{time.time() - t0:6.0f}s] recording {k}: {dur:.1f}s", flush=True)

    rows = []
    for ok, (oc, od) in orig.items():
        for rk, (rc, rd) in rec.items():
            cost, shift, direction = pair_cost(oc, rc)
            rows.append({"original": ok, "recording": rk, "cost": round(cost, 4), "shift": shift,
                         "direction": direction, "orig_sec": round(od, 1), "rec_sec": round(rd, 1)})
        print(f"[{time.time() - t0:6.0f}s] done {ok}", flush=True)

    ranking = {}
    for ok in orig:
        rs = sorted([r for r in rows if r["original"] == ok], key=lambda r: r["cost"])
        costs = np.array([r["cost"] for r in rs])
        others = costs[1:]
        z = (costs[0] - others.mean()) / (others.std() + 1e-9)
        ranking[ok] = {
            "best": rs[0]["recording"], "best_cost": rs[0]["cost"], "best_shift": rs[0]["shift"],
            "direction": rs[0]["direction"], "second": rs[1]["recording"], "second_cost": rs[1]["cost"],
            "z_vs_others": round(float(z), 2),
            "top5": [(r["recording"], r["cost"], r["shift"]) for r in rs[:5]],
        }
    out = {"note": "추정. 비용이 낮고 z가 크게 음수면 같은 곡일 가능성이 높다. 사용자 확인 필요.",
           "ranking_by_original": ranking, "pairs": rows}
    (RESULTS / "discover_pairs.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for ok, r in ranking.items():
        print(ok, "->", r["best"], r["best_cost"], "shift", r["best_shift"], "z", r["z_vs_others"],
              "| 2nd", r["second"], r["second_cost"])
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
