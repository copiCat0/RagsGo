"""Phase 1-b 정답 초안 (사람이 들어서 확인하기 전까지는 '임시 정답'이다).

1. 원곡 박 격자: 원곡 드럼 트랙에서 DP 박을 뽑는다. 원곡 템포가 일정하면(직선 맞춤 잔차가 작으면)
   직선 격자를 쓰고, 아니면 DP 박을 그대로 쓴다. 마디 1의 첫 박 = 곡의 첫 박 (악보와 첫 소리로 확인).
   킥이 오는 박 위치로 다운비트 위상을 한 번 더 확인한다.
2. 합주 녹음으로 옮기기: 오프라인 DTW 경로(Phase 1-a)로 원곡 박 시각을 합주 녹음 시각으로 바꾼다.
3. 다듬기: 합주 녹음 자체의 DP 박을 쓴다.
   - DP 박 간격이 고르면(빠지거나 더한 박이 없으면) 원곡 첫 박에 가장 가까운 DP 박부터 차례로 센다 ('count').
     옮긴 시각은 시작점을 정하는 데만 쓰고, 나머지는 검증에만 쓴다 (센 박과 옮긴 시각의 차이를 기록).
   - 고르지 않으면 옮긴 시각에서 0.25박 안의 DP 박으로 바꾸고, 없으면 보간한다 ('matched'/'interp').
4. 확인 도구: 합주 녹음에 박 클릭을 입힌 WAV와, 재생하면서 마디·페이지를 보여 주는 HTML (exp/check/).

주의: 2단계가 자동 정렬이라, 확인 전 정답은 평가 대상과 같은 계열의 방법에 기대고 있다 (기획 6장).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from ragsgo_exp import beats as B  # noqa: E402
from ragsgo_exp import features as F  # noqa: E402
from ragsgo_exp.dataio import RESULTS, SONGS, load_wav  # noqa: E402

CHECK = EXP / "check"
RATE = F.FEAT_RATE


def page_of_bar(bar, page_last_bars):
    for p, last in enumerate(page_last_bars, start=1):
        if bar <= last:
            return p
    return len(page_last_bars)


def original_grid(song, meta):
    y = load_wav(f"{song}_orig_drums")
    odf, _ = B.onset_strength(y)
    bpm = B.estimate_tempo(odf, prior_bpm=meta["bpm"])
    dp = B.dp_beats(odf, bpm=bpm)
    i = np.arange(len(dp))
    A = np.vstack([i, np.ones_like(i)]).T.astype(float)
    (period, t0), *_ = np.linalg.lstsq(A, dp, rcond=None)
    resid = dp - (t0 + period * i)
    n_beats = meta["bars"] * meta["beats_per_bar"]
    info = {"tempo_bpm_autocorr": round(bpm, 3), "dp_beats": len(dp), "dp_first": round(float(dp[0]), 3),
            "dp_last": round(float(dp[-1]), 3), "linear_period_s": round(float(period), 5),
            "linear_bpm": round(float(60 / period), 3), "linear_resid_std_ms": round(float(resid.std() * 1000), 1),
            "n_beats_score": n_beats}
    if resid.std() < 0.02:
        grid = t0 + period * np.arange(n_beats)
        info["grid"] = "linear"
    else:
        ext = dp[-1] + np.median(np.diff(dp[-32:])) * np.arange(1, max(0, n_beats - len(dp)) + 1)
        grid = np.concatenate([dp, ext])[:n_beats]
        info["grid"] = "dp"

    # 다운비트 위상 확인: 킥(30~120 Hz 온셋)이 박 번호 % 4 의 어디에 많은가
    N, H = 1024, 256
    frames = np.lib.stride_tricks.sliding_window_view(y, N)[::H] * np.hanning(N)
    X = np.abs(np.fft.rfft(frames, axis=1))
    f = np.fft.rfftfreq(N, 1 / F.SR)
    Lk = np.log1p(100 * X[:, (f >= 30) & (f < 120)] / (N / 2))
    kick = np.zeros(len(Lk))
    kick[1:] = np.maximum(Lk[1:] - Lk[:-1], 0).sum(axis=1)
    te = B.frame_event_time(np.arange(len(kick)))
    k_at = np.array([kick[(te > t - 0.04) & (te < t + 0.04)].max(initial=0) for t in grid])
    info["kick_by_beat_in_bar"] = [round(float(k_at[ph::meta["beats_per_bar"]].mean()), 2)
                                   for ph in range(meta["beats_per_bar"])]
    return grid, info


def to_live(grid_sec, m):
    """원곡 시각(초) → 합주 녹음 시각(초). 오프라인 경로는 10 Hz 특징 프레임 번호로 저장돼 있다.
    경로 밖(곡 맨 앞뒤 몇 프레임)은 기울기 1로 늘린다."""
    rf, lf = m["ref_frames"].astype(float), m["live_of_ref"].astype(float)
    ref_frame = grid_sec * RATE - 0.5
    live_frame = np.interp(ref_frame, rf, lf)
    live_frame = np.where(ref_frame < rf[0], lf[0] + (ref_frame - rf[0]), live_frame)
    live_frame = np.where(ref_frame > rf[-1], lf[-1] + (ref_frame - rf[-1]), live_frame)
    return (live_frame + 0.5) / RATE


def count_based(mapped, live_dp):
    """DP 박이 고르면 첫 박부터 센다. 반환: (정답 시각 또는 None, 시작 DP 박 번호)."""
    ibi = np.diff(live_dp)
    med = np.median(ibi)
    if np.any((ibi < 0.75 * med) | (ibi > 1.33 * med)):
        return None, None
    j0 = int(np.argmin(np.abs(live_dp - mapped[0])))
    if j0 + len(mapped) > len(live_dp):
        return None, j0
    return live_dp[j0:j0 + len(mapped)].copy(), j0


def refine(mapped, live_dp, beat_s):
    out = mapped.copy()
    src = np.array(["interp"] * len(mapped), dtype=object)
    used = set()
    for i, t in enumerate(mapped):
        if np.isnan(t):
            continue
        j = int(np.argmin(np.abs(live_dp - t)))
        if abs(live_dp[j] - t) <= 0.25 * beat_s and j not in used:
            out[i] = live_dp[j]
            src[i] = "matched"
            used.add(j)
    ok = src == "matched"
    idx = np.arange(len(out))
    out[~ok] = np.interp(idx[~ok], idx[ok], out[ok])
    # 순서가 뒤집힌 곳이 있으면 기록만 한다
    return out, src, int(np.sum(np.diff(out) <= 0))


def click_track(y, beat_times, downbeat_mask, sr=F.SR):
    out = y.astype(np.float32).copy()
    peak = float(np.abs(out).max()) or 1.0
    n = int(0.03 * sr)
    tt = np.arange(n) / sr
    env = np.exp(-tt / 0.008)
    hi = 0.35 * peak * np.sin(2 * np.pi * 1760 * tt) * env
    lo = 0.25 * peak * np.sin(2 * np.pi * 1100 * tt) * env
    for t, down in zip(beat_times, downbeat_mask):
        i = int(round(t * sr))
        if 0 <= i and i + n <= len(out):
            out[i:i + n] += (hi if down else lo).astype(np.float32)
    m = float(np.abs(out).max())
    return out * (0.99 / m) if m > 0.99 else out


HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ 박 확인</title>
<style>
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a66; --accent:#b4471f; --card:#ffffff; --line:#e4e1da; }
@media (prefers-color-scheme: dark) { :root { --bg:#161614; --fg:#ecebe6; --muted:#a3a19a; --accent:#f08a5d; --card:#1f1f1c; --line:#33322e; } }
body { background:var(--bg); color:var(--fg); font:16px/1.5 -apple-system, "Apple SD Gothic Neo", sans-serif; margin:0; padding:16px; }
main { max-width:760px; margin:0 auto; }
h1 { font-size:20px; margin:0 0 4px; }
p.note { color:var(--muted); margin:4px 0 12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin:10px 0; }
.big { font-size:44px; font-weight:700; font-variant-numeric:tabular-nums; }
.sub { color:var(--muted); font-variant-numeric:tabular-nums; }
.beat { display:inline-block; width:22px; height:22px; border-radius:50%; border:2px solid var(--line); margin-right:6px; }
.beat.on { background:var(--accent); border-color:var(--accent); }
audio { width:100%; }
button { font:inherit; padding:8px 12px; border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--fg); cursor:pointer; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
textarea { width:100%; min-height:120px; font:13px/1.4 ui-monospace, Menlo, monospace; background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:8px; box-sizing:border-box; }
kbd { border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:13px; }
</style></head><body><main>
<h1>__TITLE__: 박 클릭 확인</h1>
<p class="note">합주 녹음에 임시 정답 박을 클릭으로 입혔습니다. 높은 클릭은 마디 첫 박입니다.
클릭이 드럼 박과 어긋나거나, 화면의 마디 번호가 악보와 다르면 <kbd>M</kbd>을 누르세요 (그 시각이 기록됩니다).
<kbd>Space</kbd> 재생/정지, <kbd>←</kbd>/<kbd>→</kbd> 5초 이동.</p>
<audio id="a" controls preload="auto" src="__WAV__"></audio>
<div class="card"><div class="big" id="pos">-</div><div id="dots"></div><div class="sub" id="info">-</div></div>
<div class="card"><button class="primary" id="mark">어긋남 표시 (M)</button>
<button id="clear">표시 지우기</button>
<p class="note">표시한 곳 (복사해서 전달해 주세요):</p><textarea id="out" readonly></textarea></div>
</main>
<script>
const GT = __DATA__;
const a = document.getElementById("a"), pos = document.getElementById("pos"), info = document.getElementById("info"),
      dots = document.getElementById("dots"), out = document.getElementById("out");
const KEY = "ragsgo-check-__SONG__";
let marks = [];
try { marks = JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (e) { marks = []; }
const bpb = GT.beats_per_bar;
dots.innerHTML = Array.from({length: bpb}, (_, i) => `<span class="beat" id="d${i}"></span>`).join("");
function find(t) { let lo = 0, hi = GT.t.length - 1, k = -1; while (lo <= hi) { const m = (lo + hi) >> 1; if (GT.t[m] <= t) { k = m; lo = m + 1; } else hi = m - 1; } return k; }
function fmt(t) { const m = Math.floor(t / 60), s = (t - 60 * m).toFixed(1).padStart(4, "0"); return `${m}:${s}`; }
function render() {
  const t = a.currentTime, k = find(t);
  if (k < 0) { pos.textContent = "곡 시작 전"; info.textContent = fmt(t); }
  else if (t > GT.t[GT.t.length - 1] + 2) { pos.textContent = "곡 끝 뒤"; info.textContent = fmt(t); }
  else {
    pos.textContent = `${GT.bar[k]}마디 · ${GT.beat[k]}박 · ${GT.page[k]}쪽`;
    let nxt = "";
    for (const tb of GT.turns) { if (tb.t > t) { nxt = ` · 다음 넘김 ${tb.bar}마디 시작까지 ${(tb.t - t).toFixed(1)}초`; break; } }
    info.textContent = `${fmt(t)}${nxt}${GT.src[k] === "interp" ? " · 이 박은 보간값" : ""}`;
    for (let i = 0; i < bpb; i++) document.getElementById("d" + i).className = "beat" + (i === GT.beat[k] - 1 ? " on" : "");
  }
  requestAnimationFrame(render);
}
function show() {
  out.value = "시각(초),마디,박\\n" + marks.map(m => `${m.t.toFixed(2)},${m.bar},${m.beat}`).join("\\n");
  try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch (e) {}
}
function mark() { const t = a.currentTime, k = find(t); marks.push({t, bar: k >= 0 ? GT.bar[k] : 0, beat: k >= 0 ? GT.beat[k] : 0}); show(); }
document.getElementById("mark").onclick = mark;
document.getElementById("clear").onclick = () => { marks = []; show(); };
document.addEventListener("keydown", e => {
  if (e.key === "m" || e.key === "M") { mark(); e.preventDefault(); }
  else if (e.key === " ") { a.paused ? a.play() : a.pause(); e.preventDefault(); }
  else if (e.key === "ArrowLeft") { a.currentTime = Math.max(0, a.currentTime - 5); e.preventDefault(); }
  else if (e.key === "ArrowRight") { a.currentTime = a.currentTime + 5; e.preventDefault(); }
});
show(); render();
</script></body></html>
"""


def main(song="tac"):
    meta = SONGS[song]
    bpb = meta["beats_per_bar"]
    grid, info = original_grid(song, meta)
    m = np.load(RESULTS / f"offline_{song}_map.npz")
    mapped = to_live(grid, m)
    live = load_wav(f"{song}_live")
    odf, _ = B.onset_strength(live)
    live_bpm = B.estimate_tempo(odf, prior_bpm=meta["bpm"])
    live_dp = B.dp_beats(odf, bpm=live_bpm)
    beat_s = 60.0 / meta["bpm"]
    gt, j0 = count_based(mapped, live_dp)
    if gt is not None:
        src = np.array(["count"] * len(gt), dtype=object)
        n_inversions = 0
        info["gt_method"] = f"count (DP 박 {j0}번부터 차례로)"
    else:
        gt, src, n_inversions = refine(mapped, live_dp, beat_s)
        info["gt_method"] = "matched/interp"

    n = len(grid)
    beat_no = np.arange(1, n + 1)
    bar = (beat_no - 1) // bpb + 1
    bib = (beat_no - 1) % bpb + 1
    page = np.array([page_of_bar(b, meta["page_last_bars"]) for b in bar])
    ok = src != "interp"
    d = (mapped - gt)[ok]
    half = len(d) // 2
    info.update({
        "live_tempo_bpm_autocorr": round(live_bpm, 3), "live_dp_beats": len(live_dp),
        "live_dp_ibi_min_max": [round(float(np.diff(live_dp).min()), 3), round(float(np.diff(live_dp).max()), 3)],
        "not_interpolated_fraction": round(float(ok.mean()), 3), "n_interp": int((~ok).sum()),
        "mapped_minus_gt_ms": {"median": round(float(np.median(d) * 1000), 1),
                               "p90_abs": round(float(np.percentile(np.abs(d), 90) * 1000), 1),
                               "max_abs": round(float(np.abs(d).max() * 1000), 1),
                               "median_first_half": round(float(np.median(d[:half]) * 1000), 1),
                               "median_second_half": round(float(np.median(d[half:]) * 1000), 1)},
        "order_inversions": n_inversions,
        "live_song_first_beat_sec": round(float(gt[0]), 3), "live_song_last_beat_sec": round(float(gt[-1]), 3),
        "live_dp_beats_after_song": int(np.sum(live_dp > gt[-1] + 0.5 * beat_s)),
        "live_dp_beats_before_song": int(np.sum(live_dp < gt[0] - 0.5 * beat_s)),
    })
    with open(RESULTS / f"gt_{song}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["beat", "bar", "beat_in_bar", "page", "orig_sec", "live_mapped_sec", "live_sec", "source"])
        for i in range(n):
            w.writerow([beat_no[i], bar[i], bib[i], page[i], round(float(grid[i]), 4),
                        round(float(mapped[i]), 4), round(float(gt[i]), 4), src[i]])
    (RESULTS / f"gt_{song}_summary.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))

    CHECK.mkdir(exist_ok=True)
    wav = CHECK / f"{song}_clicks.wav"
    sf.write(str(wav), click_track(live, gt, bib == 1), F.SR, subtype="PCM_16")
    turns = [{"bar": int(b) + 1, "t": float(gt[(int(b)) * bpb])} for b in meta["page_last_bars"][:-1]]
    data = {"t": [round(float(x), 3) for x in gt], "bar": bar.tolist(), "beat": bib.tolist(),
            "page": page.tolist(), "src": src.tolist(), "beats_per_bar": bpb, "turns": turns}
    html = (HTML.replace("__TITLE__", meta["title"]).replace("__WAV__", wav.name)
            .replace("__SONG__", song).replace("__DATA__", json.dumps(data)))
    (CHECK / f"{song}_check.html").write_text(html)


if __name__ == "__main__":
    main(*sys.argv[1:])
