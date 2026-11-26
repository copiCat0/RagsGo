"""Phase 1-b 견고성 시험 (띵바츄, 임시 정답 기준). 합주 녹음을 변형해 같은 처리를 돌린다.

변형은 흉내일 뿐이다. 실제 휴대폰을 보면대 자리에 두고 녹음한 소리(R8)를 대신하지 못한다.
- phone: 휴대폰 마이크 대역 (150 Hz ~ 7 kHz)
- phone+agc: 대역 + 자동 음량 조절 흉내
- noise: 방 잡음 (SNR 10 dB, 0 dB)
- quiet: 입력이 20 dB 작음
- tempo: 밴드가 원곡보다 5% 빠르거나 느림 (위상 보코더)
- worst: phone + agc + SNR 10 dB + 5% 빠름
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

MIN_STRENGTH = 0.4  # 넘김 확인 때 최근 5박 박 세기 평균의 하한 (드럼 없는 구간 시험으로 정함, 같은 곡이라 과적합 위험)


def conditions(y):
    yield "clean", y, 1.0
    yield "phone", A.phone_mic(y), 1.0
    ph_agc = A.compress(A.phone_mic(y))
    yield "phone+agc", ph_agc, 1.0
    yield "noise_snr10", A.add_noise(y, 10), 1.0  # 잡음 조건은 곡 시작 감지 기준을 6 dB로 낮춘다 (아래)
    yield "noise_snr0", A.add_noise(y, 0), 1.0
    yield "quiet_-20dB", (y * 0.1).astype(np.float32), 1.0
    yield "tempo_x1.05", A.time_stretch(y, 1.05), 1.05
    yield "tempo_x0.95", A.time_stretch(y, 0.95), 0.95
    yield "worst", A.time_stretch(A.add_noise(ph_agc, 10), 1.05), 1.05


def main(song="tac"):
    meta = SONGS[song]
    gt = RB.load_gt(song)
    bpm = json.loads((RESULTS / f"gt_{song}_summary.json").read_text())["linear_bpm"]
    start = gt["live_sec"][0] - RB.PRE_ROLL
    y = load_wav(f"{song}_live")[int(start * F.SR):]
    ref_bs = RB.ref_beat_chroma(song, gt["orig_sec"])
    rows = []
    for name, yy, rate in conditions(y):
        rise = 6.0 if ("noise" in name or name == "worst") else 20.0  # 잡음이 크면 '연주 시작' 음량 상승이 작다
        out = run_pipeline(yy, ref_bs, bpm, p_stay=4.0, p_skip=4.0, p_jump=30.0, activity_rise_db=rise)
        g = dict(gt)
        g["live_sec"] = (gt["live_sec"] - start) / rate   # 정답 시각도 같이 늘이거나 줄인다
        beat_s = 60.0 / (bpm * rate)
        sel = (out["all_beats"] > g["live_sec"][0] - 0.35) & (out["all_beats"] < g["live_sec"][-1] + 0.35)
        f, _, _ = RB.f_measure(out["all_beats"][sel], g["live_sec"])
        ev, _, _ = RB.evaluate_numbers(out["beats"], out["number"], g["live_sec"], g["beat"], beat_s)
        events = [BeatEvent(float(out["beats"][k]), float(out["ready"][k]), float(out["period"][k]),
                            int(out["number"][k]), float(out["margin"][k]), float(out["strength"][k]))
                  for k in range(len(out["beats"]))]
        pages = RT.evaluate(g, bpm * rate, events, [b + 1 for b in meta["page_last_bars"][:-1]],
                            meta["beats_per_bar"], 0.0, MIN_STRENGTH)
        allb = RT.evaluate(g, bpm * rate, events, list(range(4, meta["bars"] + 1)), meta["beats_per_bar"], 0.0,
                           MIN_STRENGTH)
        allb0 = RT.evaluate(g, bpm * rate, events, list(range(4, meta["bars"] + 1)), meta["beats_per_bar"], 0.0, 0.0)
        row = {"condition": name, "activity_rise_db": rise, "tracker_f70": round(f, 3),
               "beat_number_correct": ev["beat_number_correct"], "min_strength": MIN_STRENGTH,
               "pages": RT.tally(pages), "all_bar_lines": RT.tally(allb), "all_bar_lines_no_strength_gate": RT.tally(allb0)}
        rows.append(row)
        print(name, "f70", row["tracker_f70"], "num", row["beat_number_correct"],
              "pages ok", row["pages"]["ok"], "/5 | bars ok", row["all_bar_lines"]["ok"], "manual",
              row["all_bar_lines"]["manual"], "early", row["all_bar_lines"]["early"], "late",
              row["all_bar_lines"]["late"], "| no gate: ok", row["all_bar_lines_no_strength_gate"]["ok"], "late",
              row["all_bar_lines_no_strength_gate"]["late"], flush=True)
    (RESULTS / f"robustness_{song}.json").write_text(json.dumps(
        {"note": "변형 흉내 시험, 임시 정답 기준. 실제 기기 녹음을 대신하지 못함", "rows": rows}, indent=2,
        ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
