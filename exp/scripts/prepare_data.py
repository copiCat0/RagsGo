"""Task 2: 합주 녹음과 원곡 분리 트랙을 22,050 Hz 모노 WAV로 만든다.

만든 파일은 exp/data/ 에만 둔다 (기획 문서 4.9절: 원곡 소리는 밖으로 보내지 않는다).
"""
import json
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 22050
EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "data"
RESULTS = EXP / "results"
SRC = Path("/Users/iwonjun/Documents/03_개인/50_취미/합주/1030")
REC = Path("/Users/iwonjun/Documents/03_개인/60_사이드프로젝트/RagsGO/합주녹음")
TAU = SRC / "자처,띵바츄"

SONGS = {
    "tac": {  # Think About'chu
        "live": SRC / "ThinkAboutChu_02.mp4",
        "stems": {
            "bass": TAU / "TaU_bass.wav",
            "drums": TAU / "TaU_drums .wav",  # 파일 이름에 공백이 있다
            "guitar": TAU / "TaU_guitar.wav",
            "piano": TAU / "TaU_piano.wav",
            "vocals": TAU / "TaU_vocals.wav",
        },
    },
    "n103": {  # 넬 "1:03" (U16)
        "live": REC / "103.m4a",
        "stems": {name: SRC / "악보" / f"{name}.wav" for name in ("bass", "drums", "other", "vocals")},
    },
}


def convert_live(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
         "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def load_mono(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g)
    return y.astype(np.float32)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for song, spec in SONGS.items():
        live_path = DATA / f"{song}_live.wav"
        convert_live(spec["live"], live_path)

        stems = {name: load_mono(p) for name, p in spec["stems"].items()}
        n = min(len(s) for s in stems.values())
        stems = {k: v[:n] for k, v in stems.items()}
        full = np.sum(list(stems.values()), axis=0)
        drums = stems["drums"]
        nodrums = full - drums
        peak = float(max(np.abs(x).max() for x in (full, nodrums, drums)))
        scale = 0.99 / peak if peak > 0.99 else 1.0  # 세 파일에 같은 배율
        for name, sig in (("orig_full", full), ("orig_nodrums", nodrums), ("orig_drums", drums)):
            sf.write(str(DATA / f"{song}_{name}.wav"), sig * scale, SR, subtype="PCM_16")

        info = sf.info(str(live_path))
        summary[song] = {
            "live_sec": round(info.frames / info.samplerate, 2),
            "orig_sec": round(n / SR, 2),
            "peak_before_scale": round(peak, 3),
            "scale": round(scale, 4),
        }
        print(song, summary[song])

    (RESULTS / "prepare_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
