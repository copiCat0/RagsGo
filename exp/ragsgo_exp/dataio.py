"""데이터 경로, 곡 정보, 특징 캐시."""
from pathlib import Path

import numpy as np
import soundfile as sf

from . import features as F

EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "data"
RESULTS = EXP / "results"

# 템포는 악보 표기 (띵바츄.pdf ♩=85, 103.pdf ♩=84). 둘 다 4/4.
# n103의 원곡은 Phase 1-b에서 틀린 것으로 확인했다: 악보 폴더 분리 트랙은 C장조 곡이고 1:03 합주는 A♭장조다.
# 그래서 n103은 원곡 기준 실험에서 뺀다 (기록용으로만 남긴다).
# page_last_bars: 페이지마다 마지막 마디 번호 (악보 PDF의 줄 첫 마디 번호로 읽음). 마지막 페이지는 넘김이 없다.
SONGS = {
    "tac": {"title": "Think About'chu", "bpm": 85, "beats_per_bar": 4, "bars": 98,
            "page_last_bars": [14, 34, 54, 74, 94, 98]},
    "n103": {"title": "1:03 (넬)", "bpm": 84, "beats_per_bar": 4, "invalid_original": True},
}


def load_wav(name: str) -> np.ndarray:
    y, sr = sf.read(str(DATA / f"{name}.wav"), dtype="float32")
    if sr != F.SR:
        raise ValueError(f"{name}: sample rate {sr} != {F.SR}")
    return y


def load_features(song: str, which: str, causal: bool, hpss: bool) -> "F.Features":
    tag = f"{song}_{which}_{'causal' if causal else 'centered'}_{'hpss' if hpss else 'raw'}"
    cache = DATA / f"feat_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        return F.Features(chroma=z["chroma"], silent=z["silent"], times=z["times"], energy_db=z["energy_db"])
    f = F.chroma_features(load_wav(f"{song}_{which}"), causal=causal, hpss=hpss)
    np.savez(cache, chroma=f.chroma, silent=f.silent, times=f.times, energy_db=f.energy_db)
    return f
