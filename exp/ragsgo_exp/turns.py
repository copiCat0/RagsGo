"""넘김 예측 절차 시뮬레이션 (기획 4.7절 '넘김 예측 절차').

박 이벤트(박 시각, 그 박을 알게 된 시각, 박 간격, 원곡 박 번호 추정, 확신 여유)를 알게 된 시각 순서로 받아
넘김 하나를 처리한다.
1. 넘김 2마디 전: 최근 4박 동안 박 번호가 1씩 올랐고, 박이 세게 들렸고(박 세기 평균 >= min_strength),
   확신 여유가 기준 이상이면 넘김 시각을 예약한다. 박 세기를 보는 이유: 드럼이 빠지면 박 추적기가
   엉뚱한 박을 꾸준히 세도 번호는 1씩 올라 확인을 통과하기 때문이다 (Phase 1-b 드럼 없는 구간 시험).
   안정되지 않았으면 1마디 전까지 기다리고, 그때도 안 되면 "직접" 표시(자동 넘김 포기).
   안정되지 않은 동안 나온 번호(시작 직후·재탐색 중 잠깐 튄 값)는 넘김 판단에 쓰지 않는다.
2. 그 뒤 매 박: 예약 시각 = 박 시각 + (넘김 박 번호 - 박 번호) x 박 간격 - 출력 지연.
3. 넘김 1마디 전: 같은 확인을 한 번 더 한다. 어긋나면 예약을 취소하고 "직접".
4. 예약 시각에 넘긴다. 다음 박 보고가 예약 시각보다 늦게 오면 그 전 예측대로 넘긴다.
   박 번호가 넘김 박을 이미 지났다고 나오면 바로 넘긴다 (늦은 넘김).
연주 중단(무음)으로 넘김을 막는 부분은 아직 없다.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BeatEvent:
    time: float      # 박이 울린 시각 (추정, 초)
    report: float    # 박을 알게 된 시각 (초)
    period: float    # 박 간격 추정 (초)
    number: int      # 원곡 박 번호 추정 (1부터)
    margin: float    # 확신 여유 (클수록 확실)
    strength: float = 1.0  # 박 세기 0~1 (박이 실제로 세게 들렸나)


@dataclass
class TurnResult:
    status: str                    # "auto" | "manual" | "missed"
    fire_time: Optional[float]
    reason: str = ""
    predictions: List[float] = field(default_factory=list)


def _ok(events, k, consistency_beats, min_margin, min_strength=0.0):
    if k < consistency_beats:
        return False
    recent = [events[j] for j in range(k - consistency_beats, k + 1)]
    nums = [e.number for e in recent]
    steady = all(b - a == 1 for a, b in zip(nums, nums[1:]))
    past = recent[:-1]  # 박 세기는 박 뒤 약 35 ms에 정해지므로 지금 박은 빼고 지난 박들만 본다
    strong = sum(e.strength for e in past) / len(past) >= min_strength
    return steady and strong and events[k].margin >= min_margin


def simulate_turn(events, turn_beat: int, beats_per_bar: int, check_bars: int = 2, recheck_bars: int = 1,
                  min_margin: float = 0.0, output_latency: float = 0.0, consistency_beats: int = 4,
                  min_strength: float = 0.0) -> TurnResult:
    ev = sorted(events, key=lambda e: e.report)
    checked = rechecked = False
    scheduled = None
    preds = []
    for k, e in enumerate(ev):
        if scheduled is not None and scheduled <= e.report:
            return TurnResult("auto", scheduled, "fired at scheduled time", preds)
        remaining = turn_beat - e.number
        if not checked:
            # 위치가 안정되지 않은 동안(최근 박 번호가 1씩 오르지 않음) 나온 번호로는 판단하지 않는다.
            # 시작 직후나 재탐색 중에 잠깐 튄 번호가 넘김 지점을 지나쳐도 무시한다.
            if not (0 < remaining <= check_bars * beats_per_bar):
                continue
            if _ok(ev, k, consistency_beats, min_margin, min_strength):
                checked = True
            elif remaining <= recheck_bars * beats_per_bar:
                return TurnResult("manual", None, "not stable by the 1-bar point", preds)
            else:
                continue
        if checked and not rechecked and remaining <= recheck_bars * beats_per_bar:
            rechecked = True
            if not _ok(ev, k, consistency_beats, min_margin, min_strength):
                return TurnResult("manual", None, "1-bar recheck failed", preds)
        if checked:
            if remaining <= 0:
                return TurnResult("auto", e.report, "count passed the turn beat: fire now (late)", preds)
            scheduled = e.time + remaining * e.period - output_latency
            preds.append(scheduled)
    if scheduled is not None:
        return TurnResult("auto", scheduled, "fired after last event", preds)
    return TurnResult("missed", None, "never reached the 2-bar check", preds)
