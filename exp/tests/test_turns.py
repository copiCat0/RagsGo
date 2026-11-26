import numpy as np

from ragsgo_exp.turns import BeatEvent, simulate_turn

P = 0.7  # 박 간격 (초)


def regular_events(n_beats, start=1.0, lag=0.02, offset=0, margin=2.0):
    """박 번호 1..n_beats가 규칙적으로 오는 이벤트. offset만큼 박 번호를 잘못 센다."""
    return [BeatEvent(time=start + i * P, report=start + i * P + lag, period=P, number=i + 1 + offset,
                      margin=margin) for i in range(n_beats)]


def test_fires_at_turn_beat_when_everything_is_right():
    ev = regular_events(40)
    res = simulate_turn(ev, turn_beat=33, beats_per_bar=4)
    assert res.status == "auto"
    assert abs(res.fire_time - (1.0 + 32 * P)) < 1e-6


def test_output_latency_is_subtracted():
    ev = regular_events(40)
    res = simulate_turn(ev, turn_beat=33, beats_per_bar=4, output_latency=0.1)
    assert abs(res.fire_time - (1.0 + 32 * P - 0.1)) < 1e-6


def test_low_margin_two_bars_before_gives_manual():
    ev = regular_events(40, margin=0.1)
    res = simulate_turn(ev, turn_beat=33, beats_per_bar=4, min_margin=0.5)
    assert res.status == "manual"
    assert res.fire_time is None


def test_count_jump_before_recheck_cancels():
    ev = regular_events(40)
    # 1마디 전 확인 직전에 박 번호가 2 뛴다 (박을 잘못 셈)
    for e in ev[26:]:
        e.number += 2
    res = simulate_turn(ev, turn_beat=33, beats_per_bar=4)
    assert res.status == "manual"


def test_late_report_still_fires_from_last_prediction():
    ev = regular_events(40)
    # 넘김 1박 전 박(32번)의 보고가 넘김 시각 뒤에 온다 → 그 전 박의 예측으로 넘긴다
    ev[31].report = 1.0 + 32 * P + 0.3
    res = simulate_turn(ev, turn_beat=33, beats_per_bar=4)
    assert res.status == "auto"
    assert abs(res.fire_time - (1.0 + 32 * P)) < 1e-6


def test_wrong_count_fires_at_wrong_time():
    ev = regular_events(40, offset=1)   # 처음부터 박 번호를 하나 크게 셈 (확인으로는 못 잡는 경우)
    res = simulate_turn(ev, turn_beat=33, beats_per_bar=4)
    assert res.status == "auto"
    assert abs(res.fire_time - (1.0 + 31 * P)) < 1e-6  # 1박 일찍 넘긴다


def test_unstable_glitch_far_past_the_turn_is_ignored():
    ev = regular_events(40)
    for e, wrong in zip(ev[:3], (300, 120, 7)):   # 시작 직후 위치를 못 잡고 튀는 동안
        e.number = wrong
    res = simulate_turn(ev, turn_beat=33, beats_per_bar=4)
    assert res.status == "auto"
    assert abs(res.fire_time - (1.0 + 32 * P)) < 1e-6


def test_not_stable_by_one_bar_before_gives_manual():
    ev = regular_events(40)
    for i in range(22, 30):                        # 넘김 2마디 전부터 1마디 전까지 번호가 계속 흔들림
        ev[i].number += (i % 2) * 3
    res = simulate_turn(ev, turn_beat=33, beats_per_bar=4)
    assert res.status == "manual"


def test_weak_beats_before_the_turn_give_manual():
    ev = regular_events(40)
    for e in ev:
        e.strength = 0.9
    for e in ev[20:]:
        e.strength = 0.05                  # 드럼이 빠져 박이 거의 안 들린다
    res = simulate_turn(ev, turn_beat=33, beats_per_bar=4, min_strength=0.3)
    assert res.status == "manual"
