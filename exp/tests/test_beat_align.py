import numpy as np

from ragsgo_exp.beat_align import beat_online_align


def loop_song(n_beats, seed=0):
    """8박 화음 고리가 반복되고, 고리마다 조금씩 다른 성분(멜로디)이 섞인 박 동기 크로마."""
    rng = np.random.default_rng(seed)
    loop = rng.standard_normal((12, 8))
    x = np.tile(loop, (1, n_beats // 8 + 1))[:, :n_beats] + 0.6 * rng.standard_normal((12, n_beats))
    x -= x.mean(axis=0, keepdims=True)
    return x / np.linalg.norm(x, axis=0)


def perform(ref, path, noise, seed=1):
    rng = np.random.default_rng(seed)
    x = ref[:, path] + noise * rng.standard_normal((12, len(path)))
    x -= x.mean(axis=0, keepdims=True)
    return x / np.linalg.norm(x, axis=0)


def test_follows_one_beat_per_beat():
    ref = loop_song(200)
    path = np.arange(200)
    est, _ = beat_online_align(perform(ref, path, 0.5), ref)
    assert np.mean(est[8:] == path[8:]) > 0.97


def test_recovers_after_missed_and_extra_beat():
    ref = loop_song(200)
    # 60번째 박에서 박 하나를 놓치고(원곡 2칸 전진), 120번째 박에서 박을 하나 더 셈(원곡 제자리)
    path = np.concatenate([np.arange(0, 60), np.arange(61, 120), [119], np.arange(120, 200)])
    est, _ = beat_online_align(perform(ref, path, 0.5), ref)
    ok = est == path
    assert ok[8:59].mean() > 0.97
    assert ok[66:118].mean() > 0.95       # 놓친 박 뒤 6박 안에 회복
    assert ok[126:].mean() > 0.95         # 더 센 박 뒤 6박 안에 회복


def test_is_causal():
    ref = loop_song(120)
    path = np.arange(120)
    live = perform(ref, path, 0.5)
    a, _ = beat_online_align(live, ref)
    live2 = live.copy()
    live2[:, 80:] = perform(loop_song(120, seed=9), path, 0.5)[:, 80:]
    b, _ = beat_online_align(live2, ref)
    assert np.array_equal(a[:80], b[:80])


def test_margin_is_small_when_input_is_noise():
    ref = loop_song(120)
    rng = np.random.default_rng(5)
    noise = rng.standard_normal((12, 60))
    noise /= np.linalg.norm(noise, axis=0)
    _, margin_noise = beat_online_align(noise, ref)
    _, margin_song = beat_online_align(perform(ref, np.arange(60), 0.5), ref)
    assert np.median(margin_song[20:]) > 3 * np.median(margin_noise[20:])


def test_jump_back_without_pause_info_is_found_eventually():
    """멈춤 없이 위치가 바뀐 경우(드물다): 높은 벌점으로도 결국 찾되, 오래 걸려도 된다 (6마디 안)."""
    ref = loop_song(200)
    path = np.concatenate([np.arange(0, 100), np.arange(40, 140)])   # 100번째 박에서 40번 박으로 돌아가 다시 침
    live = perform(ref, path, 0.5)
    est, _ = beat_online_align(live, ref, p_stay=4.0, p_skip=4.0, p_jump=8.0)
    ok = est == path
    assert ok[8:100].mean() > 0.97
    assert ok[124:].mean() > 0.97
    est_nojump, _ = beat_online_align(live, ref, p_stay=4.0, p_skip=4.0)
    assert (est_nojump == path)[124:].mean() < 0.2


def test_reset_after_pause_finds_new_position_within_four_bars():
    """멈췄다 다시 시작한 첫 박에서만 벌점을 거의 없앤다. 기획 통과 기준: 재시작 뒤 4마디(16박) 안."""
    ref = loop_song(200)
    path = np.concatenate([np.arange(0, 100), np.arange(40, 140)])
    live = perform(ref, path, 0.5)
    pj = np.full(len(path), 30.0)          # 평소에는 거의 안 뛴다
    pj[100] = 0.5                          # 다시 시작한 첫 박
    est, _ = beat_online_align(live, ref, p_stay=4.0, p_skip=4.0, p_jump=pj)
    assert (est == path)[8:100].mean() > 0.97
    assert (est == path)[116:].mean() > 0.97


def test_no_false_jumps_on_normal_playing():
    ref = loop_song(300, seed=3)
    path = np.arange(300)
    est, _ = beat_online_align(perform(ref, path, 0.6, seed=4), ref, p_stay=4.0, p_skip=4.0, p_jump=8.0)
    assert np.mean(est[8:] == path[8:]) > 0.97


def test_start_index_sets_the_first_position():
    ref = loop_song(120)
    path = np.arange(5, 100)                      # 첫 박을 놓쳐서 원곡 5번 박부터 들어온 경우
    est, _ = beat_online_align(perform(ref, path, 0.5), ref, p_stay=4.0, p_skip=4.0, start_window=1, start_index=5)
    assert np.mean(est == path) > 0.98


def test_position_dependent_jump_cost_uses_a_hint():
    """반복이 거의 같은 곡이라도, 다시 칠 위치 근처만 싸게 뛰게 하면(사용자가 구간을 탭한 경우) 바로 찾는다."""
    rng = np.random.default_rng(11)
    loop = rng.standard_normal((12, 32))
    ref = np.tile(loop, (1, 8)) + 0.15 * rng.standard_normal((12, 256))   # 32박 고리 8번, 차이는 아주 작다
    ref -= ref.mean(axis=0, keepdims=True)
    ref /= np.linalg.norm(ref, axis=0)
    path = np.concatenate([np.arange(0, 150), np.arange(64, 160)])
    live = perform(ref, path, 0.4)
    hint = np.full(256, 1e9)
    hint[62:67] = 0.0                       # 64번 박 근처만 허용
    est, _ = beat_online_align(live, ref, p_stay=4.0, p_skip=4.0, p_jump=30.0, jump_cost_at={150: hint})
    assert (est == path)[152:].mean() > 0.97


def test_jump_cost_can_depend_on_where_we_were():
    """뛰기 벌점을 그때까지의 추정(멈춘 위치)으로 정할 수 있다 (인과적)."""
    ref = loop_song(200)
    path = np.concatenate([np.arange(0, 100), np.arange(40, 140)])
    live = perform(ref, path, 0.5)
    seen = {}

    def cost(k, est_so_far):
        if k != 100:
            return None
        seen["stop"] = int(est_so_far[-1])
        v = np.full(200, 1e9)
        v[max(0, seen["stop"] - 64):seen["stop"]] = 0.5     # 멈춘 곳보다 앞(16마디 안)으로만
        return v
    est, _ = beat_online_align(live, ref, p_stay=4.0, p_skip=4.0, p_jump=30.0, jump_cost_at=cost)
    assert seen["stop"] == 99
    assert (est == path)[106:].mean() > 0.97


def test_reset_restarts_from_a_given_position():
    """재시작 위치를 사용자가 알려 주면 그 박에서 위치를 새로 시작한다 (이전 경로는 버린다)."""
    rng = np.random.default_rng(12)
    loop = rng.standard_normal((12, 32))
    ref = np.tile(loop, (1, 8)) + 0.15 * rng.standard_normal((12, 256))
    ref -= ref.mean(axis=0, keepdims=True)
    ref /= np.linalg.norm(ref, axis=0)
    path = np.concatenate([np.arange(0, 150), np.arange(64, 160)])
    prior = np.full(256, 1e9)
    prior[64] = 0.0
    est, _ = beat_online_align(perform(ref, path, 0.4), ref, p_stay=4.0, p_skip=4.0, reset_at={150: prior})
    assert (est == path)[150:].mean() > 0.98
