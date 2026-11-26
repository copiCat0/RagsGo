"""Phase 1-b 박 단위 인과 정렬.

박 추적기가 박을 하나 낼 때마다, 그 박 동안의 크로마(박 동기 크로마)를 원곡의 박 동기 크로마와 비교해
원곡 박 번호를 갱신한다. 기본은 박 하나에 원곡 박 번호 1 증가다. 제자리(박을 하나 더 셈)와
2칸 전진(박을 하나 놓침)은 벌점을 준다. 박을 세는 일(기획 4.7절 2번)과 정렬로 감시하는 일(3번)을
한 동적계획법으로 합친 것이다. 미래 박은 쓰지 않는다.

재시작(R6): p_jump를 주면 어느 위치로든 뛸 수 있다(벌점 p_jump). 연주가 멈췄다 다시 시작한 직후에는
p_jump를 낮춰 쉽게 뛰게 한다 (박마다 다른 값을 배열로 줄 수 있다). JumpDTW(Fremerey 2010)와 같은 생각이다.
"""
import numpy as np

INF = 1e18


def beat_online_align(live: np.ndarray, ref: np.ndarray, p_stay: float = 1.0, p_skip: float = 1.0,
                      start_window: int = 8, valid=None, invalid_cost: float = 0.5, p_jump=None,
                      start_index: int = 0, jump_cost_at=None, reset_at=None):
    """live (12, K), ref (12, N): 열마다 L2 정규화된 박 동기 크로마.

    반환: est (K,) 원곡 박 번호(0부터), margin (K,) 확신 여유.
    margin = (최선 위치 ±1박 밖의 최소 누적 비용) - (최소 누적 비용). 클수록 다른 후보와 차이가 크다.
    jump_cost_at: {live 박 번호 k: (N,) 위치별 뛰기 벌점} 또는 함수 f(k, 그때까지의 추정) -> (N,) 또는 None.
    reset_at: {live 박 번호 k: (N,) 위치별 시작 비용}. 그 박에서 이전 경로를 버리고 위치를 새로 시작한다
    (사용자가 다시 칠 마디를 탭한 경우).
    값이 있는 박에서는 p_jump 대신 이 값을 쓴다
    (재시작 구간을 사용자가 탭으로 알려 주거나, 단락 시작에 가중치를 줄 때).
    """
    K, N = live.shape[1], ref.shape[1]
    if p_jump is not None:
        p_jump = np.broadcast_to(np.asarray(p_jump, dtype=np.float64), (K,))
    est = np.empty(K, dtype=np.int64)
    margin = np.empty(K)
    D = None
    for k in range(K):
        if valid is not None and not valid[k]:
            c = np.full(N, invalid_cost)
        else:
            c = 1.0 - ref.T @ live[:, k]
        if D is not None and reset_at is not None and k in reset_at:
            D = np.asarray(reset_at[k], dtype=np.float64) + c
        elif D is None:
            D = np.full(N, INF)
            a = min(max(start_index, 0), N - 1)
            b = min(a + start_window, N)
            D[a:b] = c[a:b]                                     # 시작 위치: start_index부터 start_window칸
        else:
            prev = D
            D = np.full(N, INF)
            D[1:] = prev[:-1]                                   # 1칸 전진 (기본)
            D = np.minimum(D, prev + p_stay)                    # 제자리
            D[2:] = np.minimum(D[2:], prev[:-2] + p_skip)       # 2칸 전진
            v = None
            if jump_cost_at is not None:
                v = jump_cost_at(k, est[:k]) if callable(jump_cost_at) else jump_cost_at.get(k)
            if v is not None:
                D = np.minimum(D, prev.min() + np.asarray(v, dtype=np.float64))
            elif p_jump is not None:
                D = np.minimum(D, prev.min() + p_jump[k])       # 어디로든 뛰기 (재시작)
            D = D + c
        j = int(np.argmin(D))
        est[k] = j
        out = D.copy()
        out[max(0, j - 1):j + 2] = INF
        margin[k] = float(out.min() - D[j]) if N > 3 else 0.0
    return est, margin
