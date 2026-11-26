"""Phase 1-a 온라인 정렬기 (인과 전방 동적 계획법).

합주 프레임이 하나 들어올 때마다, 원곡 위치가 0, 1, 2칸 나아가는 경우를 비교해 누적 비용을 갱신한다.
그 프레임에서 누적 비용이 가장 작은 원곡 위치를 현재 위치로 낸다. 미래 프레임은 쓰지 않는다.
템포 비율은 0~2배까지 따라간다 (원곡 기준).
"""
import numpy as np

INF = 1e18


def online_align(live: np.ndarray, ref: np.ndarray, silent=None, p_stay: float = 0.05,
                 p_skip: float = 0.05, start_window: int = 50, silent_cost: float = 0.5) -> np.ndarray:
    """live (12, N), ref (12, M): 열마다 L2 정규화된 크로마. 반환: 길이 N의 원곡 위치 추정."""
    n_live, n_ref = live.shape[1], ref.shape[1]
    est = np.empty(n_live, dtype=np.int64)
    D = None
    for i in range(n_live):
        if silent is not None and silent[i]:
            c = np.full(n_ref, silent_cost)  # 소리가 없으면 어느 위치든 같은 비용
        else:
            c = 1.0 - ref.T @ live[:, i]
        if D is None:
            D = np.full(n_ref, INF)
            w = min(start_window, n_ref)
            D[:w] = c[:w]  # 곡 앞부분에서만 시작할 수 있다
        else:
            prev = D
            D = prev + p_stay                                  # 원곡 위치 그대로
            D[1:] = np.minimum(D[1:], prev[:-1])               # 1칸 전진
            D[2:] = np.minimum(D[2:], prev[:-2] + p_skip)      # 2칸 전진
            D = D + c
        est[i] = int(np.argmin(D))
    return est
