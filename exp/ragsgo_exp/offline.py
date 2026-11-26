"""오프라인 subsequence DTW (진단용). 원곡 전체(ref)를 합주 녹음(live) 안에서 찾는다.

걸음은 (1,1), (1,0), (0,1)이고 가중치는 같다 (librosa 기본 설정과 같은 방식).
한 행을 한 번에 계산한다. 같은 행 안의 (0,1) 걸음은 누적 최솟값으로 푼다:
D[j] = min(A[j], D[j-1] + C[j])  =>  D[j] = S[j] + min_{k<=j}(A[k] - S[k]),  S = cumsum(C)
"""
import numpy as np


def subseq_dtw(ref: np.ndarray, live: np.ndarray):
    """반환: 시간 순서 경로 wp (L, 2) = (ref_idx, live_idx), 경로 위 평균 코사인 거리."""
    C = 1.0 - ref.T.astype(np.float64) @ live.astype(np.float64)  # (N, M)
    N, M = C.shape
    D = np.empty((N, M))
    D[0] = C[0]  # 합주 녹음의 어디서든 시작할 수 있다
    for i in range(1, N):
        prev = D[i - 1]
        A = np.empty(M)
        A[0] = prev[0]
        A[1:] = np.minimum(prev[1:], prev[:-1])  # (1,0) 걸음과 (1,1) 걸음
        A += C[i]
        S = np.cumsum(C[i])
        D[i] = S + np.minimum.accumulate(A - S)  # (0,1) 걸음

    i, j = N - 1, int(np.argmin(D[N - 1]))
    path = [(i, j)]
    while i > 0:
        cands = [(D[i - 1, j], i - 1, j)]
        if j > 0:
            cands.append((D[i - 1, j - 1], i - 1, j - 1))
            cands.append((D[i, j - 1], i, j - 1))
        _, i, j = min(cands)
        path.append((i, j))
    wp = np.array(path[::-1])
    return wp, float(C[wp[:, 0], wp[:, 1]].mean())


def subseq_dtw_slope(ref: np.ndarray, live: np.ndarray):
    """기울기 제한 subsequence DTW. 걸음은 (1,1), (1,2), (2,1)이라 국소 템포 비율이 0.5~2배로 묶인다.

    원곡 프레임마다 가중치가 정확히 1이 되게 비용을 더한다. 그래서 합주 쪽 길이가 다른 경로끼리도
    누적 비용을 그대로 비교할 수 있고, 반환하는 비용은 원곡 프레임당 평균 코사인 거리다.
    """
    C = 1.0 - ref.T.astype(np.float64) @ live.astype(np.float64)  # (N, M)
    N, M = C.shape
    D = np.full((N, M), np.inf)
    B = np.zeros((N, M), dtype=np.int8)  # 0: (1,1), 1: (1,2), 2: (2,1)
    D[0] = C[0]
    cols = np.arange(M)
    for i in range(1, N):
        cand = np.full((3, M), np.inf)
        cand[0, 1:] = D[i - 1, :-1] + C[i, 1:]
        cand[1, 2:] = D[i - 1, :-2] + 0.5 * (C[i, 1:-1] + C[i, 2:])
        if i >= 2:
            cand[2, 1:] = D[i - 2, :-1] + C[i - 1, 1:] + C[i, 1:]
        B[i] = np.argmin(cand, axis=0)
        D[i] = cand[B[i], cols]

    i, j = N - 1, int(np.argmin(D[N - 1]))
    end_cost = D[i, j]
    path = [(i, j)]
    while i > 0:
        b = B[i, j]
        i, j = (i - 1, j - 1) if b == 0 else (i - 1, j - 2) if b == 1 else (i - 2, j - 1)
        path.append((i, j))
    return np.array(path[::-1]), float(end_cost / N)
