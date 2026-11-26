# RagsGO Phase 1-a 실험 계획 (첫 진단)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 보유 데이터 2곡(Think About'chu, 넬 "1:03")으로, 드럼 소리가 큰 합주 녹음을 원곡에 실시간 방식으로 맞출 수 있는지 처음으로 본다.

**Architecture:** 노트북에서 오프라인으로 돌리되, 합주 녹음은 프레임 순서대로 넣고 미래 프레임을 쓰지 않는다(인과 처리). 기준은 원곡 분리 트랙의 합(드럼 뺀 것)이다. 사람이 만든 정답이 아직 없으므로, 전체 구간을 보고 맞춘 오프라인 DTW 경로와 비교하는 진단만 한다. 오프라인 경로는 정답이 아니다.

**Tech Stack:** Python 3.12 (uv 가상환경), numpy, scipy, librosa, soundfile, matplotlib, pytest, ffmpeg(설치돼 있음)

**근거:** 기획 문서 `2026-11-18-ragsgo-feasibility-plan.md`의 4.7·4.8절, 6장 Phase 1, 14장 권장 1번. 사용자가 2026-11-18에 진행을 지시했다 ("맞을거여 진행해볼까").

**이번 범위 밖 (Phase 1-b 이후):** 사람이 확인한 정답 박·마디표, 실시간 박 추적, 드럼 패턴 특징, 넘김 예측 시뮬레이션, 3곡 이상으로 하는 판정.

**제약**
- 디스크 여유가 6.2 GB뿐이다 (2026-11-18 확인). torch 같은 큰 패키지는 이번에 설치하지 않는다.
- 프로젝트 폴더는 git 저장소가 아니다. 커밋 단계는 생략한다.
- 원곡에서 만든 파일은 `exp/data/`에만 두고 밖으로 보내지 않는다 (기획 문서 4.9절).
- 아래에서 `R`은 `/Users/iwonjun/Documents/03_개인/60_사이드프로젝트/RagsGO`이다. 명령은 모두 절대경로로 쓴다.

---

### Task 1: 실험 환경

**Files:** Create `R/exp/requirements.txt`, `R/exp/pytest.ini`

**Step 1:** `uv venv --python 3.12 R/exp/.venv`

**Step 2:** `uv pip install --python R/exp/.venv/bin/python -r R/exp/requirements.txt`

**Step 3:** `R/exp/.venv/bin/python -c "import librosa, soundfile, scipy, matplotlib; print(librosa.__version__)"` → 버전이 출력되면 통과

### Task 2: 데이터 준비

**Files:** Create `R/exp/scripts/prepare_data.py`

**만드는 것:** `R/exp/data/{tac,n103}_{live,orig_full,orig_nodrums,orig_drums}.wav` (22,050 Hz, 모노, 16비트)
- 합주 녹음(mp4, m4a)은 ffmpeg로 변환한다.
- 분리 트랙은 soundfile로 읽어 전체 합, 드럼 뺀 합, 드럼만을 만든 뒤 모노로 바꾸고 리샘플한다. 세 파일은 같은 배율로 음량을 맞춘다.

**확인:** 길이가 tac_live 307.6초, n103_live 257.6초, tac_orig 281.4초, n103_orig 244.0초 (±0.1초)

### Task 3: 특징 모듈 (테스트 먼저)

**Files:** Create `R/exp/ragsgo_exp/features.py`, Test `R/exp/tests/test_features.py`

**테스트**
1. 440 Hz 사인파의 크로마 최댓값이 A(인덱스 9)에 있다.
2. 인과 HPSS는 t 이후 프레임을 바꿔도 t까지의 출력이 같다.
3. 10 Hz로 묶은 크로마는 열마다 L2 노름이 1이다.

**구현:** STFT(창 4096, hop 512, center=False) → HPSS(시간축 중앙값 21프레임 ≈ 0.49초, 인과 모드는 과거만 씀 / 주파수축 중앙값 17 bin / 부드러운 마스크) → 크로마 12차원 → 프레임별 정규화 후 로그 압축 → 10 Hz 평균 → L2 정규화. 무음 프레임을 표시한다.

**실행:** `R/exp/.venv/bin/python -m pytest R/exp/tests/test_features.py -q` → 실패 확인 → 구현 → 통과 확인

### Task 4: 오프라인 진단 정렬 (같은 곡 확인, R11)

**Files:** Create `R/exp/scripts/run_offline.py`
- 원곡(드럼 뺀 것) 전체를 합주 녹음 안에서 찾는 subsequence DTW를 한다 (코사인 거리, 합주 쪽은 가운데 기준 HPSS).
- 출력: 합주 녹음에서 곡의 시작과 끝, 평균 템포 비율, 10초 구간별 기울기, 기울기가 튀는 구간, 경로 평균 비용
- 같은 곡 확인: 1:03 합주를 1:03 원곡(맞는 짝)과 띵바츄 원곡(틀린 짝)에 각각 맞춰 비용을 비교한다. 띵바츄 합주도 반대로 한다.
- 출력 파일: `R/exp/results/offline_{tac,n103}.csv`, `.png`, `R/exp/results/offline_summary.json`

### Task 5: 온라인 정렬기 (테스트 먼저)

**Files:** Create `R/exp/ragsgo_exp/online.py`, Test `R/exp/tests/test_online.py`

**알고리즘:** 전방 동적 계획법이다. 합주 프레임 하나마다 원곡 위치가 0, 1, 2칸 나아가는 경우 중 누적 비용이 가장 작은 것을 고른다 (템포 비율 0~2배, 머무름과 2칸 이동에 작은 벌점). 그 프레임에서 누적 비용이 가장 작은 원곡 위치를 현재 위치로 낸다. 미래 프레임을 쓰지 않는다. 무음 프레임은 모든 위치에 같은 비용을 준다.

**테스트**
1. 원곡 시퀀스를 0.8배·1.25배 빠르기로 바꾼 입력을 넣으면 추정 위치가 실제 위치에서 2프레임(0.2초) 안이다.
2. i 이후 입력을 바꿔도 i까지의 추정이 같다.

### Task 6: 온라인 진단 실행

**Files:** Create `R/exp/scripts/run_online.py`
- 합주: 인과 HPSS 크로마. 원곡: 드럼 뺀 합의 크로마 (원곡 쪽은 오프라인 처리 허용, 기획 문서 4.8절)
- 시작: 오프라인에서 찾은 곡 시작 1초 전부터 넣는다 (시작 버튼을 누른 상황)
- 비교 대상: Task 4의 오프라인 경로 (정답 아님)
- 지표: 차이의 중앙값, 90%, 95%, 최댓값 (초, 박). 0.5박 안·0.5마디 안 비율. 일정한 지연(차이의 중앙값)을 뺀 뒤의 같은 지표
- 출력 파일: `R/exp/results/online_{tac,n103}.csv`, `.png`, `R/exp/results/online_summary.json`

### Task 7: 결과 정리

**Files:** Create `R/docs/results/2026-11-18-phase1a-results.md`. Modify 기획 문서의 상태 줄.
- 모든 수치에 "진단, 정답 아님"을 표시한다.
- 다음 단계(사람이 확인하는 정답 만들기)에 필요한 사용자 작업을 적는다.
