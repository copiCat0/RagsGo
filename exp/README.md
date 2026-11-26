# RagsGO 실험 코드 (Phase 1)

모든 명령은 이 폴더(`exp/`)에서 venv 파이썬으로 실행한다.

```bash
PY=/Users/iwonjun/Documents/03_개인/60_사이드프로젝트/RagsGO/exp/.venv/bin/python
$PY -m pytest -q                       # 단위 테스트 (56개)
```

## 모듈 (`ragsgo_exp/`)

| 파일 | 내용 |
|---|---|
| features.py | STFT, 인과 HPSS, 크로마 (10 Hz 묶음 / 프레임 단위, 음량 기준 global·running) |
| online.py, offline.py | Phase 1-a 프레임 단위 온라인 정렬기, 오프라인 DTW |
| beats.py | 온셋, 템포, 오프라인 DP 박 추적, 인과 박 추적 HMM(박 세기 포함), 연주 시작·멈춤 감지 |
| beat_align.py | 박 단위 인과 정렬 (박 세기 + 크로마 감시, 재시작 뛰기, 사용자 탭 재시작) |
| pipeline.py | 소리 배열 → 박마다 원곡 박 번호 (위 단계를 모두 잇는 인과 처리) |
| turns.py | 넘김 예측 절차 시뮬레이션 (2마디 전 예약, 매 박 갱신, 1마디 전 재확인, "직접") |
| augment.py | 견고성 시험용 소리 변형 (빠르기, 휴대폰 대역, 압축기, 잡음, 드럼 제거) |
| dataio.py | 데이터 경로, 곡 정보(악보 페이지 끝 마디), 특징 캐시 |

## 스크립트 (`scripts/`, 실행 순서)

| 순서 | 스크립트 | 결과 (`results/`) |
|---|---|---|
| 1 | prepare_data.py | `data/*.wav` (원곡 파생 소리는 data/ 밖으로 보내지 않는다) |
| 2 | run_offline.py | offline_* (Phase 1-a) |
| 3 | run_online.py, explore_online.py | online_*, explore_online.json (Phase 1-a) |
| 4 | discover_pairs.py, verify_pairs.py [all] | discover_pairs.json, verify_pairs*.json (원곡-녹음 짝 찾기) |
| 5 | make_ground_truth.py tac | gt_tac.csv, gt_tac_summary.json, `check/tac_check.html` + `check/tac_clicks.wav` |
| 6 | run_beats.py tac [벌점 벌점 시작창] | beats_tac_*.json/csv |
| 7 | run_turns.py tac | turns_tac.json |
| 8 | run_restarts.py | restarts_tac.json |
| 9 | run_robustness.py tac, run_drumless.py tac | robustness_tac.json, drumless_tac.json |

결과 해석은 `docs/results/2026-11-18-phase1a-results.md`, `docs/results/2026-11-18-phase1b-results.md`.

## 새 곡 넣기 (원곡 파일을 받았을 때)

1. `scripts/prepare_data.py`의 SONGS에 합주 녹음과 원곡(분리 트랙 또는 파일)을 넣는다. 분리 트랙이 없으면 원곡 파일 하나를 쓰고, 드럼은 HPSS로 줄인다(코드 추가 필요).
2. `ragsgo_exp/dataio.py`의 SONGS에 템포, 박자, 마디 수, 페이지 끝 마디를 넣는다 (악보가 없으면 페이지 끝은 비워 두고 모든 마디선으로 잰다).
3. run_offline → make_ground_truth → (사람 확인) → run_beats / run_turns / run_restarts.
