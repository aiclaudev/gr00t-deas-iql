# 학습된 GR00T actor의 RoboCasa 평가 파이프라인

## 현재 준비 범위

현재 평가 경로는 BC2 actor 단독이다. CPU 모의 검증과 4개 task의 실제 짧은
actor-action/영상 검사를 마쳤다. 2026-09-19 제출했던 task당 10 episode 평가
잡(137033–137037)은 사용자가 짧은 미리보기로 범위를 바꾸어 취소했다.
최신 실행은 아래 «짧은 영상 미리보기» 절을 따른다. 기존 DEAS critic 재선택 경로는 이번 파이프라인의
검증 범위에 포함하지 않는다.

모델과 시뮬레이터를 같은 워커 프로세스에서 실행한다:

```text
RoboCasa 카메라 이미지 + 로봇 상태 + 작업 지시
  → Gr00tPolicy.get_action(observation)
  → action chunk (기본 16 simulator steps)
  → RoboCasa env.step(actions)
  → 다음 observation
  → episode 종료 시 성공 여부·실행 길이 기록
```

코어 반복문은 `gr00t/eval/rollout.py`의 `evaluate_vector_policy()`이며,
`scripts/eval_policy_robocasa.py`가 checkpoint 로딩, 환경 생성, 기록을 연결한다.
외부 inference 서버 없이 actor checkpoint를 직접 로드한다.

## 환경과 평가 기본값

기존 `.venv-robocasa`와 `scripts/robocasa/environment.sh`를 사용한다.
상세 설치·기존 환경 smoke 결과는 [RoboCasa on B200](robocasa_b200.md)를 참조한다.
이 환경은 `groot-train` 개인 conda 환경의 패키지를 읽는 overlay이다.

| 항목 | 기본값 |
|---|---|
| 모델 | 각 학습 seed의 BC2 최종 actor |
| Task | CoffeeSetupMug, PnPMicrowaveToCounter, TurnOffStove, PnPCounterToMicrowave |
| Episode | task/seed당 50; `--episodes`로 변경 |
| 벡터 환경 | 1; `--n-envs`로 변경 |
| 학습 seed / 평가 seed | 별도로 기록; 평가 seed 기본값은 학습 seed, `--eval-seed`로 고정 가능 |
| Action horizon / denoising | 16 / 4 |
| Action noise | 0 |
| Object split | B |
| Layout/style | (1,1), (2,2), (4,4), (6,9), (7,10) |
| Generative textures | 꺼짐 |
| 평가 잡 자원 | GPU 1, CPU 8, RAM 96 GiB, sub/own, NGC PyTorch 25.04 |
| 시간 제한 | 04:00:00; **아직 실제 policy 평가로 측정하지 않은 초기 한도** |
| 기록 | NAS CSV/JSON/JSONL + 선택적 W&B |

50 episode는 위 다섯 layout/style 후보에서 실행하는 task당 총합이다.
각 layout/style에 50 episode씩 할당한다는 의미는 아니다.
2026-09-19 사용자 요청으로 seed 43·44 학습을 취소했다. 현재 유효한 학습 기록은
`production/active-summary.json`의 seed 42 하나다.
1 seed × 4 tasks = 4개 GPU 평가 잡, 총 200 episode를 준비하며,
집계용 CPU 잡 1개를 뒤에 연결한다. 이는 전체 평가를 위한 예시 계획이며,
현재 실행 중인 짧은 미리보기와는 별도다.

## 현재 학습 기록으로 계획만 확인

레포 루트에서 다음 명령을 실행한다. 기본 동작은 dry-run이며 파일 생성,
클러스터 조회, 잡 제출 없이 JSON 계획만 출력한다.

```bash
python3 scripts/robocasa/submit_evaluations.py \
  --training-summary output/deas-training/20260918T172011.166909379Z/production/active-summary.json \
  --output-root output/robocasa-evaluation/seed42-bc2 \
  --episodes 50 --n-envs 1 --dry-run
```

이 계획은 각 BC2 actor 학습의 성공을 요구하고, 마지막 학습 critic 잡의 종료도
기다린다. 따라서 현재 4-GPU 학습과 평가가 own 할당을 두고 경쟁하지 않게 한다.
현재 학습이 실패한 경우 해당 체크포인트를 정상 결과로 간주하지 않는다.
실제 제출 시에는 완료된 학습의 의존성을 확인·정리하여, 이미 종료되어 큐에서
제거된 잡 ID를 새 dependency로 넘기지 않는다. 진행 중인 학습의 dependency는 유지한다.

**향후 평가를 실행하기로 결정했을 때만** 위 명령의 `--dry-run`을 `--submit`으로
바꾼다. `--submit`은 즉시 Slurm 제출을 수행한다. 명시적인 제출 모드에서는
`snode --json`과 기존 사용자 큐를 확인하고, 최종 manifest에 접수된 잡 ID를 기록한다.
제출 여부와 실제 잡 ID는 각 실행의 `manifest.json`에 기록된다.

`--output-root`는 아직 없는 경로여야 한다. 접수 도중 실패해도 이미 받은 ID를
`manifest.json`에 보존하고, 기존 디렉터리에 다시 제출하지 않는다. 결과·manifest를
지우고 재시도하지 말고 기록과 큐를 먼저 확인한다.

## 임의의 actor checkpoint 하나로 준비

```bash
python3 scripts/robocasa/submit_evaluations.py \
  --actor /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/ACTOR_DIRECTORY \
  --seed 42 --eval-seed 42 \
  --tasks CoffeeSetupMug \
  --episodes 1 --n-envs 1 --report-to none \
  --output-root output/robocasa-evaluation/actor-check --dry-run
```

`ACTOR_DIRECTORY`를 실제 checkpoint 디렉터리로 바꾼다. 이 모드는 config와
`experiment_cfg/metadata.json`이 존재해야 한다. 학습 summary 모드는 아직 생성되지 않은
미래 checkpoint 경로도 계획할 수 있다. 워커에서는 가중치 shard와 normalization
metadata를 다시 검사하며, 학습 steps 정보가 있으면 최종 step도 확인한다.

새 Slurm 실행 스크립트는 `slurm/robocasa_pipeline_eval.sbatch`이고,
로그인에서 직접 실행하면 거부한다. 모든 모델 평가·속도 측정은 워커에서 한다.
임시 파일·JIT/동적 모듈 캐시는 dohyunlee 홈 아래에 둔다.

## 저장 결과와 W&B

```text
OUTPUT_ROOT/
  manifest.json                 # 계획, 의존성, 접수된 job IDs, 부분 제출 오류
  capacity.json                 # 실제 제출 시점 자원 snapshot
  existing-jobs.txt
  logs/
  results/seed-42/CoffeeSetupMug/
    episodes.jsonl              # episode 완료마다 append
    eval.csv                    # episode/env index, success, length, elapsed_seconds
    result.json                 # running/completed/failed, 성공 수/분모, 설정·seed·checkpoint
    success.txt                 # 성공적으로 전체 평가가 끝난 경우의 최종 성공률
  aggregate/
    summary.json
    runs.csv
    by_task.csv
    summary.md
```

Episode의 `elapsed_seconds`는 rollout 시작부터의 누적 시간이다.
`result.json`의 `walltime_seconds`는 모델·환경 준비와 종료를 포함한다.
중간 실패 시 완료된 episode와 오류를 보존한다. 강제 종료는 마지막 running 상태를
남길 수 있으므로, 집계는 반드시 **completed + 요청한 전체 episode 수 + 설정 일치**를
확인한다. 누락·실패·부분 결과를 성공률 0으로 넣지 않는다.

W&B는 planner 기본 `--report-to wandb`이며 프로젝트는 `aiclaudev/gr00t1.5 finetune`이다.
평가 seed/task/run group과 episode 성공 여부, 길이, 누적 성공률, 소요 시간 및 설정을
기록한다. 모델·데이터·영상 파일 업로드는 하지 않는다. `--report-to none`으로 로컬 기록만
남길 수 있다. 직접 Python evaluator를 쓰면 기본은 `--report_to none`이다.

## 집계

제출 모드에서는 모든 평가 잡이 종료된 뒤 CPU 집계 잡을 자동 실행한다.
집계 워커 안에서 Slurm 명령을 다시 호출하지 않는다.

이미 존재하는 평가 manifest를 직접 집계하려면 다음을 사용한다:

```bash
python3 scripts/robocasa/aggregate_results.py \
  --manifest output/robocasa-evaluation/seed42-bc2/manifest.json
```

미완료 결과가 있으면 보고서는 쓰되 종료 코드는 1이다. 진행 중인 상태를 정리하려면
`--allow-incomplete`를 추가한다. Task별 pooled 성공률과 seed별 성공률 평균·표준편차를
기록하며, 일부 seed만 완료됐다면 완료 수/전체 수와 incomplete 상태를 명시한다.

## 검증 범위

CPU 모의 테스트로 episode/autoreset 집계, 부분 실패 보존, W&B 호출 형식,
제출 계획·의존성·중복 방지·부분 제출, 결과 집계의 누락/실패 처리와 설정 일치를 검사한다.
실제 BC2 checkpoint로 4개 task의 짧은 policy rollout과 MP4 인코딩·디코딩을
확인했다. 짧은 미리보기로 전체 episode 성공률을 계산하지 않는다.


## 모든 episode 영상 저장

계획기에 `--save-video`를 추가하면 평가한 모든 episode의 H.264 MP4를
`results/seed-SEED/TASK/videos/env_0/rl-video-episode-N.mp4`에 저장한다.
`videos.json`에는 전체 episode 번호·성공 여부·길이와 파일 경로를 대응시킨다.
영상은 NAS에 저장하며 W&B로 영상 파일을 업로드하지 않는다.
설치된 PyAV encoder를 사용하여 프레임을 스트리밍 저장한다.

```bash
python3 scripts/robocasa/submit_evaluations.py \
  --actor output/deas-training/20260918T183345.797990880Z/02-bc-rollout \
  --seed 42 --episodes 10 --n-envs 1 --save-video \
  --output-root output/robocasa-evaluation/seed42-bc2-10ep-video-20260919 --dry-run
```

기본 `--qos own`은 현재 학습이 own 4장을 사용하면 대기한다.
`--qos extra`는 같은 sub 계정의 추가 자원을 요청하며 선점될 수 있다.
실제 실행은 위 명령의 `--dry-run` 대신 `--submit`을 명시한다.


## 짧은 영상 미리보기 (2026-09-19 현재 요청)

사용자 요청으로 대기 중이던 Slurm preview array `137114`를 취소하고,
로그인 노드의 GPU 1장에서 한도를 정한 영상 디버깅을 완료했다.
4개 task 모두 추론 5회 / simulator 80 steps에서 멈췄다. 각 MP4를 CPU로
디코딩하여 15 FPS, 81프레임, 640×360, 5.4초를 확인했다. 프로세스는
정상 종료했고 로그인 GPU 메모리를 반환했다. 성공률 평가는 수행하지 않았다.

`scripts/robocasa/policy_smoke.py`는 모델을 한 번 불러와 4개 task를 차례로
처리한다. 각 task의 첫 episode를 초기화한 뒤 GR00T를 최대 5번 호출한다.
호출마다 16개의 action을 실행하므로 task당 최대 80 simulator steps,
전체 최대 20회 추론 / 320 simulator steps다. 자연 종료 시에는 더 일찍
멈추며, 호출 한도에 도달하면 episode의 성공·실패 판정을 기다리지 않고 끝낸다.
다음 episode를 시작하지 않는다.

- BC2 seed 42 최종 actor, 평가 seed 42, denoising 4.
- 로그인 GPU 1장, 전경 실행. 모델과 시뮬레이터의 연결·action·영상 저장 확인용이다.
- 전체 프로세스에 590초 timeout과 10초 종료 유예를 적용한다.
  로그인 디버깅의 계산 10분 / wall-clock 20분 한도를 넘겨 계속 실행하지 않는다.
- Task마다 15 FPS MP4 하나를 저장한다. FPS는 영상 재생 속도이며
  시뮬레이터 제어 주기는 바꾸지 않는다.
- 영상과 `report.json`은 NAS에 저장한다. 미리보기는 전체 episode 평가나
  속도 벤치마크가 아니며 W&B 성공률 집계에 넣지 않는다.
- `report.json`에 실제 inference 호출 수, simulator steps, 중단 이유와
  영상 경로를 기록한다. Task별 완료 여부는 이 보고서와 영상 파일로 확인한다.

현재 로그인 실행의 출력 경로:

```text
/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/robocasa-preview/seed42-bc2-5calls-15fps-login-20260919/
  report.json
  video-validation.json
  TASK/videos/env_0/rl-video-episode-0.mp4
```

취소한 array의 제출 기록은 기존
`output/robocasa-preview/seed42-bc2-5calls-15fps-20260919/manifest.json`에 남아 있다.

동일한 짧은 미리보기를 워커에서 실행하려면 `slurm/robocasa_preview.sbatch`를 사용한다.
더 긴 미리보기는 호출 수와 시간 제한을 수정한 뒤 워커로 제출하고,
전체 episode 평가는 앞서 설명한 `submit_evaluations.py`를 사용한다.
이 스크립트는 array `0-3%1`로 task마다 별도 잡을 만들고 동시에 GPU 1장만
사용한다. 기본 요청은 `sub/extra`, GPU 1 / CPU 8 / RAM 96 GiB, 15분이며
extra는 선점될 수 있다. 제출 전 여유 자원과 예상 시간을 다시 확인한다.
재실행 시에는 새 출력 디렉터리와 `logs/`를 준비하고 actor와 출력 루트를 넘긴다.
Python 옵션은 `--task`, `--inference-steps`, `--video-fps`, `--seed`로 조절한다.
