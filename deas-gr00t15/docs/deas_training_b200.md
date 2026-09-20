# B200에서 DEAS 세 단계 순차 학습

## 현재 실행 범위 — 2026-09-19 변경

사용자 요청으로 seed 43·44의 6개 잡(136452–136457)을 시작 전 취소했다.
seed 42만 유지한다. 확인 당시 BC1(136449)·BC2(136450)는 완료했고,
critic(136451)은 실행 중이었다. 아래 3-seed 계획과 제출 표는 최초 제출 이력이다.
향후 평가는 `output/deas-training/20260918T172011.166909379Z/production/active-summary.json`을
사용한다. 기존 `summary.json`은 최초 제출 이력으로 보존한다.

## 학습 계획

| 순서 | 학습 | 시작 모델 | 입력 데이터 |
|---|---|---|---|
| 1 | Demo BC | 원본 GR00T N1.5 | robocasa_mg_gr00t_100 |
| 2 | 성공 rollout 추가 BC | 1단계의 최종 모델 | 네 작업의 demos + success_rollouts |
| 3 | DEAS critic | 원본 GR00T N1.5 | 네 작업의 demos + rollouts |

seed 42, 43, 44마다 아래 세 단계를 실행하여 본 학습은 총 9개 잡이다.
각 단계: B200 4장, GPU당 batch 32, 전체 batch 128, gradient accumulation 1,
10,000 optimizer steps. CPU 8개/GPU(총 32개), RAM 768 GiB, sub 계정 own QOS.
환경은 기존 NAS의 groot-train conda를 직접 활성화하며, 워커 이미지는
nvcr.io/nvidia/pytorch:25.04-py3를 사용한다.

세 단계를 별도의 sbatch로 한 번에 제출하고 afterok 의존성으로 연결한다.
성공한 단계 다음에만 후속 단계가 실행되며, 대기 중인 후속 단계는 GPU를 점유하지 않는다.
같은 seed는 afterok, 다음 seed의 첫 잡은 이전 seed critic의 afterany에 연결한다.
따라서 한 seed의 실패는 해당 seed 후속 단계를 취소하고, 독립적인 다음 seed는 계속한다.
각 단계 사이에는 클러스터 자원과 우선순위에 따라 대기 시간이 생길 수 있다.
Critic의 실행 순서는 BC 이후지만, 초기 가중치는 README와 동일하게 원본 모델이다.

## 본 학습 제출 상태

2026-09-18 18:33 UTC에 seed 42, 43, 44의 본 학습 9개 잡이 모두 접수되었다.
접수는 학습 완료를 의미하지 않으며, 현재 실행·대기 상태는 `squeue`와 `sjob`으로 확인한다.

| Seed | Demo BC | 성공 rollout 추가 BC | Critic |
|---|---|---|---|
| 42 | 136449 | 136450 | 136451 |
| 43 | 136452 | 136453 | 136454 |
| 44 | 136455 | 136456 | 136457 |

잡 ID와 seed별 출력 경로는
[제출 기록](../output/deas-training/20260918T172011.166909379Z/production/summary.json)에 저장되어 있다.
이미 제출한 실행이므로 아래 제출 명령을 다시 실행하지 않는다. 검증 후 제출 게이트는
기존 `production/` 디렉터리를 확인하여 재실행 시 중복 제출을 거부한다.
직접 sweep 명령에는 이 게이트가 없으므로 동일 학습을 중복 제출하지 않도록 주의한다.

## 시간 제한과 확인 범위

로그인에서의 짧은 검사는 정확성·메모리·저장 확인용이었다. 그 로그의 timing으로
워커 성능을 추정하지 않았다. 2026-09-18에 4-GPU 워커에서 세 단계 모두 50-step
검증을 완료했다(136242, 136243, 136244). 최종 체크포인트와 W&B finished 상태,
50 step, 유한한 loss, seed 42 / GPU 4 / GPU당 batch 32를 확인했다.
BC는 분할 safetensors, critic은 단일 model.safetensors로 저장되며 둘 다 검증한다.
실제 제출의 시간 제한은 워커 측정값과 체크포인트 저장 비용에 30% 여유를 더했다:
Demo BC 08:15:00, rollout BC 05:45:00, critic 06:10:00. 이는 완료 예상 시각이
아닌 각 잡의 종료 한도이며, 50-step 측정을 장시간 실행으로 외삽한 값이다.
submit_deas_training.sh는 임의 시간값을 기본으로 넣지 않고 --time 또는 단계별 시간 인수를 요구한다.
기본 10,000 step을 유지하고, 별도 짧은 워커 검증에는 --steps 옵션을 명시한다.

## 기록할 항목

- 공통: loss, learning rate, gradient norm, optimizer step, 전체 batch, 누적 학습 샘플 수.
- 성능: 실제 학습 시간, step당 시간, 초당 샘플 수. 측정 구간과 GPU 수를 함께 남긴다.
- Slurm: 제출·시작·종료 시각, 대기 시간, 워커 실행 시간, 잡 ID, 최종 상태.
- 자원: GPU 사용률·VRAM, CPU/RAM. W&B system stats와 sjob 지표를 사용한다.
- Critic: value/critic loss, Q1/Q2/V와 target의 평균·표준편차·범위, reward, expectile ratio.
- 재현: 설정, 코드 커밋과 수정 여부, 시작 체크포인트, 데이터 경로, 패키지 버전.
- FLOPs: 기존 BC/critic 디버그의 total_flos는 0이었다. HF 기본 입력 토큰 추정이 이 모델의 eagle_* 입력과 맞지 않으므로, 이 값을 실제 연산량이나 MFU로 해석하지 않는다.

W&B 목적지는 aiclaudev / gr00t1.5 finetune이다. 모델·데이터 파일의 업로드는
요청 범위에 포함하지 않는다. 사용자는 온라인 학습 지표·설정·경로 기록을 승인했다.
--seed 기본값은 42이며 모델 초기화, mixture sampling, Trainer seed/data_seed에 적용된다.
난수 seed는 고정하지만 deterministic kernel 강제는 하지 않는다.

## 자동 제출 명령

설정만 미리 확인(파일 생성 및 잡 제출 없음):

```bash
bash bash_scripts/submit_deas_seed_sweep.sh --seeds 42,43,44 --gpus 4 --steps 10000 --dry-run
```

새로운 독립 실행을 위한 제출 형식(위에 기록된 실행을 재제출하지 않음):

```bash
bash bash_scripts/submit_deas_seed_sweep.sh --seeds 42,43,44 --gpus 4 --steps 10000 \
  --time-bc-demo "$DEMO_TIME" --time-bc-rollout "$ROLLOUT_TIME" --time-critic "$CRITIC_TIME"
```

50-step 검증 뒤 CPU 후속 잡 136245가 자동 제출을 시도했지만 워커에서 sacct를
찾지 못해 제출 전에 실패했다. 이 잡으로 본 학습은 생성되지 않았다. 이후 로그인에서
산출물과 W&B 업로드를 검증하고, 성공한 검증 결과를 사용해 위의 본 학습 9개 잡을 제출했다.

제출에 사용한 명령(기록용, 재실행하지 않음): 로컬 작업은 상태 확인과 sbatch 제출이며,
학습은 워커에서만 한다. 추가 GPU 검증 없이 세 seed의 9개 학습 잡을 자동으로 연결했다.

```bash
/home/nas_main/dohyunlee/miniconda3/envs/groot-train/bin/python \
  scripts/submit_deas_after_preflight.py \
  --preflight-root /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/deas-training/20260918T172011.166909379Z \
  --seeds 42,43,44 --target-steps 10000
```

본 학습과 검증의 출력 폴더는 분리되며, 본 학습은 원본 모델에서 다시 시작한다.
검증 실패 시 본 학습을 제출하지 않는다. 제출 기록 production/ 디렉터리가 있으면
중복 제출을 거부하므로, 실패 후에는 기록과 squeue를 확인한다. 위의 직접 sweep 명령과
검증 후 제출 명령을 함께 실행하지 않는다.

W&B는 seed와 stage로 구분한다. 일반 loss는 Trainer 집계를 사용하지만,
기존 critic의 개별 Q/V 진단값은 rank 0 배치에서 나온 값이다.
추가 성능 지표는 각 단계의 performance.jsonl에도 저장한다.
GPU-hours는 워커 스크립트 시작 이후 할당 GPU 수 × 경과 시간이며,
큐 대기와 컨테이너 초기 시작 시간은 포함하지 않는다. Slurm 최종 경과 시간과 구분한다.
Unix 소켓의 경로 길이 제한 때문에 TMPDIR는 홈의 .tmp/deas-<jobid>로 짧게 두고,
나머지 모델 캐시는 단계 출력 폴더 아래에 둔다. 모든 생성 파일은 dohyunlee 홈 아래다.
