# SVF 12조합: 학습 → 평가 → 다음 학습

## 준비 범위

배치 코드만 준비했다. 학습·평가·Slurm 제출은 실행하지 않았다.

설정: `configs/svf_sweep_seed42.json`.
기존 `configs/svf_joint_seed42_staged.json`을 읽고 **κ와 g만** 조합별로 바꾼다.

- κ: **0.4, 0.6, 0.8, 1.0**
- g: **0.25, 0.5, 0.8**
- κ 순서대로 각 g를 순회한다. 총 **12조합**, seed **42**.
- 각 조합은 동일한 BC2 actor / 동결 DEAS Q에서 새로 시작한다.
  이전 조합의 학습 가중치를 다음 조합에 넘기지 않는다.
- Global batch 128, 각 학습 **5,000 updates**에서 저장·중단.
- Actor cosine horizon **10,000**, soft value MSE + critic-trunk + 고정 LR.
- 학습 자원은 기존 **4 GPU / CPU 32 / RAM 768 GiB / sub-own** 템플릿을 유지한다.
- W&B 설정도 기존 학습 설정을 유지한다. 평가 run name에 κ/g 조합을 포함한다.

## 순서와 GPU 배치

```text
01 k=0.4 g=0.25 학습 (4 GPU)
  → checkpoint-5000/actor로 4개 task 평가 (각 1 GPU, 동시에 실행 가능)
  → 결과 집계 (CPU만)
  → 02 k=0.4 g=0.5 학습
  → 4개 task 평가 → 집계
  → …
  → 12 k=1.0 g=0.8 학습 → 4개 task 평가 → 집계
```

학습 12 + 평가 48 + CPU 집계 12 = **72개 개별 sbatch**다.
여러 조합의 학습을 동시에 실행하지 않는다. 각 단계가 끝나면 해당 자원을 반환한다.
모든 잡은 나중에 한 번의 제출 명령으로 예약되고 Slurm 의존성이 순서를 제어한다.
워커에서 추가 제출하거나 로그인 데몬으로 다음 잡을 감시하지 않는다.

- 평가: 해당 학습의 `afterok`.
- 집계: 해당 4개 평가의 `afterany`. 실패·누락도 집계 결과에 기록하기 위해서다.
- 다음 학습: 앞 조합 집계의 `afterok`.
- 집계는 4개 평가 결과가 모두 유효하고 완료됐을 때만 성공한다.
  평가 실패·누락 시 후속 학습은 시작되지 않는다. 성공률 0% 자체는 정상 평가 결과다.
- `--kill-on-invalid-dep=yes`로 충족 불가능한 후속 의존성은 취소되도록 한다.

## 평가 조건

4개 task: CoffeeSetupMug, PnPMicrowaveToCounter, TurnOffStove, PnPCounterToMicrowave.
학습한 **actor의 action을 RoboCasa에 직접 입력**한다. 추가 critic BoN 선택은 하지 않는다.
SVF의 `train/actor` 최종 링크는 10,000 step에서 생성되므로, 여기서는 반드시
`train/checkpoint-5000/actor`를 사용한다. 별도 래퍼가 부모 `complete.json`의 step을 검증한다.

평가 seed=42, n_envs=1, action_horizon=16, denoising_steps=4, 영상 저장=false를
기존 평가 방식에 맞춰 준비했다. 설정 파일에서 변경할 수 있다.
**Task당 episode 수, 학습 walltime, 평가 walltime은 null(미정)**이다.
사용자가 먼저 eval을 확인한 뒤 지정한다. 세 값이 정해지기 전에는 제출할 수 없다.

## 미리보기 — 제출 없음

```bash
cd /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
/home/nas_main/dohyunlee/miniconda3/envs/groot-train/bin/python scripts/submit_svf_sweep.py
```

`--json`을 추가하면 모든 명령·의존성·출력 경로를 볼 수 있다.
기본 실행은 디렉터리 생성과 클러스터 호출도 하지 않는다.

나중에 제출할 때만 `--submit`과 다음 값을 전달한다.

- `--train-time HH:MM:SS`: 측정한 학습·저장 시간에 여유를 더한 값.
- `--eval-time HH:MM:SS`: task 평가 시간에 여유를 더한 값.
- `--eval-episodes N`: task당 episode 수.
- `--run-root /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/svf-joint/sweeps/새이름`: 선택 사항.

제출 시 완료된 Q teacher, 학습 입력 설정, `snode --json` 자원을 확인한다.
기존 run-root 재사용은 거부한다. 일부 제출만 성공하면 job ID와 응답을
`manifest.json`에 남기고 멈추며 자동 재시도·자동 취소는 하지 않는다.

## 결과 경로

각 조합 디렉터리 아래:

- `training_config.json`: κ/g가 반영된 학습 설정.
- `train/checkpoint-5000/`: 10,000 step으로 재개할 상태와 actor export.
- `eval/manifest.json`: task별 actor·seed·job ID·결과 경로.
- `eval/results/seed-42/<TASK>/result.json`: task별 평가 결과.
- `eval/aggregate/summary.json`, `summary.md`, `by_task.csv`: 조합별 집계.

## 검증

CPU 테스트에서 정확한 12조합, 72개 잡의 의존성, 미래 체크포인트 경로,
잘못된 완료 표시 거부, dry-run의 무부작용, mock 제출의 ID 저장과 실패 중단을 확인했다.
실제 SVF 학습·RoboCasa 평가·Slurm 실행 검증은 이번 준비 범위에 포함하지 않았다.
