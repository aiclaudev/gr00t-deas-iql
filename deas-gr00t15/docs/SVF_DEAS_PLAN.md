# Frozen DEAS Q + joint SVF

## 현재 상태와 결정 — 2026-09-19

본 학습은 **코드와 실행 설정 준비까지**다. 별도 사용자 요청에 따라 LoRA는
로그인 GPU 1장·2 update correctness 검증을 완료했다(하단 결과). Slurm 제출,
critic 종료 감시나 자동 후속 제출은 하지 않았다. 본 학습 규모의 메모리·속도는 미검증이다.

- 순차 학습 비교를 제외하고, 첫 optimizer step부터 soft value와 policy를 함께 학습한다.
- seed 42 최종 IQL critic을 고정 Q teacher로 사용한다. 원래 IQL의 V(s)를 SVF에 사용하지 않는다.
- seed 42 BC2를 복제해 한쪽은 고정 reference flow, 한쪽은 개선할 actor로 둔다.
- Actor의 VLM backbone은 고정하고 action head(projectors + DiT)를 학습한다.
- κ·g sweep 없이 첫 조합을 κ=0.4, g=0.25로 준비했다. RoboCasa 최적값으로 검증한 값은 아니다.

설정: [configs/svf_joint_seed42.json](../configs/svf_joint_seed42.json)

| 항목 | 준비한 값 |
|---|---|
| GPU / CPU / RAM / QOS | 4 / 32 / 768 GiB / sub·own |
| Seed / optimizer steps | 42 / 10,000 |
| Global batch | 128 = GPU 4 × microbatch 4 × accumulation 8 |
| Actor / soft-value LR | 1e-5 / 3e-4, 일정한 LR |
| κ / g / c=κ²/g | 0.4 / 0.25 / 0.64 |
| MC 후보 K / SDE steps | 8 / 10 |
| t_min / guidance norm cap | 0.1 / 평균 BC target norm의 2배 |
| 데이터 | 기존 4 task의 demos + success_rollouts, BC2 정규화 통계 고정 |
| 저장 주기 | 1,000 optimizer steps 및 중단/완료 시점 |
| W&B | 온라인, aiclaudev / gr00t1.5 finetune, 지표와 설정만 |
| Walltime | 미지정. 워커의 짧은 검사에서 측정 후 20–30% 여유를 더한다. |

λ는 `c × max(mean(std_k Q), 1e-3)`로 추정한다. 각 microstep에서 4 rank의
Q spread를 합쳐 **16개 상태**로 같은 λ를 사용한다. Gradient accumulation을 포함한
128개 전체에서 λ를 추정하는 것은 아니다. Guidance norm cap은 rank별 microbatch 평균이다.

## 구조와 수식

기본 MSE soft value는 `V(s, x_t, t)`의 독립적인 두 scalar head다.
선택적 HL-Gauss CE 모드에서는 각 head가 분포를 출력하고 기대값을 V로 사용한다.
각 head는 `512 × 4 hidden layers + LayerNorm + GELU`이며, 입력은 고정된 critic의
64차원 특징, 64차원 state, 16×32 noisy action chunk, 64차원 Fourier time feature다.
실제 action 12차원 외 padding은 V 입력·guidance·actor loss에서 제거한다.

Teacher Q1/Q2는 기존 101-bin HL-Gauss 출력을 그대로 유지하고 expectation의 minimum을 쓴다.
새 V의 두 head는 각각 같은 Monte Carlo soft target을 학습한다.
기본 MSE는 scalar를 회귀하고, HL-Gauss CE 옵션은 아래 분포 학습 절차를 사용한다.
Actor guidance에는 두 V head의 평균을 사용한다.

```text
x_t = (1-t) noise + t data_action
lambda = (kappa^2/g) * max(mean_std(Q), 1e-3)
V_target = lambda * logmeanexp(Q(reference_SDE_endpoints) / lambda)
actor_target = data_action - noise + clipped[ kappa^2(1-t)/(t*lambda) * grad_x V ]
```

λ 추정과 V target은 서로 독립적으로 뽑은 noise/time/SDE 경로를 사용한다.
`t < 0.1`에서 actor guidance는 0이다. Q와 reference는 no_grad이고,
actor target의 guidance를 detach하므로 actor loss가 V를 업데이트하지 않는다.
Actor/V는 각각 별도 LR·gradient clipping을 사용한다.

FMRL 출처: `/home/nas_main/dohyunlee/jh_ws/fmrl/agents/qflow_svf.py`.
원본의 outer Q TD/EMA 업데이트를 제거한 **고정 DEAS Q 변형**이다.
원본의 일부 aggregate-head objective와 달리 여기서는 각 V head를 target에 회귀한다.

Time embedding은 `f=exp(linspace(0, log(256), 32))`,
`concat(sin(t*f), cos(t*f))`이다. 학습하지 않는 주파수이며 2π를 곱하지 않는다.
t는 episode time이 아닌 flow time(0=noise, 1=action)이다. Actor의 기존 time embedding은 유지한다.

## 저장된 critic과 연결할 때의 주의점

현재 IQL 코드에는 같은 backbone output을 value loss와 Q loss에서 수정하여,
현재 관측의 online Q에 feature 전처리가 **두 번** 적용되는 경로가 있다.
SVF adapter는 저장된 online Q의 학습 경로를 재현하도록 `teacher_feature_passes=2`를
명시한다. 진행 중인 IQL 코드는 변경하지 않았으며, 이 adapter가 teacher 학습의
불일치나 Q 품질 자체를 복구한다는 뜻은 아니다. 실제 실행 전 최종 Q 출력의 유한성,
후보 간 spread, soft target·guidance가 정상인지 확인해야 한다.

BC2와 critic의 state/action 통계는 다르므로 actor 좌표→환경 좌표→critic 좌표로 변환한다.
Rotation6d, binary action, 상수 범위와 padding을 별도로 처리한다.
두 모델의 frozen backbone은 각 체크포인트의 가중치를 유지한다.

## 준비 파일

- `gr00t/model/svf/`: time-conditioned soft value, SVF objective, GR00T/DEAS adapter와 joint model.
- `scripts/train_svf.py`: torchrun 학습, W&B, 저장·재개, HF actor export.
- `scripts/submit_svf.py`: 기본 dry-run. 명시적 `--submit --time`에서만 제출.
- `slurm/svf_joint.sbatch`: 기존 groot-train 환경 + NGC PyTorch worker.
- `tests/test_svf_*.py`: CPU 수식·변환·gradient 분리·작은 DDP·저장/재개 검사.

## 나중에 실행하는 절차

명령 확인만 하고 제출하지 않기:

```bash
cd /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T
python scripts/submit_svf.py
python scripts/submit_svf.py --preflight-steps 20
```

실제 제출은 critic job 136451이 COMPLETED이고 최종 checkpoint가 10,000 step인지
확인한다. 최종 critic 파일이 미완성이면 중간 checkpoint로 대체하지 않는다.
제출 직전에 `snode --json`과 현재 계정 잡을 확인하며, 다른 잡을 취소하지 않는다.

아래는 **실행 예시**다. `LIMIT`은 향후 점검 시 정하며 지금 실행하지 않는다.

```bash
# Critic 완료 후 짧은 4-GPU 워커 검사; LIMIT은 해당 검사에 줄 시간 한도.
python scripts/submit_svf.py --submit --time "$LIMIT" --preflight-steps 20
# 결과/finite loss/DDP/저장·재개 확인 후, 측정한 본 학습 walltime으로 이어서 실행.
python scripts/submit_svf.py --submit --time "$TRAIN_LIMIT" \
  --run-root "$RUN_ROOT" --resume "$RUN_ROOT/train/checkpoint-20"
```

Preflight도 `steps=10000`을 유지하므로 같은 설정으로 step 20부터 재개할 수 있다.
Preflight의 W&B는 꺼지고 본 학습부터 온라인으로 기록한다.
설정·배치·seed·teacher가 바뀌는 재개는 거부한다. 최종 actor는
`RUN_ROOT/train/actor`로 export되어 기존 RoboCasa actor eval 경로에 넣을 수 있다.

## 기록과 검증 범위

Loss(actor/V), Q spread, λ, 후보 가중치 집중도, guidance/base norm 비율,
clipping 비율, actor/V gradient norm, walltime, seconds/step, ETA,
최대 allocated/reserved VRAM, samples seen, trainable parameter 수를 기록한다.
FLOPs는 프로파일러로 측정하지 않았으므로 추정치를 실측처럼 기록하지 않는다.

CPU 테스트는 수식, padding, 정규화, frozen teacher 불변, 실제 joint wrapper의
작은 모형 backward, 2-rank Gloo + accumulation, 공유 λ,
checkpoint와 sample/RNG 재개를 검증한다. 이것만으로 실제 GR00T CUDA,
4-GPU NCCL, 최종 Q 품질, RoboCasa 성능이 검증된 것은 아니다.

### 준비 단계 검증 결과

2026-09-19: SVF CPU 테스트 **30개 통과**(2-rank Gloo 포함).
실제 설정의 경로·metadata·global batch 검증, Python/셸 문법,
제출 없는 dry-run, `git diff --check`를 통과했다.
이 준비 단계에서는 GPU 모델 로딩·NCCL·실제 학습을 실행하지 않았다.
이후 수행한 제한된 LoRA GPU 검증은 아래 결과를 참고한다.

## 선택적 DiT LoRA

기본 `--actor-tuning full`은 기존 action head 전체 학습을 유지한다.
`--actor-tuning dit-lora --lora-rank 16 --lora-alpha 16 --lora-dropout 0`으로
BC2의 DiT attention Q/K/V/output에만 LoRA를 붙일 수 있다.
실제 BC2에서 대상 Linear는 64개이며 rank 16의 trainable adapter는 3,276,800개다.

- 기준 BC2 reference를 먼저 복사한 후 student에만 adapter를 삽입한다.
- Actor 원본 가중치는 고정하고 LoRA A/B와 soft value만 joint로 학습한다.
- State/action projectors, VL self-attention, future/position tokens는 고정한다.
- Student action head의 원본 가중치와 adapter는 FP32, 계산은 BF16 autocast다.
  동결된 VLM과 reference/Q는 BF16이다.
- B를 0으로 초기화해 eval 모드에서 adapter 삽입 자체가 초기 BC 출력을 바꾸지 않도록 한다.
- Resume용 checkpoint에는 adapter 구조와 가중치를 보존한다. 일반 RoboCasa eval용 actor는
  LoRA를 합친 원래 GR00T 형식으로 export하며, 학습 중인 모델은 merge하지 않는다.
- Rank/alpha/dropout 또는 학습 모드를 바꾸는 resume는 거부한다.

준비한 별도 설정은 `configs/svf_joint_seed42_lora.json`이다. 예를 들어 다음 명령은
제출 없이 LoRA 학습 명령을 출력한다:

```bash
python scripts/submit_svf.py --config configs/svf_joint_seed42_lora.json
```

`smoke_svf_lora.py`는 로그인 correctness 전용이다. GPU 1장, batch 1,
optimizer update 2회, K=8, flow steps=10으로 finite loss/gradient 분리/adapter와 V 업데이트/
저장·복원을 확인한다. 속도 지표·처리량·온라인 W&B를 기록하지 않는다.
외부 590초 timeout과 내부 제한을 적용하며, 본 학습용 `train_svf.py`의 sbatch 요구는 유지한다.

### LoRA 로그인 검증 결과 (2026-09-19)

- CPU SVF 통합 테스트: **51 passed, 10 subtests passed**. 2-rank Gloo,
  LoRA 병합 export의 vanilla 모델 로딩, 재개 호환성 검사를 포함한다.
- 로그인 GPU 0 한 장, batch 1, worker 0, 정확히 2 optimizer update로 통과했다.
  BC2와 critic `checkpoint-4000`, CoffeeSetupMug demo 한 샘플을 사용했다.
  이 중간 critic은 correctness 확인용이며 본 학습 설정은 최종 critic 경로를 유지한다.
- DiT LoRA 3,276,800개와 soft value 2,307,074개 파라미터만 학습한다.
  LoRA A/B gradient, B 변화, soft value 변화, finite loss/gradient를 확인했다.
  고정 actor 원본·BC reference·DEAS Q에는 gradient나 가중치 변경이 없었다.
- Adapter/soft value/optimizer 저장·복원 후 실제 actor velocity와 soft value 예측의
  최대 절대 차이는 모두 0이었다.
- PyTorch 최대 GPU 메모리: allocated **13.823 GiB**, reserved **14.008 GiB**.
  이는 batch 1 correctness 테스트 값이며 본 학습 batch의 필요량이나 속도 측정값이 아니다.
- GPU 프로세스는 정상 종료됐다. 본 학습·Slurm 제출·온라인 W&B·속도 비교는 실행하지 않았다.
  4-GPU NCCL, 수렴 및 RoboCasa 정책 성능은 이번 검증 범위에 포함되지 않는다.
- 결과: `output/code-checks/svf-lora-login-20260919-a/report.json`
  / 로그: `.tmp/svf-lora-login-20260919-a.log`.

## 선택적 critic trunk 초기화

`--soft-value-init random`(기본) 또는 `--soft-value-init critic-trunk`을 선택한다.
제출용 JSON 설정에서는 `"soft_value_init": "critic-trunk"`으로 지정한다.
또는 `python scripts/submit_svf.py --config configs/svf_joint_seed42_lora.json --soft-value-init critic-trunk`으로
제출 없이 조합된 명령을 확인할 수 있다.
Actor의 `full`/`dit-lora` 선택과 독립적인 옵션이다. 원래 IQL V(s)가 아닌
고정 teacher의 **online Q1/Q2**에서 각각 soft value head 1/2를 초기화한다.

| 부분 | DEAS Q | Soft value / 초기화 |
|---|---|---|
| 입력 | feature 64 + state 64 + action 512 = 640 | time 64 추가, 총 704 |
| 첫 Linear | 640 → 512 | 기존 입력 열 복사·action 좌표 보정, time 열은 0 |
| Hidden | 512×4, LayerNorm, GELU | 각 Linear/LayerNorm의 가중치와 bias 복사 |
| 출력 | 512 → 101 logits, softmax의 bin 기대값 | 512 → 1 scalar를 새로 초기화 |

BC actor 좌표의 action과 critic 좌표의 action은 다르다. 이미 critic 좌표인
feature/state 부분은 그대로 복사하고, action 변환은 첫 층에 접는다:

```text
s = action_scale.repeat(16), d = action_bias.repeat(16)
W_new_action = W_Q_action * s
b_new = b_Q + W_Q_action @ d
W_new_time = 0
```

Padding 및 상수 연속 좌표는 대응 scale/bias가 0이다. Gripper/control 같은 binary
좌표는 미분 가능한 연속 확장으로 사용한다. Teacher의 hard threshold를 soft value에
넣으면 noisy action에 대한 gradient가 사라지므로 복사하지 않는다. Clean 0/1 action에서는
일치하지만 noisy 값에서는 teacher의 threshold 경로와 다른 초기 표현이다.

Fourier time feature 자체는 유지한다. 새 time 열이 0이므로 처음에는 t에 무관하며,
해당 열은 동결하지 않아 학습하면서 시간 의존성이 생긴다. 복사는 독립적인 tensor로
수행하고 teacher는 고정한다. 학습 모델은 BF16 teacher 값을 FP32 soft value로 복사하므로
원래 FP32 checkpoint와의 bitwise 동일성을 뜻하지 않는다.

이 옵션은 **critic의 hidden 표현 재사용**이다. 101-bin softmax 기대값을 scalar Linear로
정확히 접을 수 없으므로 초기 soft value 출력이나 action gradient가 Q와 같지는 않다.
Scalar 출력층을 새로 만들기 때문에 초기 value scale까지 Q와 같아지는 것도 아니다.
또한 teacher는 min(Q1,Q2), soft value guidance는 두 head 평균이며,
clean action의 Q 표현이 noisy x_t의 soft continuation value를 이미 표현한다는 보장은 없다.
MSE soft-target 학습과 기존 guidance clipping을 유지한다.

초기화 선택은 run/checkpoint identity에 기록한다. 재개 때 저장된 soft value가 초기값을
덮어쓰며, random/critic-trunk를 바꾸는 resume는 거부한다. 이전 checkpoint의 누락된
설정은 random으로 해석한다. 수렴이나 RoboCasa 성능 개선은 별도로 평가해야 한다.

### Critic 초기화의 CPU 검증 (2026-09-19)

실제 `checkpoint-4000`의 Q1/Q2 가중치만 CPU에서 읽어 초기화를 확인했다.
학습 모델과 같은 BF16 teacher → FP32 복사 조건에서, 연속 action 좌표 변환을
명시적으로 적용한 Q trunk와의 hidden 출력 최대 차이는 두 head 모두
`9.54e-7`이었다. 초기 time 불변성, time 열의 유한한 gradient, scalar readout 보존,
teacher 불변과 gradient 부재를 확인했다. 이 옵션은 GPU에서 추가 실행하지 않았다.
결과: `output/code-checks/svf-critic-trunk-cpu-20260919/report.json`.

현재 SVF CPU 회귀 검사: **63 passed, 17 subtests passed**. 초기화·LoRA·좌표 변환·
2-rank Gloo·저장/재개·제출 dry-run을 포함한다.

## Soft value loss 선택: MSE / HL-Gauss CE

`--soft-value-loss mse`(기본) 또는 `--soft-value-loss hl-gauss`을 선택한다.
Loss와 초기화는 독립적이며 기존 설정은 `mse + random`을 유지한다.

| Soft-value loss | random | critic-trunk | critic-full |
|---|---|---|---|
| mse | scalar head 전체 새로 생성 | hidden만 복사, scalar 출력 새로 생성 | 허용 안 함 |
| hl-gauss | 분포 head 전체 새로 생성 | hidden만 복사, 분포 출력 새로 생성 | Q1/Q2의 분포 출력층까지 복사 |

제출 없는 명령 확인:

```bash
python scripts/submit_svf.py --config configs/svf_joint_seed42_lora.json \
  --soft-value-loss hl-gauss --soft-value-init critic-full
```

### CE의 target과 policy gradient

기존 frozen Q의 scalar expectation으로 구한
`y = lambda * logmeanexp(Q_k / lambda)`를 그대로 사용한다.
이를 같은 DEAS HL-Gauss 설정의 Gaussian histogram으로 변환해
각 head의 logits와 soft-label CE를 계산한다. Q 후보들의 분포를 단순 평균하거나
가장 가까운 bin의 one-hot 정답으로 바꾸지 않는다. Actor의 flow MSE는 유지한다.

`V_h = sum(softmax(logits_h) * bin_centers)`이고, policy guidance는
`grad_x mean_h(V_h)`이다. Argmax나 bin index를 미분하지 않는다.
CE와 expectation 계산은 FP32로 수행한다. 원래 IQL critic의 loss 구현은 변경하지 않는다.

Bin 수, support 양 끝, Gaussian sigma는 선택한 teacher의 `hlg`에서 가져온다.
현재 critic 설정은 101 bins, support `[-100,0]`, sigma 약 `0.742574`이다.
양수 lambda에서 y는 후보 Q의 최솟값과 최댓값 사이에 있으므로 teacher의 support 안에 있다.
Support 밖의 잘못된 target이나 NaN을 정상 데이터처럼 조용히 처리하지 않는다.

`critic-full` 초기화도 시간 열은 0으로 시작하고 action 정규화 차이는 첫 층에 접는다.
이는 FP32로 계산한 복사 Q head 각각의 연속 action 확장을 재현하는 방식이다.
Noisy binary action의 hard threshold 경로, BF16 연산 반올림, teacher의 min(Q1,Q2)와
soft value의 head 평균까지 모두 동일하다는 뜻은 아니다.

### 어떤 의미에서 smooth한가

MSE는 scalar 예측에 대한 gradient가 `2*(V-y)`여서 큰 오차에 민감하다.
HL-Gauss CE의 logit gradient는 `p-target_prob`이며, Gaussian target이 이웃 bin 사이에서
연속적으로 이동한다. 두 방식 모두 연속적인 V를 만들 수 있다. CE라는 이유만으로
policy에 필요한 `grad_x V`가 더 매끄럽거나 정확하다고 보장되지는 않는다.
Softmax 집중, Gaussian smoothing, 유한 support 경계의 편향을 함께 확인한다.
또한 CE로 출력한 기대값은 Gaussian histogram projection의 기대값이므로 원 scalar y와
엄밀히 동일하지 않을 수 있다. 이산 bin을 쓰더라도 기대값은 연속값이다.

Loss raw 숫자는 CE와 MSE 사이에서 직접 비교하지 않는다. 두 방식 공통으로
raw target에 대한 value 예측 오차와 guidance norm/clip fraction을 기록하고,
CE에서는 예측 분포와 target projection의 영향을 추가로 확인한다.
출력 구조가 달라지므로 loss를 바꿔 중간 checkpoint를 resume할 수 없다.
Checkpoint에는 `soft_value_config.json`으로 loss/bins/support/sigma를 함께 저장하며,
이전 loss 정보가 없는 checkpoint는 MSE로 해석한다.

### MSE / CE 옵션 검증 (2026-09-19)

CPU SVF 통합 검사 **91 passed, 37 subtests passed**. MSE와 CE 각각의 실제 joint
wrapper forward/backward, CPU 2-rank Gloo gradient 누적·동기화, teacher 동결,
HL-Gauss 경계 target·입력 미분, 초기화 및 loss/support 변경 시 resume 거부를 확인했다.

실제 `checkpoint-4000` Q1/Q2를 CPU에서 읽어 `critic-full`을 확인했다.
BF16 가중치를 FP32로 계산하고 binary action을 clean 0/1로 맞춘 조건에서,
복사 전후 logits 최대 차이는 `9.54e-7`, 각 head 기대값 최대 차이는 `3.81e-6`이었다.
시간 열 gradient, CE joint backward, support 경계 확률의 유한성과 정규화도 통과했다.
CPU 확인은 수식·연결 검증이며 정책 성능이나 GPU 처리량 측정이 아니다.
CE 옵션의 GPU 실행이나 본 학습 제출은 하지 않았다.
결과: `output/code-checks/svf-hl-gauss-cpu-20260919/report.json`.
