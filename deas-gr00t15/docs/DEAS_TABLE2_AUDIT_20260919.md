# DEAS Table 2 대비 학습·평가 점검

점검 시점: 2026-09-19 15:46 UTC / 2026-09-20 00:46 KST.
읽기 전용 소스·로그·저장된 추론 기록 점검. 이 점검에서 GPU 실행, 새 학습/평가 제출, 실행 코드 변경은 하지 않았다.

## 판단 요약

- **BC2가 전혀 학습되지 않았다는 증거는 없다.** 완료된 성공률은 22.0%이고, 논문 Filtered BC의 18.5%와 비슷한 범위다. BC1과 BC2의 training loss도 감소했다. 다만 BC1 평가와 held-out BC loss가 없어 BC1 자체의 성능과 BC2의 순수 개선량은 확정할 수 없다.
- **현재 Q가 actor를 개선한다는 증거는 없다.** 두 방법 모두 완료한 평가 seed 0·1에서는 BC2 87/400, BoN50 86/400이다.
- **평가 분산이 실제로 크지만, 그것만으로 설명하고 넘어갈 상황도 아니다.** Critic 학습에서 online Q와 target-Q의 feature 처리 횟수가 다르고, 실제 LR도 논문과 다르다. 이 두 사항은 코드에서 확인됐다. 성공률 저하의 원인이라는 인과관계까지 증명된 것은 아니다.
- 현재 Q를 고정해 SVF sweep으로 넘어가기 전에 feature 경로와 held-out Q ranking을 먼저 확인하는 것이 타당하다.

## 1. 논문과 성공률 비교

출처: [DEAS Table 2 및 Appendix A.2](https://arxiv.org/html/2510.07730v1).
논문은 task당 50 episode × **3 evaluation seeds**를 사용한다. 이 문구를 3개의 독립 training seeds로 해석하면 안 된다.

단위: %.

| Task | 논문 GR00T N1.5 | 논문 Filtered BC | 현재 BC2, 150ep | 논문 DEAS | 현재 BoN50, seed 0·1 100ep |
|---|---:|---:|---:|---:|---:|
| CoffeeSetupMug | 4.7 | 14.7 | 22.67 | 28.7 | 22.0 |
| PnPCounterToMicrowave | 21.3 | 25.3 | 22.00 | 36.0 | 25.0 |
| PnPMicrowaveToCounter | 7.3 | 14.7 | 28.67 | 18.0 | 24.0 |
| TurnOffStove | 14.7 | 19.3 | 14.67 | 18.0 | 15.0 |
| 평균 | 12.0 | 18.5 | **22.00** | **25.2** | **21.50** |

현재 BC2는 demos + successful rollouts로 학습했으므로 Filtered BC 열이 가장 가까운 비교 대상이다. 위 표의 현재 두 방법은 완료 seed 수가 다르다. **공정한 내부 비교는 아래처럼 같은 완료 seed 0·1로 제한한다.** 숫자가 같은 eval seed여도 정책별 RNG 소비가 달라 환경 초기화까지 엄밀히 paired인 실험이라고 보장하지 않는다.

| Task | BC2 성공/100 | BoN50 성공/100 | 차이, percentage points |
|---|---:|---:|---:|
| CoffeeSetupMug | 23 | 22 | -1 |
| PnPCounterToMicrowave | 18 | 25 | +7 |
| PnPMicrowaveToCounter | 30 | 24 | -6 |
| TurnOffStove | 16 | 15 | -1 |
| 합계 | 87/400 = 21.75% | 86/400 = 21.50% | **-0.25** |

전체 BC2 seed별 성공 횟수:

| Task | seed 0 | seed 1 | seed 2 | 합계 |
|---|---:|---:|---:|---:|
| CoffeeSetupMug | 10 | 13 | 11 | 34/150 |
| PnPCounterToMicrowave | 6 | 12 | 15 | 33/150 |
| PnPMicrowaveToCounter | 9 | 21 | 13 | 43/150 |
| TurnOffStove | 8 | 8 | 6 | 22/150 |

점검 시점 BoN50은 564/600 episode 진행 상태였다. seed 2는 Coffee 2/44, Microwave→Counter 15/49, Stove 9/50, Counter→Microwave 3/21이었다. 이 부분 결과를 서로 다른 task 가중치로 합쳐 최종 성능이라고 해석하지 않았다.

이전 BoN10 캠페인은 103/600 = 17.17%였다. 현재까지 N=50으로 늘려도 BC2 대비 이득은 확인되지 않는다. 평가 난수와 선택 정책의 차이 때문에 이것만으로 N=50이 N=10보다 우수하다고 확정할 수도 없다.

## 2. 평가 분산

- BC2 전체 성공률은 evaluation seed별 **16.5%, 27.0%, 22.5%**다. 표본 표준편차는 5.27 percentage points다.
- Microwave→Counter만 보면 **18%, 42%, 26%**, 표본 표준편차 12.22 points다.
- BC2 132/600의 Wilson 95% 구간은 약 **18.87–25.49%**다.
- 같은 seed 0·1의 BoN50−BC2 차이는 -0.25 points이고, 독립 이항 근사 95% 구간은 약 **[-5.96, +5.46] points**다.

위 구간은 training seed 42의 고정 체크포인트에 조건부인 참고치다. 환경/scene 간 상관, 실제 paired 초기화, 학습 seed 분산을 모델링하지 않았다. 논문 Table 2의 seed별 원자료도 없으므로 현재와 논문 간 유의성 검정이라고 해석하면 안 된다.

## 3. 실제 학습량과 loss

세 단계 모두 seed 42, global batch 128 = 4 GPU × 32, 10,000 optimizer steps를 완료했다.

| Training metric | 처음 1,000 step 로그 평균 | 마지막 1,000 step 로그 평균 |
|---|---:|---:|
| BC1 flow loss | 0.091816 | 0.036147 |
| BC2 flow loss | 0.039497 | 0.020732 |
| Critic CE | 1.35703 | 1.29714 |
| Value loss | 0.80212 | 0.59085 |

검사한 로그 수치는 finite다. Held-out validation은 없으므로 loss 감소는 학습 수행의 근거이지 성공률이나 Q ranking 검증을 대신하지 않는다.

### 논문 설정과 차이

| 항목 | 논문 Appendix A.2 | 실제 실행 |
|---|---|---|
| BC | batch 32 × 30K, AdamW, LR 1e-4 cosine | batch 128 × 10K, AdamW, LR 1e-4 cosine |
| Critic | batch 64 × 30K, Adam, LR 3e-4 | batch 128 × 10K, Adam, **LR 1e-4 constant** |
| Actor future tokens | 사용하지 않음 | 32개, 학습·평가 모두 사용 |
| BoN | N=10, task별 greedy / softmax temperature 1 중 좋은 결과 | N=50, 모든 task greedy |

논문 기재 batch를 global batch로 해석하면 BC의 sample 노출은 1.28M 대 논문 0.96M으로 더 많지만 optimizer update 수는 적다. 단순히 10K/30K를 보고 학습량 1/3이라고 할 수 없다. 같은 해석에서 Critic의 sample 노출은 1.28M 대 1.92M이며 update 수와 LR도 다르다. 공개 BC 스크립트 기본값도 2 GPU × 16 = global 32지만, 논문의 실제 제출 인자까지 확인한 것은 아니므로 sample 노출 비교는 이 해석에 조건부다.

**설정 함정:** `critic_lr`와 `value_lr`에 3e-4가 기록되어 있지만 실제 optimizer는 `config.learning_rate` 하나만 사용하며 실행값은 1e-4다. [실제 optimizer 생성](/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/scripts/gr00t_deas_critic_finetune.py:368).

데이터:

- BC1: MG100 전체 24개 task, 2,400 episodes / 687,211 frames.
- BC2: 4개 task demos 400 + successful rollouts 182 = 582 episodes / 173,882 frames.
- Critic: demos 400 + all rollouts 1,200 = 1,600 episodes / 736,682 frames.
- 데이터셋 sampling은 길이 가중 기본값이다. Critic의 rollout frame 비중은 약 84%다.
- Critic의 frozen backbone 초기값은 BC2가 아닌 원본 GR00T이며 현재 README 기본값과 일치한다.

논문 본문 RL actor 설명은 demos + rollout dataset이라고 표현하고, 공개 README는 successful rollouts를 BC2에 사용한다. 현재 실행은 README 경로를 따른다. 본문만으로 논문 actor가 반드시 현재와 동일한 필터링을 했다고 확정하지 않는다.

## 4. 확인된 critic feature 경로 문제

`gr00t/model/action_head/deas_critic.py`의 기존 소스에서:

1. `process_backbone_output()`은 `BatchFeature` 안의 `backbone_features`를 in-place 교체한다 (line 219).
2. `forward()`는 같은 객체로 value loss를 먼저, critic loss를 다음에 호출한다 (lines 418–419).
3. value 경로는 transform을 한 번 적용한다 (line 232). 이 feature를 V와 target-Q에 사용한다.
4. critic 경로는 같은 current feature에 transform을 다시 적용한다 (line 322). online Q는 두 번 변환된 feature로 학습한다.

변환을 F, 원본 backbone feature를 x라고 쓰면:

| 경로 | 입력 feature |
|---|---|
| V 및 V 학습용 target-Q | F(x) |
| online Q | F(F(x)) |
| 다음 상태 V bootstrap | F(x_next) |

target-Q는 online Q의 EMA인데 입력 표현이 다르다. 이는 학습 일관성 문제다. 해당 파일의 현재 git diff는 비어 있어 이번 평가 코드에서 새로 만든 변경은 아니다. 다만 F와 F²의 실제 차이 크기 및 학습 성능에 미친 영향은 아직 측정하지 않았다.

현재 평가 `gr00t/model/checkpoint_bon_policy.py:109`는 저장된 online Q 입력에 맞게 **2회** 처리한다. 따라서 현재 eval에만 1회 처리를 넣어서 해결할 문제가 아니다. 학습 경로 수정 후 기존 checkpoint를 아무 검증 없이 그대로 재사용하면 입력 의미가 바뀐다.

## 5. 평가 실행과 Q 값 점검

현재 actor/critic 체크포인트의 step 10K, task 순서, action 순서, critic 자체 정규화, padding, denoising 4, action chunk 16을 확인했다. Held-out object split B와 5개 layout/style 조합, task별 horizon은 원본 evaluator 설정과 일치한다. 원본 스크립트 n_envs=5와 달리 현재 1이므로 같은 숫자 seed라도 원본과 동일 rollout을 보장하지 않는다.

현재 evaluator는 simulator step 중 한 번이라도 success가 발생하면 기록하고, autoreset step을 episode 집계에서 제외한다. 명백하게 성공을 낮게 집계하는 오류는 찾지 못했다. 이는 모든 제어/시뮬레이터 동작을 완전히 검증했다는 뜻은 아니다.

완료된 BoN50 task×seed 9개에서 각 12개의 균등 간격 trace를 표본 추출했다. 총 108 calls, 5,400 Q scores를 CPU로 검사했다.

- Q는 전부 finite, 각 call의 50개 action 후보는 서로 달랐다.
- 출력 action은 Q argmax 후보와 **108/108 정확히 일치**했다.
- 항상 첫 후보를 고르는 문제는 없었고 최대 Q 동점은 1/108이었다.
- 후보 Q max−min의 task/seed별 중앙값은 약 0.027–0.161이었다. 후보 간 차이는 대체로 작지만 완전히 상수는 아니다.
- Q가 음수인 것 자체는 오류가 아니다. 이 구현은 reward shift와 두 할인율을 쓰며 Q는 성공확률이 아니다.

작은 Q 차이만으로 critic collapse를 확정할 수 없다. 필요한 것은 그 차이가 실제로 좋은 action을 높은 순서에 놓는지에 대한 검증이다.

## 6. 다음 검증 순서

1. 작은 고정 batch로 feature 경로를 수치 비교하고, online Q / EMA target-Q가 같은 입력을 보도록 학습 경로를 정리한다. 기존 체크포인트 호환성과 새 학습 코드는 분리해서 다룬다.
2. 별도 held-out rollout에서 Q와 실제 return/success의 관계, 성공·실패 데이터 구분, action 변경에 따른 ranking을 점검한다. Monte Carlo return과 Q는 같은 reward shift/할인 정의로 비교하고 expectile bootstrap 차이를 고려해야 한다. 기존 training rollouts를 쓰면 in-sample 검사라고 표시한다.
3. Critic LR 연결을 명시적으로 고친 설정으로 소규모 별도 학습을 검증한 뒤, 동일 평가 protocol의 BC2 대비 이득으로 판정한다. 현 CE 숫자만으로 재학습 종료나 SVF 준비 완료를 결정하지 않는다.
4. BC1 자체 성능이 필요하면 동일 protocol의 BC1 평가를 추가해 BC1→BC2 변화를 직접 측정한다.

위 항목은 후속 제안이며 이번 점검에서 실행하지 않았다. 진행 중 평가와 체크포인트는 변경하지 않았다.

## 근거 파일

프로젝트 루트: `/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T`.

- 실제 학습: `output/deas-training/20260918T183345.797990880Z/{01-bc-demo,02-bc-rollout,03-critic}/trainer_state.json` 및 `arguments.tsv`.
- 현재 평가: `output/robocasa-comparison/seed42-bc2-vs-bon50-eval012-50ep-20260919/manifest.json`.
- task 결과: 위 평가 루트의 `{bc2|bon}/results/eval-seed-{0,1,2}/{task}/result.json`, 진행 중 결과는 `episodes.jsonl`.
- Q 표본: 위 평가 루트의 `bon/results/eval-seed-*/{task}/inference/call-*.npz` 중 제한된 표본만 검사.
- 이전 N=10: `output/robocasa-bon/seed42-bon10-eval012-50ep-video-inputs-20260919`.
- 학습 구현: `scripts/gr00t_deas_critic_finetune.py`, `gr00t/model/action_head/deas_critic.py`.
- 평가 구현: `gr00t/model/checkpoint_bon_policy.py`, `scripts/eval_policy_robocasa.py`, `gr00t/eval/rollout.py` 및 현재 BoN의 `source-snapshot/repo`.
