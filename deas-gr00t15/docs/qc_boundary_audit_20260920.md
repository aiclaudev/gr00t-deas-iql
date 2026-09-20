# QC / DEAS chunk 경계 검사 (2026-09-20)

## 검사 범위

- 학습 job 138941의 `20260919T160821Z-critic-lr3e4-featurefix-seed42` source snapshot 사용.
- 검사한 dataset/critic 파일 SHA-256이 manifest의 학습 소스와 일치.
- 4개 태스크 × demos/rollouts = 8개 데이터셋의 첫 에피소드 하나씩 검사. 전체 데이터 빈도 추정은 아님.
- reward/done 두 parquet 열만 읽고 snapshot에서 추출한 실제 loader와 critic reward/target 산식을 CPU에서 실행.
- 모델·영상 로딩, GPU 실행, 학습·평가 제출 없음.

재현: `scripts/robocasa/audit_chunk_boundaries.py --manifest <training-manifest> --output <json>`

상세 결과: `output/critic-boundary-audit/20260920/qc-boundary-audit.json`

## 확인된 동작

8개 표본 모두 next.done은 마지막 행에서만 1이다. 성공 데모 4개는 원본 reward가 마지막 행에만 1이며, rollout 표본 4개는 모두 reward 0이다.

DEAS loader는 성공이 있는 RoboCasa 에피소드의 마지막 15개 reward를 1로 변경한다. 따라서 원본 reward와 학습 reward는 다르다. 범위 밖 reward/done은 0으로 채운다.

critic은 prod(done)을 사용하므로 16개 중 마지막 done만 1인 완전한 chunk도 종료로 판정하지 않는다. bootstrap 계수는 0.99^16 = 0.8514577711로 남는다. 에피소드 끝을 넘는 chunk도 sampler가 선택할 수 있다.

negative_reward=True에서 padding reward도 0에서 -1로 바뀌어 누적 보상에 포함된다. action_mask는 action 차원 padding용이며 시간축 유효성 마스크가 아니다. critic loss에 chunk-valid 마스크는 없다.

### 성공 데모의 끝: 실제 loader와 산식 재현

V(next)=-50은 설명을 위한 가정값이며 체크포인트 예측을 측정한 값이 아니다. 성공 데모 표본 4개 모두 같은 결과다.

| 실제 남은 스텝 | DEAS 할인 reward | padding만의 기여 | bootstrap 계수 | V(next)=-50일 때 타깃 | QC valid[-1] |
|---|---:|---:|---:|---:|---:|
| 16 | -1.0000 | 0 | 0.851458 | -43.5729 | 1 |
| 8 | -2.4517 | -2.4517 | 0.851458 | -45.0245 | 0 |
| 1 | -7.1470 | -7.1470 | 0.851458 | -49.7199 | 0 |

성공 데모의 마지막 행을 실제 terminal로 해석하면 16-step chunk의 타깃은 bootstrap 없이 -1이어야 한다. 현재 코드는 -43.5729가 된다. 이는 feature 차원과 독립적인 target 처리 문제다.

## QC 원본과의 비교

- [데이터 샘플러](https://github.com/ColinQiyangLi/qc/blob/main/utils/datasets.py): 누적 episode boundary로 valid를 만들며 종료 transition 자체까지 유효하다.
- [critic/actor loss](https://github.com/ColinQiyangLi/qc/blob/main/agents/acfql.py): critic loss에 valid[..., -1], BC flow loss에 스텝별 valid 적용. 단일 discount의 chunk return과 다음 chunk Q로 타깃 구성.
- [실행 루프](https://github.com/ColinQiyangLi/qc/blob/main/main.py): terminals = terminated or truncated, masks = 1 - terminated로 경계와 bootstrap을 구분.

QC는 중간에 끝난 chunk의 critic loss 전체를 제외한다. 종료가 정확히 chunk 마지막 transition이면 샘플을 유지하고 실제 terminal에서 bootstrap을 끊는다. 앞서 제안했던 가변 길이 chunk 타깃은 별도 변형이며 QC 원본과 다르다.

QC 원본 기본 learner는 FQL 계열이며 IQL이 아니다. 요청한 구성은 **QC의 chunk 정의·마스킹을 적용한 scalar IQL Q/V**다. 타깃에 sampled next-chunk Q 대신 V를 사용하고 V는 expectile regression으로 학습한다. IQL의 V target도 학습되지 않은 invalid chunk Q를 사용하지 않도록 처리해야 한다.

## 새 구현에서 필요한 규칙

1. action chunk 16 유지, 단일 discount 사용.
2. chunk_valid와 bootstrap_mask 분리. padding은 실제 transition으로 취급하지 않음.
3. QC에 맞춰 중간에 경계를 넘는 chunk는 Q/V loss에서 제외. 마지막 transition에서 종료되는 완전한 chunk는 포함.
4. 실제 terminal과 timeout 구분. 표본에는 next.done만 있어 변환 원본에서 의미 확인 필요. 모든 마지막 행을 성공 terminal로 간주하지 않음.
5. 기존 성공 reward 마지막 15개 확장의 유지 여부를 명시. 자동 계승하거나 제거하지 않음.
6. 정상 chunk, 마지막 terminal, 중간 terminal, 첫 terminal, timeout, padding 검사.

이번 검사에서 앞의 다섯 경계/timeout 사례의 기대 마스크를 CPU로 확인했고, 실제 8개 에피소드의 32개 위치를 검사했다. 학습 코드 수정과 재학습은 하지 않았다. 기존 성공률 저하의 원인이 전부 이 문제인지는 아직 검증하지 않았다.
