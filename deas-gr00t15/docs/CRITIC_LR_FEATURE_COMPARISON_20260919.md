# Critic LR / feature 처리 비교 — 2026-09-19

사용자 요청으로 아래 두 개의 독립 학습을 제출했다. 기존 seed42 10K critic은 별도로 보존된다.

| Job | Feature 경로 | 실제 Adam LR | 초기화 | Steps |
|---|---|---|---|---|
| 138923 | 기존 Q 2회 / V·target-Q 1회 | 3e-4 constant | 원본 GR00T에서 새 학습 | 10,000 |
| 138941 | 수정 Q·V·target-Q 모두 1회 | 3e-4 constant | 원본 GR00T에서 새 학습 | 10,000 |

공통 조건: seed42, global batch128, accumulation1, 1 B200, CPU24, RAM192GiB, sub-own, Decord, loaderworkers16, 저장 주기1,000step, 시간 제한6h30m. 데이터는 동일한 4개 task의 demos+전체rollouts다. 학습 메트릭과 설정은 기존 허용된 W&B `aiclaudev/gr00t1.5 finetune`에 별도 run으로 기록한다. 모델 및 소스 업로드는 비활성화했다.

출력 루트:

- LR만 변경: `/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/deas-training/20260919T155726Z-critic-lr3e4-seed42`
- Feature도 수정: `/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/deas-training/20260919T160821Z-critic-lr3e4-featurefix-seed42`

각 루트의 `03-critic`에 checkpoint, `logs`에 Slurm stdout/stderr, `manifest.json`에 실제 제출 인자와 자원·검증 결과가 저장된다. 각 worker는 자기 `source-snapshot/repo`만 import하므로 이후 원본 저장소 변경이 실행 중 모델 코드에 영향을 주지 않는다.

## 수정과 호환성

- 새 critic은 `process_backbone_output`에서 원본 BatchFeature를 덮어쓰지 않고 새 mapping을 반환한다. value loss 계산 후 online Q loss가 같은 원본 feature를 받는다.
- 새로 GR00T에서 생성되는 critic의 `config.critic_cfg.online_q_feature_passes=1`을 저장한다.
- 표식이 없는 기존 checkpoint는 2로 취급하며 기존 학습·평가 경로를 보존한다.
- BoN evaluator와 SVF teacher가 checkpoint 표식에 따라 1회/2회 처리를 선택한다.
- feature-fixed worker는 학습 전 model config와 head config에 모두 표식1이 있는지 검사한다.

## 검증

CPU 테스트 총19개 통과:

- `tests/test_checkpoint_bon.py`:12개. 실제 critic loss 메서드로 V/target-Q/Q/next-V의 입력 변환 횟수와 원본 보존, 기존 경로 재현, checkpoint 표식 처리 확인.
- `tests/test_critic_config.py`, `tests/test_svf_model.py`:7개. 새 checkpoint 표식 생성·직렬화, 기존 checkpoint 기본값 및 SVF forward/gradient 경로 확인.

GPU correctness 검증은 제출된 각 본 학습의 시작 로그와 첫 step으로 확인한다. 상태 확인 명령:

```bash
sjob -w 138923
sjob -w 138941
```

이 비교는 기존 LR1e-4·4GPU 학습과 자원 구성이 다르다. 새 두 run끼리는 자원·배치·LR를 맞췄다. 개선 여부는 학습 loss만으로 확정하지 않으며, 완료 후 동일 평가 조건의 성공률과 Q ranking을 비교해야 한다. 추가 eval은 자동 제출하지 않았다.
