# BC1 vs BC1+BoN50 — 성공 즉시 종료 평가

- Actor: 기존 seed42 BC1, step10,000.
- Critic: BC2 평가와 동일한 기존 완료 critic, step10,000.
- 4개 task × 평가 seed0 × 50episode × 2방법 =400episode.
- 최대 GPU1장. task별 별도 잡을 전역 afterany 체인으로 순차 실행.
- 각 잡: GPU1/CPU8/RAM96GiB/sub-own/시간 상한1시간.
- BoN:N50,temperature0,denoising4,예측·실행 chunk최대16.
- 성공한 simulator step에서 즉시 종료. 실패는 task horizon500/600step.
- 성공 여부 정의는 동일하며 episode 길이·난수 소비는 과거 full-horizon 평가와 달라질 수 있다.
- W&B OFF. 결과·영상·BoN입력/후보/Q점수는 로컬 저장.
- 이전 막 제출했던 full-horizon 체인138948~138958은 이 프로토콜로 교체하기 위해 취소했다.

| Method | Task | Job | Dependency |
|---|---|---|---|
| BC1 | CoffeeSetupMug | 138991 |  |
| BC1 | PnPMicrowaveToCounter | 138992 | afterany:138991 |
| BC1 | TurnOffStove | 138993 | afterany:138992 |
| BC1 | PnPCounterToMicrowave | 138994 | afterany:138993 |
| BC1+BoN50 | CoffeeSetupMug | 138996 | afterany:138994 |
| BC1+BoN50 | PnPMicrowaveToCounter | 138997 | afterany:138996 |
| BC1+BoN50 | TurnOffStove | 138998 | afterany:138997 |
| BC1+BoN50 | PnPCounterToMicrowave | 138999 | afterany:138998 |

최종 집계 잡: 139001

결과 루트: `/home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/robocasa-comparison/seed42-bc1-vs-bon50-eval0-50ep-serial-20260919-stop-success`

진행도:

```bash
python /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/scripts/robocasa/show_eval_progress.py --manifest /home/nas_main/dohyunlee/jh_ws/DEAS-Isaac-GR00T/output/robocasa-comparison/seed42-bc1-vs-bon50-eval0-50ep-serial-20260919-stop-success/manifest.json
```

검증: 새 성공 종료+기존 rollout/video 테스트16개, 결과/직렬 제출 테스트14개 총30개 통과. 성공이 chunk 3번째 step에 발생했을 때 3step만 실행, 실패 horizon 유지, NEXT_STEP autoreset 집계와 영상 종료 확인.
