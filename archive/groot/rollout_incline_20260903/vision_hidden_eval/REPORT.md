# 실제 rollout의 GR00T vision hidden stage 분리

## 결론

가장 좋은 조건은 `R3b_vision_pca256`이며 blocked OOF ordered macro F1은 **0.868**, 경계 평균 오차는 **2.78초**다.

| 표현 | Raw macro F1 | Ordered macro F1 | 경계 MAE | 경계별 오차(초) |
|---|---:|---:|---:|---|
| R3b_vision_native | 0.720 | 0.866 | 2.83초 | 5.90, 2.47, 1.90, 1.07 |
| R3b_vision_pca256 | 0.729 | 0.868 | 2.78초 | 5.90, 2.27, 1.90, 1.07 |
| R3_full_native | 0.728 | 0.868 | 2.78초 | 5.90, 2.27, 1.90, 1.07 |
| R3_full_pca256 | 0.733 | 0.868 | 2.78초 | 5.90, 2.27, 1.90, 1.07 |

## Stage별 결과

| Stage | 표본 | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0 | 256 | 1.000 | 0.883 | 0.938 |
| 1 | 71 | 0.634 | 1.000 | 0.776 |
| 2 | 147 | 0.938 | 0.925 | 0.932 |
| 3 | 41 | 1.000 | 0.634 | 0.776 |
| 4 | 34 | 0.850 | 1.000 | 0.919 |

Stage 1의 precision이 낮은 주원인은 첫 경계를 5.9초 일찍 예측한 것이다. Stage 3의 recall은 진입을 1.9초 늦게, 종료를 1.07초 일찍 예측해 구간이 양쪽에서 줄어든 영향을 받았다.

## 비교 영상

`reference-left_prediction-right.mp4`에서 왼쪽은 수동 검토 정답, 오른쪽은 R3b 예측이다. 테두리 색 변화 시점의 차이가 경계 오차다.

## 평가 계약

- 입력: 세 카메라 + task instruction, 5 FPS
- R3b: GR00T backbone의 image-mask token mean
- R3: vision·instruction을 포함한 attention-mask token mean
- backbone과 probe 입력 추출 시 LoRA adapter 및 action head 비활성화
- 각 stage를 시간 순서대로 5개 block으로 나눈 뒤 한 block씩 test
- test block 앞뒤 1초는 해당 fold의 train에서 제외
- PCA는 각 fold의 train feature만 사용

## 해석 제한

이 수치는 한 episode 내부의 표현 분리 가능성이다. 서로 다른 episode로의 일반화 성능은 아니며, 동일 task rollout을 추가 수집한 뒤 episode 단위 평가가 필요하다.
