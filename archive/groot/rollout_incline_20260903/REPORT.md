# 실제 GR00T Incline rollout sub-task 분리

- 데이터: `leapshared/rollout_Incline_20260903_20260903_190043` episode 0
- 지시문: `Place the four metal weights on the blue cart on the ramp.`
- 기준: 수레 위에 안정적으로 남아 있는 금속 추 개수
- 판정: 실제 외부 카메라 영상 + 오른쪽 그리퍼 상태를 함께 검토

## 분리 결과

| stage | 의미 | 구간 | 원본 프레임 | 영상 테두리 |
|---:|---|---:|---:|---|
| 0 | zero_weights | 0.00–51.10초 | 0–1532 | red |
| 1 | one_weight | 51.10–65.33초 | 1533–1959 | orange |
| 2 | two_weights | 65.33–94.70초 | 1960–2840 | yellow |
| 3 | three_weights | 94.70–102.87초 | 2841–3085 | lime |
| 4 | four_weights_terminal | 102.87–109.70초 | 3086–3290 | deepskyblue |

## 성공 경계

| 진입 stage | 시각 | 프레임 | 근거 |
|---:|---:|---:|---|
| 1 | 51.100초 | 1533 | first weight released into cart and remains |
| 2 | 65.333초 | 1960 | second weight released into cart and remains |
| 3 | 94.700초 | 2841 | third weight released into cart and remains |
| 4 | 102.867초 | 3086 | fourth weight released into cart; task terminal |

## 실패와 보조 개입

- 실패한 grasp/release 후보: 3회 — 22.07초, 40.80초, 81.63초
- 사람이 남은 추를 다시 잡기 쉬운 위치로 옮긴 구간: 3회 — 23.00초, 67.00초, 83.00초
- 사람은 수레에 추를 넣지 않았다. 성공한 네 번의 적재는 정책 팔이 수행했다.
- 따라서 이 episode는 완전 무개입 rollout이 아니라 source-object reposition 보조가 있는 rollout이다.

## 사용 범위

- 5 FPS 학습/평가용 label 549개를 CSV로 내보냈다.
- 기존 5단계 Incline probe와는 label 의미가 달라 점수를 직접 비교하지 않는다.
- 같은 지시문의 rollout을 더 모은 뒤 episode 단위 train/validation split으로 평가해야 한다.
