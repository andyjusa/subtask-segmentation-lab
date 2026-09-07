# Vision hidden 결합 sub-task 분리 실험

## 결론

최종 선택은 **GR00T R3b vision hidden + gated residual motion/action probe**다.
vision hidden이 주 분류를 담당하고, AllTracker와 robot action은 클래스별 gate를 거쳐
작은 보정값만 더한다. 단순 concat보다 중복 특징의 영향을 제한하면서 기존 특징을
활용하는 구조다.

5개 고정 seed에서 ordered macro F1은 **0.9046 ± 0.0005**, 경계 평균 오차는
**0.979 ± 0.011초**였다. 동일 실행의 vision-only는 각각 **0.8987 ± 0.0109**,
**1.033 ± 0.100초**였다. 평균 F1은 0.0059 높고 seed 간 편차는 약 21배 작았다.
다만 5-seed paired 95% CI에는 0이 포함되므로, 통계적으로 확정적인 우월성보다는
**성능 안정화가 확인된 유력한 결과**로 해석해야 한다.

## 선택 구조

```text
R3b vision hidden 2048D -> MLP --------------------------┐
                                                         ├-> 5-stage logits
AllTracker 1208D + action 14D -> small MLP -> class gate ┘
```

- 입력은 5 FPS이며 timestamp, episode 진행률, 정답 경계, 미래 frame을 쓰지 않는다.
- action은 두 gripper 차원을 제외한 14D다.
- gate는 sigmoid로 제한해 보조 branch가 vision 판단을 통째로 덮지 못하게 했다.
- probe는 349,455 parameters이며 frozen GR00T와 AllTracker는 학습하지 않는다.

## 핵심 결과

| 방법 | Ordered macro F1 | 경계 MAE | 1초 이내 | 판단 |
|---|---:|---:|---:|---|
| 기존 R3b vision MLP, 과거 seed 42 | 0.9020 | 0.997초 | 0.875 | 기준 |
| raw concat: vision+tracker+state+action | 0.8879 | 1.152초 | - | 기각 |
| late probability fusion | 0.9035 | 0.975초 | 0.900 | 개선 폭 작음 |
| projected fusion, 초기 seed 42 | 0.9066 | 0.987초 | 0.875 | seed 불안정 |
| **gated residual fusion, seed 42** | **0.9087** | **0.953초** | 0.875 | 최고 단일 run |
| residual + 과거 1초 delta | 0.9067 | 0.978초 | 0.875 | 기각 |
| R3b + R3 + motion/action | 0.9010 | 1.028초 | 0.875 | 기각 |
| residual + robot state | 0.9033 | 1.020초 | 0.850 | 기각 |
| **gated residual, 5-seed 평균** | **0.9046 ± 0.0005** | **0.979 ± 0.011초** | **0.895** | 최종 선택 |
| 5-model probability ensemble | 0.9064 | 0.962초 | 0.900 | 정확도 우선 시 선택 |

5-model ensemble은 더 높지만 probe 비용이 5배이고 episode 2를 해결하지 못해 기본
배포안에서는 제외한다. 오프라인 분석에서 정확도만 우선하면 사용할 수 있다.

## 왜 이 구조가 나았나

R3b는 cart, 추, pointing 대상 같은 시각적 진행 상태를 직접 담아 가장 강한 단일
특징이었다. AllTracker와 action은 움직임과 로봇 명령을 제공하지만 시각 특징과
겹치거나 노이즈가 있어 raw concat에서는 오히려 성능이 떨어졌다. residual gate는
vision을 기본 판단으로 유지하고 보조 정보가 확실한 경우에만 logit을 보정해 seed에
따른 과적합을 크게 줄였다.

## 남은 실패와 한계

- 느린 validation episode 2의 F1은 모든 결합에서 약 0.32에 머물렀다. 예측 경계가
  실제보다 5.4–11.1초 빨라 전체 경계 MAE 대부분을 차지한다.
- 이 문제는 feature를 더 concat하는 것으로 해결되지 않았다. 다음 실험은 느린 수행을
  포함한 train episode 수집이나 속도 변화 augmentation이 우선이다.
- ordered decoder는 전체 episode 확률을 보고 5단계 순서를 강제하는 offline
  후처리다. 실시간 전환에는 causal boundary head를 별도로 검증해야 한다.
- 동일 validation split이 과거 실험에서도 사용됐으므로 완전히 새로운 holdout에서
  최종 재검증이 필요하다.

세부 seed 결과는 `results.csv`, 각 확률과 예측은 `seed_*/predictions/`, 모델은
`seed_*/residual.pt`에 저장했다.
