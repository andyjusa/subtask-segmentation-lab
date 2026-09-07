# GR00T backbone hidden state 50-episode probe

## 결론

GR00T backbone hidden은 기존 AllTracker·state·action보다 subtask 단계를 훨씬 잘
분리했다. 최고 결과는 **R3b vision hidden only + MLP**로 ordered macro F1
**0.9020**, 경계 평균 오차 **0.997초**, 중앙값 **0.133초**다.

반면 기존 최적 feature에 hidden을 그대로 concat한 결과는 MLP F1 **0.8879**로
hidden 단독보다 낮았다. 따라서 현재 데이터에서는 `AllTracker + state + action +
hidden` 전체 결합보다 **GR00T vision hidden 단독**이 더 단순하고 정확하다.

## Hidden 추출 계약

- 모델: GR00T N1.7, Cosmos-Reason2-2B backbone
- layer: `select_layer=16`
- R3: `backbone_features`의 attention-mask mean, 2048D
- R3b: 같은 tensor의 image-mask mean, 2048D
- 입력: 외부·왼쪽 손목·오른쪽 손목 카메라와 전체 task instruction
- 시간 해상도: 5 FPS
- LoRA adapter: 비활성화
- action head: 미사용
- 저장 dtype: float16, 학습 시 float32 변환
- 데이터: 50 episodes, 9,496 samples, 50 NPZ files, 약 58MiB

영상 컨테이너의 종료 timestamp 반올림 때문에 14개 에피소드에서 마지막 관측이
1프레임 부족했다. 이 경우 마지막 영상 관측을 한 번 반복하는 causal tail padding을
적용했다. 중간 구간에는 보간이나 미래 프레임을 사용하지 않았다.

## 추출 비용

| 항목 | 결과 |
|---|---:|
| batch size | 4 |
| 평균 backbone latency | 74.36ms/sample |
| peak VRAM | 6,536.55MiB |
| 전체 추출 시간 | 995.97초 |

이 latency는 backbone을 별도로 실행한 오프라인 비용이다. 실제 GR00T VLA 추론 중
이미 계산된 hidden을 hook으로 가져오면 backbone 재실행은 필요 없고 pooling·복사
비용만 추가된다.

## 동일 split 결과

40개 outer-train episode를 32 train / 8 selection으로 나누고, 고정된 10개 test
episode에서 seed 42로 평가했다. 모든 조건은 5 FPS, train-only StandardScaler,
동일 Linear·MLP와 ordered decoder를 사용했다.

| Feature | Classifier | Ordered F1 | 경계 평균 오차 | 경계 중앙값 |
|---|---|---:|---:|---:|
| AllTracker | MLP | 0.8259 | 1.880초 | 0.567초 |
| AllTracker + state + action14 | MLP | 0.8309 | 1.733초 | 0.500초 |
| R3 full hidden only | Linear | 0.9008 | 1.015초 | 0.167초 |
| R3 full hidden only | MLP | 0.8957 | 1.040초 | **0.133초** |
| R3b vision hidden only | Linear | 0.8990 | 1.028초 | 0.183초 |
| **R3b vision hidden only** | **MLP** | **0.9020** | **0.997초** | **0.133초** |
| AllTracker + state + action14 + R3 | MLP | 0.8869 | 1.150초 | **0.133초** |
| AllTracker + state + action14 + R3b | MLP | 0.8879 | 1.152초 | **0.133초** |

## 해석

R3와 R3b가 비슷하고 R3b가 소폭 앞선 것은 모든 episode에서 instruction이 동일해
text token이 단계 구분에 새로운 정보를 거의 주지 않았기 때문이다. 반대로 raw
concat fusion은 차원이 3,286D로 늘고 tracker와 vision hidden이 중복된 시각 정보를
제공해 hidden 단독보다 성능이 하락한 것으로 보인다.

이번 결과는 단일 split·단일 seed의 1차 비교다. 최종 구조를 확정하려면 5-seed
평가와 train-only PCA/learned projection을 적용한 fusion ablation이 필요하다.
