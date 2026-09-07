# TAPNext++ 기반 5단계 Sub-task 분류

## 결론

TAPNext++ 추적 정보만으로 5단계를 어느 정도 분리할 수 있다. MLP가 Linear보다 소폭
좋았지만 차이는 작다. 대부분의 validation episode는 잘 분리했으나, 작업 진행이 유난히
느린 episode 0–2에서는 경계를 너무 일찍 예측했다. 따라서 현재 모델을 바로 자동 정지나
전환 신호로 사용하기에는 예외 처리나 시간 변화에 강한 temporal model이 추가로 필요하다.

## 고정 조건

- 입력: 400개 점의 현재 변위 800D + visibility 400D + 움직임 통계 8D
- 시간 인덱스와 정답 경계는 입력에서 제외
- 30 FPS 원본을 5 FPS로 표본화
- episode 단위 32 train / 8 model selection / 10 validation
- Linear: 단일 affine layer, 6,045 parameters
- MLP: 1208→128→64→5, 163,333 parameters
- 출력 단계 순서가 항상 0→1→2→3→4라는 조건의 ordered decoding도 평가

## 결과

| 모델 | Raw macro F1 | Ordered macro F1 | 경계 MAE | ±1초 경계 | ±2초 경계 |
|---|---:|---:|---:|---:|---:|
| 시간 중앙값 baseline | 0.637 | 0.637 | 4.44초 | 35.0% | 55.0% |
| Linear | 0.761 | 0.754 | 2.69초 | 65.0% | 75.0% |
| MLP | **0.768** | **0.771** | **2.54초** | **70.0%** | **75.0%** |

MLP의 validation episode별 ordered macro F1 중앙값은 0.931이다. 최고 episode 14는
0.984였지만, 최악 episode 2는 0.178이었다. episode 2의 실제 경계는
19.27/28.03/39.10/50.80초인데 모델은 10.40/18.80/23.40/32.40초로 모두 일찍 예측했다.

## 해석

단순 시간 baseline보다 명확히 좋아서 추적 특징이 동작과 물체 상태를 반영한다. 하지만
Linear와 MLP의 차이가 작고, 느린 수행에서 둘 다 같은 방향으로 실패한다. 현재 병목은
분류기 용량보다 단일 프레임의 추적 상태만 사용하는 입력 표현이다. 다음 실험은 최근
1–3초 특징을 받는 작은 causal temporal MLP/1D convolution 또는 Transformer와 진행 속도
augmentation이 적합하다.

이 결과는 같은 작업을 반복한 새 episode 일반화 결과이며, 새로운 작업 종류에 대한
일반화 결과는 아니다.
