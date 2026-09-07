# AllTracker 기반 5단계 Sub-task 분류

## 결론

AllTracker 추적 특징은 TAPNext++보다 전체 validation의 어려운 에피소드에 강했다. 같은
1208차원 입력과 같은 episode split에서 MLP ordered macro F1은 0.771에서 0.826으로,
경계 평균 오차는 2.54초에서 1.88초로 개선됐다. 다만 일반적인 경계의 중앙 오차는
0.42초에서 0.57초로 소폭 증가했고, 매우 느린 episode 2는 여전히 조기 전환했다.

## 고정 조건

- 데이터: `Incline_new_20260902_203009`의 episode 0–49
- 추적기 입력: 외부 카메라 256×192, 30 FPS
- 특징: 20×20 점의 현재 변위 800D + visibility 400D + 움직임 통계 8D
- 시간 인덱스와 정답 경계는 입력에서 제외
- probe 입력은 5 FPS로 표본화
- episode 단위 32 train / 8 model selection / 10 validation
- Linear: 단일 affine layer, 6,045 parameters
- MLP: 1208→128→64→5, 163,333 parameters
- 출력 순서를 0→1→2→3→4로 제한한 ordered decoding도 평가

## AllTracker 특징 생성

| 항목 | 결과 |
|---|---:|
| 에피소드 / 프레임 | 50 / 56,849 |
| 전체 영상 길이 | 1,894.97초 |
| RTX 5060 Ti 평균 추론 | 15.58 ms/frame |
| 전체 추론 시간 | 886.49초 |
| 최대 VRAM allocated / reserved | 441.7 / 662.0 MiB |
| 평균 visibility | 99.25% |
| 저장 크기 | 62 MiB |

## 분류 결과

| 추적기 | Probe | Raw macro F1 | Ordered macro F1 | 경계 평균 오차 | 경계 중앙 오차 | ±1초 | ±2초 |
|---|---|---:|---:|---:|---:|---:|---:|
| TAPNext++ | Linear | 0.761 | 0.754 | 2.69초 | **0.43초** | 65.0% | 75.0% |
| TAPNext++ | MLP | 0.768 | 0.771 | 2.54초 | **0.42초** | **70.0%** | 75.0% |
| AllTracker | Linear | 0.778 | 0.816 | 1.93초 | 0.60초 | 62.5% | **77.5%** |
| AllTracker | MLP | **0.778** | **0.826** | **1.88초** | 0.57초 | **70.0%** | 75.0% |

## 느린 episode 0–2

| Episode | TAPNext++ MLP F1 / 경계 MAE | AllTracker MLP F1 / 경계 MAE |
|---:|---:|---:|
| 0 | 0.669 / 4.09초 | **0.809 / 2.96초** |
| 1 | 0.585 / 4.72초 | **0.839 / 1.80초** |
| 2 | 0.178 / 13.05초 | **0.298 / 9.35초** |

## 해석

AllTracker의 dense flow는 느리고 우회적인 궤적에서도 단계 순서를 더 안정적으로 유지해
전체 평균과 episode 0–1을 크게 개선했다. 반면 정상 속도 에피소드에서 경계를 정밀하게
맞추는 정도는 TAPNext++와 비슷하거나 조금 낮다. Linear와 MLP의 차이도 ordered F1
0.010에 불과하므로 분류기 용량보다 tracker 표현 차이가 더 큰 변인이다.

episode 2는 두 추적기 모두 실제 경계보다 일찍 전환한다. 따라서 AllTracker로 교체하는
것만으로 속도 분포 이동 문제가 해결되지는 않으며, 다음 비교에서는 과거 1–3초 특징을
입력하는 temporal probe나 속도 augmentation을 별도 변인으로 두는 것이 적절하다.

AllTracker의 현재 방식은 전체 클립을 받아 첫 프레임 기준 dense flow를 계산하므로,
실시간 배포 지연이나 인과성까지 검증한 결과는 아니다.

## Action·state 결합 실험

AllTracker 입력에 현재 프레임과 정렬된 `observation.state` 16D와 `action` 16D를 각각 또는
함께 추가했다. 나머지 split, seed, scaler, probe 구조와 5 FPS 표본화 조건은 동일하다.

| 추가 입력 | Probe | Raw macro F1 | Ordered macro F1 | 경계 평균 오차 | 경계 중앙 오차 | ±1초 | ±2초 |
|---|---|---:|---:|---:|---:|---:|---:|
| 없음 | Linear | 0.778 | 0.816 | 1.93초 | 0.60초 | 62.5% | 77.5% |
| 없음 | MLP | 0.778 | 0.826 | 1.88초 | 0.57초 | 70.0% | 75.0% |
| state 16D | Linear | 0.766 | 0.804 | 2.09초 | 0.75초 | 57.5% | 72.5% |
| state 16D | MLP | 0.786 | 0.821 | 1.88초 | 0.53초 | 70.0% | 75.0% |
| action 16D | Linear | 0.768 | 0.805 | 2.07초 | 0.68초 | 57.5% | 75.0% |
| action 16D | MLP | 0.784 | 0.830 | 1.81초 | 0.48초 | 67.5% | 77.5% |
| state 16D + action 16D | Linear | 0.786 | 0.823 | 1.85초 | 0.63초 | 65.0% | 75.0% |
| state 16D + action 16D | MLP | 0.791 | 0.815 | 1.92초 | **0.42초** | 65.0% | 77.5% |
| action 14D, gripper 제외 | Linear | 0.764 | 0.813 | 2.00초 | 0.68초 | 67.5% | 75.0% |
| action 14D, gripper 제외 | MLP | 0.790 | **0.833** | 1.76초 | 0.53초 | **72.5%** | 75.0% |
| state 16D + action 14D, gripper 제외 | Linear | 0.782 | 0.823 | 1.91초 | 0.60초 | 65.0% | 77.5% |
| state 16D + action 14D, gripper 제외 | MLP | **0.800** | 0.831 | **1.73초** | 0.50초 | **72.5%** | **77.5%** |

### 해석

- `state+action`을 무조건 합친다고 모든 점수가 좋아지지는 않았다. 전체 32D를 넣은 MLP는
  경계 중앙 오차는 가장 작지만 ordered F1과 평균 오차가 악화됐다.
- ordered F1이 가장 높은 구성은 `AllTracker + gripper를 제외한 action + MLP`다.
- 경계 평균 오차와 Raw F1이 가장 좋은 구성은
  `AllTracker + state + gripper를 제외한 action + MLP`다.
- 경계 라벨은 오른쪽 그리퍼 action으로 생성됐다. Linear는 이 action 15번에 가장 큰
  가중치를 줬지만, 실제로 그리퍼 action을 제거했을 때 MLP 성능이 더 좋아졌다. 따라서
  이번 향상은 직접적인 라벨 누설보다 나머지 관절 운동 정보의 효과로 보는 것이 타당하다.
- 느린 episode 2의 경계 평균 오차는 기본 9.35초에서 최선 8.60초로만 줄었다. action과
  state를 추가해도 극단적인 속도 분포 이동 문제는 남는다.
