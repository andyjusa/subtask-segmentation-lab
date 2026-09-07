# Sub-task 분리 실험표

2026-09-07 정리. 아래는 **당시 저장 결과**이며 이번 재실행은 VALIDATION.md에 분리했습니다.
서로 다른 데이터·지표·분할의 숫자를 하나의 순위로 비교하면 안 됩니다.

## 1. π0.5 pick-and-place

성공 1 episode(410프레임), 실패 1 episode. picking → placing → complete.
성공 영상에서 stride로 프레임을 나눈 검증이며 episode 일반화 검증이 아닙니다.

| 입력 | 차원 | validation accuracy | 의미·한계 |
|---|---:|---:|---|
| continuous action | 6 | 98.78% | 낮은 비용, 실패에서도 다음 단계 오탐 |
| action-expert hidden | 1024 | 100.00% | replan 간 중복 특징, unique 9개 |
| VLM prefix hidden | 2048 | 93.90% | 장면 정보를 포함하나 grasp 성공 보장 안 됨 |
| 사후 FAST histogram | 2049 | 93.90% | π0.5 원래 출력 토큰이 아니라 사후 변환 |

GT 경계 170/354프레임. action raw 예측 170/350. 안정화는 연속 5프레임 조건.
실패 episode에서는 action이 197프레임에 placing을 잘못 알렸습니다.
근거: `packages/pi05/results/representation-comparison.json`, `packages/pi05/reports/experiment.md`.

## 2. GR00T LIBERO — frozen linear probe

3 tasks ×20 episodes=60, rollout 성공59/60. episode 단위60/20/20, 동일5seeds.
라벨은 시뮬레이터 predicate. 첫 predicate 최초 참이 경계이며 terminal은 주 stage 평가에서 제외합니다.

| 표현(native) | 차원 | Stage macro F1 | Boundary event F1 ±5 steps |
|---|---:|---:|---:|
| R0 raw state | 8 | .816 ± .041 | .391 ± .152 |
| R1 predicted action chunk | 112 | .815 | .221 ± .115 |
| R2 state encoder | 1536 | .936 | .106 |
| R3 VLM hidden | 2048 | .953 ± .043 | .120 ± .113 |
| R3b vision-token hidden | 2048 | .951 ± .045 | .154 ± .095 |
| R4 action encoder | 1536 | .844 ± .041 | .369 ± .166 |
| R5 final DiT hidden | 1024 | .939 | .101 |
| R6 VLM+DiT | 3072 | .948 | .017 |

숫자의 ±는 seed 간 표준편차. state/action 계열이 정확한 전환 시점에 유리했고,
VLM은 단계 구분이 좋아도 경계 시점 검출은 낮았습니다. 실패가1개뿐이라 실패 FPR은 매우 불안정합니다.
native/common256 전체 결과: `archive/groot/results.csv`.

추가 causal Transformer: R4 common256 경계 F1 .487 ± .184, R6 .384 ± .154.
표본이 적어 개선을 확정할 수 없습니다. `archive/groot/transformer_streaming/`.
계측 비용: 모델223.196ms, snapshot1.281ms, hook .033ms. 15 paired episodes에서 계측 전후 성공·종료 동일.

## 3. Backpack — hidden + causal Transformer

open backpack → put object → close backpack. 오른손 동작 기준 약한 라벨.
layer16 hidden2048D,1FPS,과거8프레임,2-layer causal Transformer.
12 held-out episodes의 5 anchor씩60케이스: 정확도80%, QA91.67%(36), 전환62.5%(24).
close 지연이12개 오류 중8개. probe 평균2.581ms는 backbone 제외입니다.
전체 프레임 F1 또는 event F1로 해석하지 않습니다.
실행 템플릿: `archive/backpack/run_hidden_transformer_template.py`; 원본 체크포인트/특징은 미포함.

## 4. 경사면 실제 시연 50개

`leapshared/Incline_new_20260902_203009`: 9496 samples,5FPS.
단계는 첫째 추 → 둘째 추 → 셋째 추 → 넷째 추 → 마지막 pointing.
gripper release 기반 약한 경계 라벨,40/10 episode 고정 분리;40 중32학습/8선택.

| 입력 / 모델 | ordered macro F1 | 경계 MAE(초) |
|---|---:|---:|
| R3b vision / MLP | .9020 | .997 |
| R3b vision / Linear | .8990 | 1.028 |
| R3 full / Linear | .9008 | 1.015 |
| R3 full / MLP | .8957 | 1.040 |
| AllTracker / MLP | .8259 | 1.880 |
| AllTracker+state+action14 / MLP | .8309 | 1.733 |
| vision+AllTracker+state+action14 단순 concat | .8879 | 1.152 |
| vision gated residual +AllTracker+action14 /5seeds | .9046 ± .0005 | .979 ± .011 |

융합과 동조건 vision-only 차이 .0059의 paired95%CI가0을 포함: **유의한 향상 확정 아님**.
단순 vision-only를 기본으로 두고 융합은 후보로 남깁니다.
ep2는 모든 후보에서 나쁘며 경계가5.4~11.1초 빨라지는 문제가 남았습니다.
이 ordered decoder는 양방향 smoothing+전체 episode DP로5단계를 강제하므로 온라인 점수가 아닙니다.
전체 tracker/action/state 조합 표는 `archive/groot/incline_new/**/results.json` 및 `vision_fusion/results.csv`.

## 5. 실제 로봇 rollout — 새로운 분포의 1 episode

`leapshared/rollout_Incline_20260903_20260903_190043`: 109.7초,549 samples.
사람이 영상에서 잡은 성공 누적 경계51.10/65.33/94.70/102.87초, 추가 수동 검수가 필요합니다.
단계는 추 성공 개수0→1→2→3→4. 시연 데이터의 마지막 pointing과 정의가 다릅니다.

| 입력 / Logistic Regression | raw macro F1 | ordered macro F1 | 경계 MAE(초) |
|---|---:|---:|---:|
| R3b native | .7195 | .8665 | 2.833 |
| R3b PCA256 | .7295 | .8681 | 2.783 |
| R3 full native | .7280 | .8681 | 2.783 |
| R3 full PCA256 | .7333 | .8681 | 2.783 |

**같은 영상의 stage-aware5fold OOF**이며 기존 분류기를 그대로 적용한 검증이 아닙니다.
선택 후보의 경계45.2/67.6/96.6/101.8초: 첫 경계5.9초 빠르고 ±1초 이내 경계0/4.
hidden에 단계 정보가 있다는 탐색 결과일 뿐, 성공 판별/온라인 pause의 준비 완료 근거는 아닙니다.

## 6. 보조 실험 — 색 변화

change/no-change20쌍=40episodes, pair 단위 분리,5seeds.
C2 R3b causal Transformer: boundary F1±5 .681±.065, 실패 FPR0, probe2.612ms.
MLP .671±.375, FPR .15. 차이 CI가 넓어 Transformer 우위 확정 안 됨.
고정색/랜덤색 및 두 경계 실험: `archive/groot/*COLOR*`, `*BOUNDARY*`.

## 7. SciEdu 73초 — Cosmos 시간창/FPS

경사면50개와 별도인 1 episode입니다. 카트 경사로 이동 → 추 추가 → 추 제거 → 책상 복귀,
참조 경계22/47.5/63.25초, 종료73초입니다.
Cosmos-Reason2-2B backbone의 vision mean에 PCA+Linear probe를 적용했습니다.
LoRA·action head는 끄고 외부 카메라만 시간에 따라 바꾸며 손목 카메라는 첫 프레임으로 고정했습니다.

| 입력 | Stage macro F1 | Balanced accuracy | backbone 평균 지연 | peak VRAM |
|---|---:|---:|---:|---:|
| 6초·1 FPS | .946 | .955 | 447ms | 약6.8GB |
| 6초·5 FPS | .970 | .975 | 2,287ms | 약9.6GB |

동일 episode 안 시간 블록 검증이며 경계 ±0.5초의 애매한 표본은 주 분류 평가에서 제외했습니다.
6초·1 FPS의 예측 경계27/50/66초는 모두 늦었습니다. 높은 Stage F1이 정확한 전환이나
새 episode 일반화·자유형 QA를 보장하지 않습니다. 비용 대비6초·1 FPS는 **이 영상 안의 후보**입니다.
전체8조건과 오답: [보존 보고서](../archive/groot/sciedu_cosmos/REPORT.md).

## 다음 검증

- 라벨을 사람2명이 독립 검수하고 성공 개수/다음 동작/pointing 정의를 먼저 고정
- 실제 rollout을 여러개 수집해 episode 단위 별도 test로 평가
- 기존 시연 probe 무학습 전이와 rollout으로 학습한 probe를 별도 표에 기록
- online causal decoder에서 event F1, 실패 FPR, 지연과 pause 후 실제 환경 변화 확인
