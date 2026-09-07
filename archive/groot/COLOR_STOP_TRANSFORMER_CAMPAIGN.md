# GR00T 색 변화 Stop 소형 Transformer 실험

## 한 줄 결론

가장 좋은 Transformer는 `R3b vision-position + exact boundary label + boundary-only loss`였다.
5-seed Boundary F1@±5는 `0.681 ± 0.065`, no-change FPR은 `0.000`이다. 기존 MLP의
`0.671 ± 0.375`, FPR `0.150`보다 평균과 안정성은 좋아졌지만 paired 차이는
`+0.010 ± 0.339`로 불확실하고, 사전 교체 기준 `+0.03`을 넘지 못했다. 따라서 현재
기본안은 작은 MLP를 유지하고 Transformer는 오정지를 더 중요하게 보는 안전 대안으로 남긴다.

## 고정 평가 계약

- 데이터: 20 counterfactual pair, 총 40 episode, episode당 24 frame
- 같은 pair의 change/no-change 영상은 항상 같은 split에 배치
- pair 단위 60/20/20 split, seeds `17, 29, 41, 53, 67`
- PCA/StandardScaler는 train split으로만 학습
- 미래 frame, Viterbi, backtracking 없이 causal first-event 평가
- 주 지표: Boundary event F1@±5
- gate: no-change FPR ≤ 0.25
- 교체 조건: 기존 MLP보다 F1이 최소 0.03 개선
- screening의 seed 17 결과로 최종 결론을 내리지 않고 상위 후보만 5-seed 재검증

## 공통 Transformer

- causal sliding window
- 기본 window 16
- model dimension 256, attention head 4개
- encoder 2 layers, FFN 512
- 약 1.13M parameters
- stage head와 boundary head
- 최대 30 epochs, validation event F1 기반 checkpoint 선택, patience 6
- 기본 boundary 학습 label은 GT ±2 frame

## 1차 단일변수 Screening

아래 수치는 seed 17 한 번의 screening 결과다. 최종 성능이 아니라 후보 축소용이다.

| 후보 | 기준선에서 바꾼 축 | Boundary F1@±5 | no-change FPR | False boundary/episode | 결정 |
|---|---|---:|---:|---:|---|
| T5 | Stage loss 제거 | **0.889** | 0.00 | 0.125 | 최종 후보 |
| T8 | 입력에 explicit delta 추가 | **0.889** | 0.00 | 0.125 | 최종 후보 |
| T13 | Vision+Instruction+DiT 3-way | 0.750 | 0.00 | 0.125 | 최고 fusion 후보 |
| T6 | Exact boundary label | 0.667 | 0.00 | **0.000** | 안정성 후보 |
| T10 | Vision+Action output | 0.667 | 0.00 | 0.250 | 탈락 |
| T11 | Vision+Instruction+Action output | 0.667 | 0.00 | 0.250 | 탈락 |
| T12 | Vision+DiT hidden | 0.667 | 0.00 | 0.250 | 탈락 |
| T4 | 128d, 1 layer | 0.533 | 0.00 | 0.875 | 탈락 |
| T7 | Boundary radius ±1 | 0.462 | 0.00 | 0.750 | 탈락 |
| T1 | Window 4 | 0.400 | 0.00 | 0.000 | 주 지표 개선 없음 |
| T0 | Transformer baseline | 0.400 | 0.25 | 1.000 | 기준선 |
| T9 | Vision+Instruction | 0.381 | 0.50 | 1.625 | gate 실패 |
| T3 | 64d, 1 layer | 0.333 | 0.00 | 0.125 | 탈락 |
| T2 | Window 8 | 0.000 | 0.00 | 0.000 | 탈락 |

### Screening 해석

- 색 변화 task에서 stage 분류는 목표와 충돌한다. 같은 색이 pair마다 stage 0/1 양쪽에
  나타나므로 stage loss를 제거하자 seed 17 성능이 크게 올랐다.
- explicit delta도 순간 변화 검출에 유리했지만 5-seed에서는 불안정했다.
- Vision+Instruction은 오히려 no-change 오정지를 늘렸다.
- Action/DiT/fusion은 vision 단독을 일관되게 개선하지 못했다.
- 작은 모델은 파라미터는 줄였지만 boundary event를 충분히 학습하지 못했다.

## 2차 결합 Ablation

상위 후보를 부모로 삼아 다른 축을 하나씩 추가했다.

| 후보 | 부모 → 추가 변경 | Boundary F1@±5 | no-change FPR | 결정 |
|---|---|---:|---:|---|
| C2 | Boundary-only → Exact label | **0.750** | 0.00 | 5-seed 재검증 |
| C1 | Boundary-only → Explicit delta | 0.667 | 0.00 | 부모보다 하락 |
| C4 | 3-way fusion → Boundary-only | 0.600 | 0.00 | 부모보다 하락 |
| C3 | Boundary-only → 128d·1 layer | 0.333 | 0.00 | 부모보다 하락 |

변경을 여러 개 합친다고 좋아지지 않았다. 유일하게 안정적으로 남은 조합은 boundary-only와
exact label이다.

## 5-seed 최종 결과

| 구조 | Boundary F1@±5 | no-change FPR | False boundary/episode | Probe 지연 | 파라미터 |
|---|---:|---:|---:|---:|---:|
| C2 R3b + boundary-only + exact label | **0.681 ± 0.065** | **0.000** | 0.125 | 2.612ms | 1,126,659 |
| 기존 R3b MLP | 0.671 ± 0.375 | 0.150 | 별도 구 evaluator | **2.411ms** | 약 73,985 |
| T6 R3b + exact label | 0.633 ± 0.065 | **0.000** | **0.100** | 2.552ms | 1,126,659 |
| T8 R3b + explicit delta | 0.612 ± 0.200 | 0.300 | 0.600 | 2.711ms | 1,192,195 |
| T5 R3b + boundary-only | 0.528 ± 0.212 | 0.150 | 0.300 | **2.442ms** | 1,126,659 |
| T0 Transformer baseline | 0.450 ± 0.080 | 0.050 | 0.375 | 2.672ms | 1,126,659 |
| T13 Vision+Instruction+DiT | 0.443 ± 0.160 | 0.150 | 0.275 | 2.713ms | 1,126,659 |

MLP와 C2는 같은 seed와 같은 pair split으로 비교했다.

- C2 − MLP Boundary F1 평균 차이: `+0.010`
- paired 95% CI: `±0.339`
- CI가 0을 포함하고 사전 최소 개선폭 0.03보다 작음

따라서 C2가 확실히 더 좋다고 결론낼 수 없다. 다만 C2는 모든 seed에서 no-change FPR이
0이었고 F1 분산도 훨씬 작아, 오정지가 중요한 환경에서는 후속 검증 가치가 있다.

## 최종 선택

### 현재 기본안: R3b MLP 유지

- 평균 F1은 C2와 사실상 동률
- 약 15배 적은 학습 파라미터
- 약 0.2ms 빠름
- 기존 라이브 stop 경로가 이미 검증됨

### 안전 대안: C2 Transformer 보존

- exact 한 프레임을 boundary positive로 사용
- stage loss 없이 boundary만 학습
- 16-frame causal history 사용
- 5 seeds 모두 no-change false stop 없음
- 데이터가 늘어나면 우선 재평가할 후보

## 실패한 가설

- 긴 temporal model이면 자동으로 MLP보다 좋아진다: 기각
- Explicit delta를 추가하면 일관되게 개선된다: seed별 변동과 FPR 때문에 기각
- Vision·Instruction·Action/DiT를 합치면 좋아진다: 기각
- 더 작은 Transformer가 작은 데이터에 유리하다: boundary recall 부족으로 기각
- Stage와 Boundary multi-task가 표현을 정규화한다: 이 task에서는 오히려 방해
- 여러 좋은 변경을 합치면 누적 개선된다: 대부분 부모보다 하락

## 제한 및 다음 검증

- 20 pair만으로는 paired CI가 크다. 최소 100 pair 이상이 필요하다.
- 합성 비커의 위치가 고정되어 있어 실제 카메라 domain shift는 측정하지 않았다.
- C2는 사전 교체 기준을 통과하지 못했으므로 live stop 영상을 새로 만들지 않았다.
- 다음 캠페인은 모델을 더 튜닝하기보다 색·조명·비커 위치·카메라 각도를 확장해야 한다.
- 확장 데이터에서도 no-change FPR 0이 유지되면 C2를 실제 hold/zero-action gate에 연결한다.

## 산출물

- 실험 계약: `.autoresearch/contract.toml`
- 실험 계획: `.autoresearch/program.md`
- 원본 ledger: `.autoresearch/runs/color-stop-transformer-20260828-v3/results.jsonl`
- Screening: `remote_results/transformer_campaign_20260828/reports/color_stop_transformer_screen/`
- 결합 ablation: `remote_results/transformer_campaign_20260828/reports/color_stop_transformer_combinations/`
- 5-seed 결과: `remote_results/transformer_campaign_20260828/reports/color_stop_transformer_final/`
- C2 5-seed 결과: `remote_results/transformer_campaign_20260828/reports/color_stop_transformer_c2_final/`

