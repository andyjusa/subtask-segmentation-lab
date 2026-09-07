# GR00T N1.7 causal sliding-window Transformer 실험

## 실험 계약

- 데이터: LIBERO 3 tasks, 60 episodes, 성공 59/60
- 정답: 환경 내부 predicate 최초 만족 시점; terminal 표본은 stage 학습에서 제외
- split: task별 episode 60/20/20, 기존 linear와 동일한 seed [17, 29, 41, 53, 67]
- 실행 표현: R0_raw_state, R1_action_chunk, R2_state_encoder, R3_vlm_hidden, R3b_vision_tokens, R4_action_encoder, R5_final_dit_hidden, R6_vlm_dit_concat
- 실행 투영: common_256, native; train-only StandardScaler/PCA
- 모델: 과거 16개 policy sample만 보는 2-layer causal Transformer
- 출력: stage 0/1 head + first-rising boundary head
- 평가: 순차 streaming forward만 사용하며 offline Viterbi/backtracking은 사용하지 않음
- 경계: validation threshold 선택 후 test event TP/FP/FN으로 F1@±5/±10 계산
- R3/R3b는 GR00T N1.7의 실제 select_layer=16 backbone 출력

## 결과

| 표현 | 투영 | Transformer Boundary F1@±5 | 기존 Linear† F1@±5 | Transformer Stage F1 | 기존 Linear Stage F1 | FP/episode | 실패 FPR | feature ms | probe ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R4_action_encoder | common_256 | 0.487 ± 0.184 | 0.369 | 0.920 ± 0.061 | 0.840 | 0.483 | 0.500 | 0.147 | 3.667 |
| R6_vlm_dit_concat | common_256 | 0.384 ± 0.154 | 0.051 | 0.939 ± 0.044 | 0.948 | 0.700 | 0.000 | 0.507 | 3.930 |
| R2_state_encoder | native | 0.292 ± 0.139 | 0.106 | 0.929 ± 0.054 | 0.936 | 0.700 | 0.500 | 0.126 | 5.279 |
| R3_vlm_hidden | native | 0.287 ± 0.076 | 0.120 | 0.961 ± 0.028 | 0.953 | 0.717 | 0.000 | 0.355 | 5.274 |
| R3b_vision_tokens | native | 0.285 ± 0.129 | 0.154 | 0.966 ± 0.018 | 0.951 | 0.733 | 0.000 | 0.329 | 5.030 |
| R4_action_encoder | native | 0.282 ± 0.073 | 0.369 | 0.915 ± 0.063 | 0.844 | 0.867 | 0.500 | 0.147 | 5.769 |
| R0_raw_state | common_256 | 0.268 ± 0.208 | 0.391 | 0.941 ± 0.035 | 0.816 | 0.700 | 0.000 | 0.005 | 3.713 |
| R2_state_encoder | common_256 | 0.268 ± 0.157 | 0.106 | 0.929 ± 0.052 | 0.936 | 0.600 | 0.000 | 0.126 | 3.667 |
| R5_final_dit_hidden | native | 0.241 ± 0.136 | 0.101 | 0.919 ± 0.062 | 0.939 | 0.717 | 0.000 | 0.145 | 3.633 |
| R3_vlm_hidden | common_256 | 0.234 ± 0.122 | 0.053 | 0.944 ± 0.037 | 0.951 | 0.667 | 0.000 | 0.355 | 3.719 |
| R1_action_chunk | common_256 | 0.201 ± 0.066 | 0.221 | 0.920 ± 0.060 | 0.815 | 0.833 | 0.000 | 0.001 | 3.765 |
| R5_final_dit_hidden | common_256 | 0.199 ± 0.092 | 0.067 | 0.928 ± 0.053 | 0.935 | 0.817 | 0.000 | 0.145 | 3.558 |
| R6_vlm_dit_concat | native | 0.173 ± 0.074 | 0.017 | 0.942 ± 0.049 | 0.948 | 0.783 | 0.000 | 0.507 | 3.833 |
| R3b_vision_tokens | common_256 | 0.171 ± 0.072 | 0.053 | 0.949 ± 0.033 | 0.950 | 0.817 | 0.000 | 0.329 | 3.631 |
| R1_action_chunk | native | 0.135 ± 0.153 | 0.221 | 0.919 ± 0.063 | 0.815 | 0.917 | 0.500 | 0.001 | 5.215 |
| R0_raw_state | native | 0.052 ± 0.041 | 0.391 | 0.934 ± 0.050 | 0.816 | 0.917 | 0.000 | 0.005 | 5.096 |

## 해석

현재 최고 Boundary F1@±5는 **R4_action_encoder (common_256)**의 0.487다. 이 순위는 강제 monotonic 경로가 아닌 실제 첫 streaming 이벤트를 기준으로 한다.
상위 두 조합의 paired 차이는 Boundary F1@±5가 0.103 ± 0.196, Stage F1이 -0.018 ± 0.019다. 두 구간 모두 0을 포함하면 통계적으로 단일 승자를 확정하지 않는다.
실용 기본안은 단일 source이면서 feature/probe 지연이 더 작은 R4 action encoder common-256이다. R6는 R3+R5 fusion 보조 결과로 유지한다.

## 제한

- 무경계 실패 episode가 1개뿐이므로 실패 FPR의 분산이 크다. 고의 실패 rollout을 task별로 추가해야 한다.
- †기존 Linear 열은 과거 evaluator 결과로, 첫 예측 뒤의 추가 rising event를 FP로 벌점 주지 않았다. Transformer의 strict event F1과 직접 우열 비교하지 않는다.
- policy sample은 8 physics step 간격이어서 ±5 env-step 평가는 양자화 영향을 받는다.
- 이번 결과는 동일한 3개 LIBERO task 내 일반화이며 leave-one-task-out 결과가 아니다.

## 산출물

- per-seed runs: 80
- `results.csv`: seed별 원본 수치
- `summary.csv` / `summary.json`: 5-seed 평균과 95% CI
- `checkpoints/`: scaler/PCA, split IDs, select_layer, decoder 계약을 포함한 checkpoint
