# GR00T N1.7 서브태스크 분리 실험 보고서

- 수집 episode: 60개
- rollout 성공: 59/60
- split: episode 단위 60/20/20, 공통 seed 5개
- 모델: frozen GR00T N1.7 + L2 logistic linear probe

## 표현별 평균 결과

| 표현 | 투영 | Boundary F1 ±5 (95% CI) | Stage macro F1 (95% CI) | 실패 FPR | 차원 | Probe ms |
|---|---:|---:|---:|---:|---:|---:|
| R0_raw_state | native | 0.391 ± 0.152 | 0.816 ± 0.041 | 1.000 | 8 | 0.000175 |
| R0_raw_state | common_256 | 0.391 ± 0.152 | 0.816 ± 0.041 | 1.000 | 256 | 0.000744 |
| R4_action_encoder | native | 0.369 ± 0.166 | 0.844 ± 0.041 | 1.000 | 1536 | 0.003287 |
| R4_action_encoder | common_256 | 0.369 ± 0.166 | 0.840 ± 0.040 | 1.000 | 256 | 0.000723 |
| R1_action_chunk | native | 0.221 ± 0.115 | 0.815 ± 0.038 | 1.000 | 112 | 0.000911 |
| R1_action_chunk | common_256 | 0.221 ± 0.115 | 0.815 ± 0.038 | 1.000 | 256 | 0.000653 |
| R3b_vision_tokens | native | 0.154 ± 0.095 | 0.951 ± 0.045 | 0.500 | 2048 | 0.003396 |
| R3_vlm_hidden | native | 0.120 ± 0.113 | 0.953 ± 0.043 | 0.500 | 2048 | 0.003416 |
| R2_state_encoder | common_256 | 0.106 ± 0.065 | 0.936 ± 0.047 | 1.000 | 256 | 0.000670 |
| R2_state_encoder | native | 0.106 ± 0.065 | 0.936 ± 0.047 | 1.000 | 1536 | 0.002295 |
| R5_final_dit_hidden | native | 0.101 ± 0.095 | 0.939 ± 0.055 | 1.000 | 1024 | 0.002972 |
| R5_final_dit_hidden | common_256 | 0.067 ± 0.061 | 0.935 ± 0.057 | 1.000 | 256 | 0.000659 |
| R3_vlm_hidden | common_256 | 0.053 ± 0.071 | 0.951 ± 0.045 | 0.500 | 256 | 0.000634 |
| R3b_vision_tokens | common_256 | 0.053 ± 0.071 | 0.950 ± 0.047 | 0.500 | 256 | 0.000648 |
| R6_vlm_dit_concat | common_256 | 0.051 ± 0.041 | 0.948 ± 0.050 | 1.000 | 256 | 0.000669 |
| R6_vlm_dit_concat | native | 0.017 ± 0.034 | 0.948 ± 0.051 | 1.000 | 3072 | 0.005696 |

## 경계 보조 지표

| 표현 | 투영 | Boundary F1 ±10 | 경계 오차 중앙값(step) | 예측 경계 수/episode |
|---|---:|---:|---:|---:|
| R0_raw_state | native | 0.683 | 7.0 | 1.133 |
| R0_raw_state | common_256 | 0.683 | 7.0 | 1.133 |
| R4_action_encoder | native | 0.656 | 7.5 | 1.333 |
| R4_action_encoder | common_256 | 0.656 | 7.5 | 1.333 |
| R1_action_chunk | native | 0.428 | 7.9 | 0.883 |
| R1_action_chunk | common_256 | 0.428 | 7.9 | 0.883 |
| R3b_vision_tokens | native | 0.410 | 13.4 | 1.150 |
| R3_vlm_hidden | native | 0.344 | 14.2 | 1.133 |
| R2_state_encoder | common_256 | 0.369 | 11.7 | 1.100 |
| R2_state_encoder | native | 0.369 | 11.7 | 1.100 |
| R5_final_dit_hidden | native | 0.356 | 13.0 | 1.250 |
| R5_final_dit_hidden | common_256 | 0.375 | 12.9 | 1.217 |
| R3_vlm_hidden | common_256 | 0.294 | 15.0 | 1.233 |
| R3b_vision_tokens | common_256 | 0.224 | 16.6 | 1.300 |
| R6_vlm_dit_concat | common_256 | 0.152 | 16.3 | 1.300 |
| R6_vlm_dit_concat | native | 0.186 | 16.4 | 1.367 |

## 선정

규칙상 1위는 **R0_raw_state (native)**다. 공통 seed별 paired 차이의 95% CI로 Boundary F1 ±5와 Stage macro F1의 통계적 동률을 판단한 뒤, 실패 episode FPR, 차원, probe 지연 순으로 더 단순한 표현을 선택했다.

전체 R0~R6 동시 계측 hook 지연은 step당 평균 0.0344 ms다.

## 계측 전후 rollout 불변성

동일 seed paired 15개에서 baseline과 계측 성공은 각각 15/15, 15/15였다. 성공 불일치 0개, 종료 step 불일치 0개였다.

## 계측 비용

- 표본: policy step 30개(첫 호출 포함)
- 모델 추론: 평균 223.196 ms/step
- 전체 feature snapshot: 평균 1.281 ms/step
- forward hook 자체: 평균 0.0330 ms/step
- peak VRAM: 5.949 GiB
- episode 내 peak host RSS 증가: 0.000 MiB

| 표현 | 평균 추출 ms | P95 ms |
|---|---:|---:|
| R0_raw_state | 0.0051 | 0.0045 |
| R1_action_chunk | 0.0007 | 0.0009 |
| R2_state_encoder | 0.1262 | 0.2166 |
| R3_vlm_hidden | 0.3551 | 0.7903 |
| R4_action_encoder | 0.1467 | 0.2458 |
| R5_final_dit_hidden | 0.1449 | 0.2667 |
| R6_vlm_dit_concat | 0.0068 | 0.0083 |
| R3b_vision_tokens | 0.3288 | 0.6488 |

R5는 discrete action token이 아니라 마지막 denoise iteration의 action-token-like continuous latent로 해석해야 한다.
