# GR00T N1.7 영상·지시문 분리 및 색 변화 Stop 실험

## 결론

색 변화처럼 **영상에만 정답 근거가 있는 경계 검출**에는 GR00T VLM의
`R3b_vision_tokens`가 가장 적합했다. 5개 고정 seed 평균 Boundary F1@±5는
`0.671 ± 0.375`였고, 다음 후보인 vision+action output MLP의 `0.499 ± 0.238`보다
높았다. 여러 표현을 단순 결합한 구조는 vision 단독보다 좋아지지 않았다.

선정한 `R3b → train-only PCA 256 → MLP(128, 64)` probe를 seed 17로 다시 학습하고,
검증 split에서 stop 규칙을 고정한 뒤 홀드아웃 split과 실시간 재추론으로 검증했다.

- 선택 임계값: `0.75`
- 연속 확인 프레임: `1`
- 홀드아웃 8 episode Boundary F1@±5: `1.000`
- 홀드아웃 색 미변화 episode 오정지율: `0.000`
- 라이브 색 변화: 정답 step 8, stop step 8, 오차 0 frame
- stop 이후 GR00T 추론 호출: 0회
- 라이브 색 미변화: 24 frame 동안 stop 없음

단, 전체 데이터가 20개 counterfactual pair로 작고 5-seed 평균의 CI가 크므로,
홀드아웃 1.0은 일반화 성능의 확정값이 아니라 **구현 및 인과적 동작 검증 결과**로
해석해야 한다.

## 실험 설계

실제 LIBERO 카메라 프레임 위에 비커와 교반봉을 합성하고 아래 지시를 고정했다.

> Keep stirring and stop immediately when the liquid changes color.

20개 pair마다 `change`와 `no_change` episode를 만들었다. 두 쌍은 지시문, robot state,
색 변화 예정 시점, 비커 위치, 교반 운동이 같고 **영상 속 액체 색 변화만 다르다**.
pair 단위로 60/20/20을 나누어 같은 pair의 두 영상이 split 사이에 섞이지 않게 했다.

이 설계 때문에 action/state만으로 정답을 맞히는 시간 shortcut을 줄일 수 있다. 다만
R1 action chunk 역시 GR00T가 영상을 보고 생성한 출력이므로 완전한 non-visual 입력은
아니다. 영상 의존성의 핵심 증거는 동일 조건의 change/no-change pair 비교다.

## 비교 표현

| 표현 | GR00T에서 가져온 위치 | 의미 |
|---|---|---|
| R1 action chunk | 최종 연속 action 출력 | 다음 horizon에 실행할 관절·gripper 명령 묶음 |
| R3 VLM hidden | VLM 전체 유효 token 평균 | 영상과 지시문이 섞인 multimodal context |
| R3b vision-position | VLM 출력 중 image mask 위치 평균 | 영상 token 위치의 사후 multimodal hidden |
| R3c instruction-position | VLM 출력 중 non-image 유효 위치 평균 | 지시문 token 위치의 사후 multimodal hidden |
| R5 final DiT hidden | 마지막 denoise iteration의 action horizon 평균 | action-token-like latent이며 discrete action token은 아님 |
| F1 | R3b + R3c | vision-position + instruction-position |
| F2 | R3b + R1 | vision-position + action output |
| F3 | R3c + R1 | instruction-position + action output |
| F4 | R3b + R3c + R1 | vision-position + instruction-position + action output |
| F5 | R3b + R5 | vision-position + final DiT hidden |
| F6 | R3c + R5 | instruction-position + final DiT hidden |
| F7 | R3b + R3c + R5 | vision-position + instruction-position + final DiT hidden |

R3b와 R3c는 입력 token 종류로 위치를 나눈 것이지 서로 독립적인 단일 modality 표현은
아니다. VLM self-attention 이후 값이라 R3c에도 영상 정보가 섞일 수 있다.

## 색 변화 경계 결과

아래는 5개 seed 평균이며 MLP는 동일한 10-epoch 예산, common-256은 train split으로만
학습한 PCA를 사용했다.

| 순위 | 표현 | Probe | Boundary F1@±5 | 95% CI | no-change FPR | Probe 지연 |
|---:|---|---|---:|---:|---:|---:|
| 1 | R3b vision-position | MLP common-256 | 0.671 | ±0.375 | 0.150 | 2.411 ms |
| 2 | F2 vision + action output | MLP common-256 | 0.499 | ±0.238 | 0.400 | 2.872 ms |
| 3 | F1 vision + instruction | MLP common-256 | 0.485 | ±0.233 | 0.350 | 1.769 ms |
| 4 | R3c instruction-position | MLP common-256 | 0.483 | ±0.178 | 0.200 | 2.653 ms |
| 5 | F3 instruction + action output | MLP common-256 | 0.447 | ±0.172 | 0.400 | 2.119 ms |
| 6 | R3 joint VLM hidden | MLP common-256 | 0.393 | ±0.215 | 0.100 | 1.922 ms |
| 7 | R1 action chunk | Linear native | 0.382 | ±0.167 | 0.850 | 0.143 ms |

R1은 빠르지만 색이 변하지 않은 episode의 85%에서 한 번 이상 잘못 멈췄다. 반면 R3b는
성능과 오정지율 모두 더 좋았다. Fusion은 불필요한 축과 작은 표본에서의 PCA 분산을
추가해 오히려 일반화를 해친 것으로 보인다.

Stage F1이 약 0.4로 낮은 것은 의도된 난이도다. 초기색과 변화색을 모두 무작위화해 같은
색이 다른 pair에서는 stage 0과 stage 1에 모두 나타날 수 있다. 이 task의 목표는 현재
색으로 장기 stage를 분류하는 것이 아니라 `z_t - z_(t-1)`에서 **변화 순간**을 찾는 것이다.

## 일반 LIBERO subtask와의 차이

기존 3개 LIBERO task, 60 episode에서는 instruction-position Linear native가 Stage F1
`0.976 ± 0.011`로 가장 높았다. 그러나 정확한 Boundary F1@±5는 `0.273 ± 0.163`에
그쳤다. 즉 “현재 어느 단계인가”는 잘 분류해도 “최초 predicate 전환의 정확한 한
프레임”을 검출하는 문제는 훨씬 어렵다.

이 결과는 표현을 task 성격에 따라 골라야 함을 보여준다.

- 언어적으로 구분되는 일반 subtask stage: instruction-position이 강함
- 시각적 순간 변화 기반 stop: vision-position + temporal delta가 강함

## 실제 Stop 실행 방식

실시간 경로는 매 프레임 다음 순서로 동작한다.

1. GR00T N1.7을 한 번 추론한다.
2. hook은 출력에 개입하지 않고 R3b hidden만 복사한다.
3. train-only scaler와 PCA로 256차원으로 변환한다.
4. `[z_t, z_t - z_(t-1)]`를 MLP에 넣는다.
5. 검증에서 정한 threshold와 확인 횟수를 만족하면 stop을 한 번만 발생시킨다.
6. 해당 시점부터 action/inference를 중단하고 카메라 시간은 계속 흐르게 한다.

데모에서는 stop 뒤 교반봉이 멈추지만 4.8초 영상은 끝까지 계속 기록된다. 이는
“시뮬레이터 전체 시간을 정지”한 것이 아니라, 현실에서 로봇 명령과 VLA 추론만 멈추고
외부 세계와 카메라는 계속 진행되는 동작을 모사한 것이다.

현재 색 변화 환경은 LIBERO 배경 위 합성 비커를 사용한 시각 benchmark다. 실제 물리
액체나 로봇 action rollout은 아니므로, 다음 단계에서는 실제 camera stream과 hold/zero
action adapter에 같은 one-shot gate를 연결해야 한다.

## 재현 산출물

- 전체 일반 task 표: `remote_results/resumed_20260827/reports/modalities_full/REPORT.md`
- 전체 색 변화 표: `remote_results/resumed_20260827/reports/color_stop/REPORT.md`
- 고정 seed 학습·split: `remote_results/resumed_20260827/reports/color_stop_online/offline_report.json`
- 라이브 결과: `remote_results/resumed_20260827/reports/color_stop_online/live_demo_summary.json`
- 재사용 probe: `remote_results/resumed_20260827/reports/color_stop_online/color_stop_online_probe.npz`
- 색 변화 stop 영상: `remote_results/resumed_20260827/reports/color_stop_online/color_stop_online_change.mp4`
- 색 미변화 대조 영상: `remote_results/resumed_20260827/reports/color_stop_online/color_stop_online_no_change.mp4`

