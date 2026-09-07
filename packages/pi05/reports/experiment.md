# π0.5 action-only 서브태스크 분할 실험

## 목적

로봇 물리 상태나 영상을 직접 읽지 않고, π0.5가 매 제어 프레임에 출력하는 6축
commanded action만으로 `집기 → 놓기 → 종료`의 설명 시점을 찾는다.

## 고정한 구조

```text
π0.5 action(6)
  → rollout 평균·표준편차 정규화
  → Linear(6, 3) + softmax
  → 현재 단계의 바로 다음 단계만 후보로 허용
  → confidence ≥ 0.5가 5프레임 연속일 때 전환
  → Task API 이벤트를 worker queue에서 순차 POST
```

시작 단계는 즉시 한 번 전송하고, 마지막 단계의 `task_sub_id`에는 `task_id`를
사용해 전체 작업 종료를 나타냈다. 네트워크 요청은 제어 루프와 분리했으며, 서버가
받은 뒤 응답만 유실된 상황에서 중복 실행을 피하기 위해 자동 재시도하지 않았다.

## 원 실험 결과

| 항목 | 결과 |
|---|---:|
| rollout 프레임 | 410 |
| action 차원 | 6 |
| 물리 상태 기준 경계 | 170, 354 |
| 선형 probe raw 경계 | 170, 350 |
| 단일 rollout stride validation accuracy | 98.78% |
| 5프레임 안정화 포함 실제 이벤트 | 175, 355 |
| Task API 전달 | 3/3 |
| 성공 seed의 최종 placement | 성공 |

98.78%는 하나의 성공 rollout을 stride로 나눈 값이다. 서로 다른 초기 위치나
실패 rollout에 대한 일반화 성능으로 해석할 수 없다.

## 확인된 한계

- action은 정책의 **의도**를 나타내므로 물리적 성공과 다를 수 있다.
- 실패 seed에서 물체를 집지 못했는데 `placing`이 발생했다.
- 한 rollout의 프레임을 나눈 validation은 시간적으로 강하게 상관되어 있다.
- 현재 선형 probe는 직전 action history, joint state, gripper state와 영상을 쓰지 않는다.

## 다음 실험

1. rollout 단위 train/validation/test 분리
2. transition latency와 boundary F1 측정
3. false `placing`, false `complete`, 중복 API 전송률 기록
4. action window + gripper width + joint velocity를 쓰는 작은 temporal baseline 비교
5. `complete`에 vision 또는 물체 상태 verifier 추가

## 내부 표현 비교 실험

같은 정책·작업·경계 라벨을 유지하고 phase probe의 입력만 네 가지로 바꿨다.

| 표현 | 차원 | 성공 seed validation | 성공 seed 안정화 경계 | 실패 seed false `placing` |
|---|---:|---:|---|---:|
| 최종 continuous action | 6 | 98.78% | 174, 354 | 197 |
| action-expert hidden slot | 1,024 | 100.00% | 174, 358 | 193 |
| VLM prefix hidden mean | 2,048 | 93.90% | 154, 354 | 254 |
| 사후 FAST token histogram | 2,049 | 93.90% | 154, 354 | 354 |

성공 seed의 물리 기준 경계는 170, 354였고 실패 seed에는 물리 전환이 없었다.
표의 validation은 기존 실험과 동일하게 하나의 rollout 내부 stride split을 사용했기
때문에 일반화 성능이 아니라 표현 분리도만 나타낸다.

### action token의 정확한 의미

π0.5는 autoregressive discrete action token 모델이 아니다. 이미지·언어 prefix와
flow-matching action expert를 결합하고, 연속 action chunk를 반복 denoising한다.
따라서 1,024차원 결과는 마지막 denoising step의 action-expert hidden slot이며,
π0.5 내부에서 action token에 가장 가까운 표현이지만 이산 token ID는 아니다.

별도 비교의 FAST token은 π0.5가 출력한 action을 q01/q99로 `[-1, 1]` 정규화한
뒤 LeRobot FAST tokenizer로 사후 변환했다. 각 계획 chunk의 token histogram을
사용했으며 π0.5 모델이 직접 생성한 token은 아니다. 진짜 autoregressive action
token 모델을 비교하려면 같은 task로 학습된 π0-FAST checkpoint가 필요하다.

### 해석

- action-expert hidden은 성공 rollout을 가장 잘 분리했지만 3,075개의 probe
  parameter를 사용해 21개인 action 기준선보다 과적합 위험이 크다.
- VLM hidden과 FAST token은 replan 50 설정에서 rollout당 서로 다른 벡터가 9개뿐이다.
  따라서 170프레임 경계를 계획 시작인 150프레임 부근으로 당겨 판단했다.
- 실패 rollout에서는 네 방식 모두 false `placing`을 냈다. 표현을 hidden/token으로
  바꾸는 것만으로 실제 grasp 성공 여부를 알 수 없다는 직접적인 결과다.
- 현재 목적이 설명 timing이면 6차원 action probe가 가장 작고 충분하다. `complete`
  또는 성공 발화는 vision·gripper·contact verifier가 승인하도록 분리해야 한다.

## 제외한 산출물

이 저장소에는 정책 체크포인트, probe `.npz`, action `.npy`, 영상, 데이터셋과
로컬 캐시가 없다. 수치 재현에는 별도의 원 rollout과 정책 가중치가 필요하다.
