# GR00T에 붙일 때 무엇이 되고 무엇이 남았나

**저장 hidden의 5단계 추론은 실행 가능하지만, 그 분류기를 이용한 경사면 실시간 pause는 아직 연결되지 않았습니다.**
기존 LIBERO pause 구현을 보존한 것과 현재 MLP를 배포한 것은 다릅니다.

**기존 계산 재사용은 적용 목표이지 현재 실시간 검증 결과가 아닙니다.** 이번 특징은 별도 오프라인
backbone 실행에서 LoRA adapter를 끄고 action head 없이 추출했습니다. 동작 중인 policy에서
그 hidden을 그대로 읽었을 때의 정확도·추가 지연·메모리·action 동등성은 따로 검증해야 합니다.

| 항목 | 경사면 vision MLP | 기존 LIBERO boundary probe |
|---|---|---|
| 입력 | vision 평균 z, 2048차원 | 표현 z와 변화량 z−이전 z의 concat |
| 출력 | 다음 동작 stage0~4, softmax5개 | 첫 경계 여부, binary score |
| 저장물 | mlp.pt, scaler.npz, results.json | mean/scale/coefficient/intercept/threshold를 가진 기존 probe artifact |
| 의미 | 추1~4 수행 구간, 마지막 pointing | 첫 predicate 달성 시점 |
| 현재 활용 | 저장 특징 관찰용 추론·HTML | 호환 LIBERO policy에서 기존 pause 경로; 이번 실기 재검증 아님 |

`mlp.pt`를 binary probe 파일 자리에 넣으면 입력·출력·라벨 정의가 모두 다릅니다.
π0.5의 phase adapter도 별도 구현이며 경사면5단계와 자동 호환되지 않습니다.

## 관찰 전용 연결 시 지켜야 할 조건

- GR00T backbone 출력에서 학습 때와 같은 token mask와 vision 평균 방식을 사용합니다.
- 해당 모델·LoRA 상태·카메라·전처리·hidden 차원을 학습 입력과 맞춥니다.
  기존 offline 추출 조건을 맞춘다는 이유로 실행 중인 policy의 LoRA를 임의로 끄면 안 됩니다.
- 저장된 scaler와 동일 Linear/MLP만 적용해 단계·확률·관찰 timestamp를 기록합니다.
  관찰 hook은 rollout action 결과를 수정하지 않아야 합니다.
- 학습은5FPS지만 policy replan 간격은 다를 수 있습니다. 실제 관찰 시간축을 기록하고 지연을 다시 측정해야 합니다.
- 전체 영상 smoothing/DP는 미래를 보므로 온라인 트리거로 사용하지 않습니다.

## 추가로 결정·검증할 작업

1. **이벤트의 의미부터 고정:** gripper release, 안정적 적재 성공, 다음 동작 시작 중 무엇을 알릴 것인가?
   현재 약한 release 라벨은 성공을 보장하지 않습니다. pointing 실제 시작도 별도 정의가 필요합니다.
2. 과거 관측만 이용하는 안정화·재시도·중복 이벤트 방지 규칙을 구현하고 독립 rollout으로 평가합니다.
3. mock adapter로 이벤트 전달을 확인한 뒤, 별도 승인된 simulator/hardware 실험에서 pause/resume에 연결합니다.
   실제 환경은 계속 변하므로 resume 시 현재 관측으로 재추론해야 합니다.
4. event F1, 실패 FPR, 경계 지연, 계측 전후 action 동등성을 측정합니다.

이 문서는 연결 조건과 미구현 범위를 설명합니다. 새로운 서버 API나 안전 제어 계약이 확정됐다는 뜻은 아닙니다.
