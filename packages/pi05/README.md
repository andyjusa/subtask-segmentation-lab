# VLA Sub-task Phase Probe

VLA 정책이 출력한 action만 사용해 로봇 작업의 순차 단계를 온라인으로 구분한
실험을 독립적으로 정리한 저장소다. 원 실험은 π0.5의 6축 commanded action에서
`picking → placing → complete`를 분류하고, 안정화된 전환을 Task API로 전달했다.

## 포함 범위

- action 정규화 + 3-class 선형 phase probe
- 다음 단계만 허용하는 순차 상태 전환
- confidence 임계값과 연속 프레임 안정화
- 제어 루프를 막지 않는 Task API 직렬 전송
- action trace로 probe를 학습하고 다시 재생하는 CLI
- 원 실험의 수치 결과와 한계
- continuous action, action-expert hidden, VLM hidden, 사후 FAST token 비교 도구

정책 체크포인트, probe 가중치, action trace, 데이터셋, 영상과 캐시는 포함하지
않는다. 필요한 파일은 사용자가 로컬에서 직접 지정한다.

## 구조

```text
src/vla_subtask_phase_probe/  핵심 detector, 학습, API 전송
tests/                        합성 action 기반 단위 테스트
examples/pick_place.json      원 실험의 phase와 Task UUID 예시
reports/experiment.md         실험 방법·결과·한계
results/summary.json          대용량 산출물을 제외한 결과 요약
results/representation-comparison.json  내부 표현 비교 결과
```

## 설치

```bash
uv sync --extra dev
```

probe를 새로 학습할 때만 PyTorch를 추가한다.

```bash
uv sync --extra dev --extra train
```

## 학습

```bash
uv run vla-subtask-phase-probe train \
  --actions /path/to/rollout.actions.npy \
  --transitions 170,354 \
  --phases picking,placing,complete \
  --output artifacts/action-phase-probe.npz
```

이 명령은 `.npz` 가중치와 같은 이름의 `.json` 측정 결과를 만든다.
`artifacts/`는 Git에서 제외된다.

## 오프라인 재생

```bash
uv run vla-subtask-phase-probe replay \
  --actions /path/to/rollout.actions.npy \
  --probe artifacts/action-phase-probe.npz \
  --phases picking,placing,complete \
  --task-id d4c55c46-db1b-5cb4-a35b-0c06bbd59be7 \
  --subtask-ids 58028ae0-dac6-5cad-ad4e-c9b98a49b46e,9c159a93-634f-5460-930b-48e9e50b9af1,d4c55c46-db1b-5cb4-a35b-0c06bbd59be7 \
  --confidence 0.5 \
  --stable-frames 5
```

## 해석 시 주의

이 probe가 판정하는 것은 정책이 **의도한 동작 단계**다. 실제 grasp나 placement
성공을 보장하지 않는다. 원 실험에서도 실패 rollout이 `placing`으로 전환된 사례가
있었다. 실제 시스템의 `complete`는 gripper, force 또는 vision verifier로 별도
확인해야 한다.

## π0.5 내부 표현 비교

π0.5에는 이산 action token이 없다. 이 저장소에서는 마지막 denoising step의
action-expert hidden slot을 같은 모델 안의 token 유사 표현으로 사용한다. 별도로
FAST tokenizer를 사후 적용한 이산 token histogram도 비교하지만, 이는 π0.5가 직접
생성한 token이 아니다. 측정 방법과 실패 rollout 결과는
[`reports/experiment.md`](reports/experiment.md)에 정리되어 있다.
