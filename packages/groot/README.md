# GR00T N1.7 subtask phase probe

영상·지시문 token 위치 분리, 단일/융합 Linear·MLP 비교, 색 변화 기반 one-shot stop의
실시간 검증 결과는 [`reports/COLOR_STOP_MODALITY_REPORT.md`](reports/COLOR_STOP_MODALITY_REPORT.md)에
정리되어 있다.

색 변화 Stop에 대한 18개 소형 causal Transformer 후보와 5-seed 검증은
[`reports/COLOR_STOP_TRANSFORMER_CAMPAIGN.md`](reports/COLOR_STOP_TRANSFORMER_CAMPAIGN.md)에
정리되어 있다.

`nvidia/GR00T-N1.7-LIBERO`의 frozen representation으로 LIBERO 장기 조작의
subtask stage와 전환 경계를 선형 분류하는 실험 패키지다. Probe는 관찰만 하며
기본 수집·평가에서는 GR00T가 생성한 action을 수정하지 않는다. 선택적으로
Online Pause를 켜면 경계에서 현재 action chunk를 폐기하고 일정 시간 정지한 뒤
같은 최신 observation으로 다시 추론한다.

비교 표현은 raw robot state, predicted continuous action chunk, state encoder,
VLM hidden, 마지막 denoise iteration의 action encoder와 final DiT hidden,
그리고 VLM+DiT fusion이다. Final DiT hidden은 discrete action token이 아니라
`action-token-like continuous latent`다.

## 원격 실행

공식 Isaac-GR00T 저장소와 LIBERO 환경이 `/root/projects/Isaac-GR00T`에 준비된
상태를 전제로 한다.

```bash
cd /root/projects/groot-subtask-phase-probe
uv sync --extra dev

ISAAC_GROOT_ROOT=/root/projects/Isaac-GR00T \
  bash scripts/collect_smoke.sh

uv run groot-probe-evaluate artifacts --output-dir reports
```

수집 결과는 `artifacts/<task>/<episode>/` 아래 `rollout.mp4`, `features.npz`,
`metadata.json`으로 저장된다. 평가 결과는 `results.csv`와 한국어 `REPORT.md`다.

## 전체 실험

```bash
ISAAC_GROOT_ROOT=/root/projects/Isaac-GR00T \
PYTHONPATH=src:/root/projects/Isaac-GR00T \
/root/projects/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \
  -m groot_subtask_phase_probe.collect \
  --tasks all --episodes 20 --seed 2000 --output-dir artifacts_full

uv run groot-probe-evaluate artifacts_full --output-dir reports
uv run python scripts/validate_artifacts.py artifacts_full
```

이번 실행은 3개 task에서 총 60 episode를 수집했고 59개가 성공했다. 동일 seed
15개를 hook 없는 baseline과 비교했을 때 성공 및 종료 step 불일치는 모두 0개였다.
선정 규칙상 최종 표현은 8차원 `R0_raw_state`다. Boundary F1은 ±5 step에서
0.391, ±10 step에서 0.683이었고 stage macro F1은 0.816이었다. R4 action
encoder는 Boundary F1 0.369로 근접했지만 1536차원이라 R0보다 복잡하다.

전체 R0~R6 동시 snapshot은 평균 1.281ms/step으로 모델 추론 평균
223.196ms/step의 약 0.57%다. 자세한 결과는 `reports/REPORT.md`, seed별 원본
수치는 `reports/results.csv`에 있다. 60개 원본 영상과 feature artifact는 용량상
저장소에서 제외하고 원격 `/root/projects/groot-subtask-phase-probe/artifacts_full`에
보관한다.

## Online Pause/Resume

먼저 episode 단위 train/validation split으로 학습한 native boundary probe를
직렬화한다. 기본 표현은 선정 결과인 8차원 `R0_raw_state`다.

```bash
uv run groot-probe-export-online artifacts_full \
  --output reports/online_boundary_probe.npz \
  --representation R0_raw_state --seed 17
```

수집 명령에 probe를 전달하면 첫 boundary rising event에서 실제 제어 루프가
pause 상태로 전환된다. LIBERO 기본 제어 주기인 20Hz를 사용한다.

```bash
ISAAC_GROOT_ROOT=/root/projects/Isaac-GR00T \
PYTHONPATH=src:/root/projects/Isaac-GR00T \
/root/projects/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python \
  -m groot_subtask_phase_probe.collect \
  --tasks stove_moka --episodes 1 --seed 1001 \
  --output-dir artifacts_online_pause \
  --online-probe reports/online_boundary_probe.npz \
  --pause-seconds 10 --pause-control-hz 20
```

동작 순서는 다음과 같다.

1. GR00T 추론 직후 feature로 boundary 확률을 계산한다.
2. 첫 rising event이면 방금 생성한 action chunk를 실행하지 않고 폐기한다.
3. pause 동안 GR00T `get_action()`은 호출하지 않는다.
4. 대신 이동 6축은 모두 0인 hold action을 `env.step()`으로 계속 실행한다. 그리퍼는
   마지막으로 실제 실행된 명령만 유지하므로 물리와 카메라는 20Hz로 계속 진행된다.
5. 10초 pause는 200 physics step이며, 8-step wrapper에서는 25번의 hold macro-step이다.
6. pause가 끝나면 policy를 reset하고, pause 이후의 최신 observation에서 새 action
   chunk를 생성한다.
7. 동일 episode에서는 한 번만 pause한다.

실제 CUDA/LIBERO smoke에서는 `env_step=80`에 pause가 한 번 발생했고, pause 동안
정확히 200 physics step과 25 macro-step을 실행하면서 GR00T 추론 호출은 0회였다.
정지를 유발한 action chunk 1개를 폐기한 뒤 rollout도 성공했다. 상세 수치는
`reports/online_pause_smoke.json`에 있다.

## Boundary pause 영상

아래 명령은 공통 test episode를 고르고 각 native representation의 첫 boundary
예측 프레임에서 영상을 10초 정지시킨다. 화면에는 `PRED`와 실제 LIBERO predicate
경계인 `GT`가 함께 표시된다. 이 기능은 비교를 위한 replay 시각화이며 rollout
action에는 개입하지 않는다.

```bash
uv run python scripts/render_pause_videos.py artifacts_full \
  --output-dir pause_videos --seed 17 --pause-seconds 10
```

이번 비교 episode의 실제 경계는 90 step이었다. R1 action chunk는 88 step,
R3/R3b/R4는 80 step, R0/R2/R5/R6는 72 step에서 pause됐다.

## Causal sliding-window Transformer

Linear probe와 같은 episode split, seeds(17/29/41/53/67), R0~R6 표현 및
train-only StandardScaler/PCA 계약을 그대로 사용한다. 차이는 현재 시점 하나 대신
최근 16개 policy sample을 2-layer causal Transformer Encoder에 넣는다는 점이다.
각 출력은 현재와 과거만 볼 수 있고, 전체 episode를 다시 훑는 Viterbi나 사후
backtracking은 사용하지 않는다.

경계 평가는 threshold를 넘는 첫 rising event가 실제로 내보낸 전환이라고 가정한다.
첫 이벤트가 너무 이르면 이후 이벤트로 정답을 복구하지 않으며, 추가 이벤트도 모두
false positive로 센다. 따라서 `Boundary F1@±5/±10`, 경계 오차 중앙값,
FP/episode, 무경계 실패 episode FPR은 실제 streaming 동작을 반영한다.

```bash
PYTHONPATH=src /root/projects/Isaac-GR00T/.venv/bin/python \
  -m groot_subtask_phase_probe.streaming_transformer artifacts_full \
  --output-dir reports/transformer_streaming \
  --linear-results reports_full/results.csv
```

일부 조합만 먼저 실행할 수도 있다. 실행별 JSON이 저장되므로 같은 명령을 다시
실행하면 완료된 조합은 건너뛴다.

```bash
PYTHONPATH=src /root/projects/Isaac-GR00T/.venv/bin/python \
  -m groot_subtask_phase_probe.streaming_transformer artifacts_full \
  --output-dir reports/transformer_streaming_smoke \
  --representations R0_raw_state --projections native --seeds 17 \
  --epochs 2 --patience 2
```

산출물은 seed별 `results.csv`, 5-seed 평균과 95% CI인 `summary.csv`, 재현 설정,
scaler/PCA와 split ID까지 포함한 checkpoint, 한국어 `REPORT.md`다. R3/R3b의
backbone layer는 실제 수집 설정인 `select_layer=16`으로 기록한다.

실제 Windows CUDA 실행은 60 episodes × 8 representations × 2 projections ×
5 seeds, 총 80개 조합을 완료했다. 평균 Boundary F1@±5 최고는 R4 action encoder
common-256의 0.487 ± 0.184였고 R6 fusion common-256은 0.384 ± 0.154였다.
paired 차이의 95% CI가 0을 포함하므로 단일 승자를 확정하지 않으며, 단일 source와
낮은 추가 지연을 고려한 실용 기본안은 R4 common-256이다. 전체 수치와 상위 두
조합의 checkpoint는 `reports/transformer_streaming/`에 보관한다.
