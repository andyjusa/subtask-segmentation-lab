# 실행과 재현 범위

## CPU: clone만으로 가능한 것

README의 `uv sync --frozen --extra train` 후 실행합니다. HF/GPU/로봇/API가 필요 없습니다.

| 명령 | 입력 | 출력 / 검증 의미 |
|---|---|---|
| `uv run python scripts/reproduce.py verify` |55개 fixture 파일|SHA-256 확인|
| `uv run --extra train python scripts/reproduce.py pi05` |410×6 actual actions|Linear probe,accuracy,raw transitions|
| `uv run --extra train python scripts/reproduce.py incline50` |50episodes R3b|32train/8selection/10validation,Linear+MLP|
| `uv run python scripts/reproduce.py rollout` |549×2048 R3/R3b|native/PCA256,새 LR,같은 영상 OOF|

예: `--output outputs/r3 --hidden-key hidden_mean` 또는 `--output outputs/seed17 --seed 17`.
출력 json,확률/예측 npz,새 probe 가중치는 outputs에만 저장됩니다.
CPU BLAS 부하 제한이 필요하면 `OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4`를 명령 앞에 붙입니다.
incline 학습은 `--threads 4`가 기본입니다. 저장된 영상 없이도 전체 수치 평가가 됩니다.

## π0.5 순차 이벤트 replay

학습 후 아래 명령은 JSON만 출력합니다. HTTP 전송이나 하드웨어 제어는 하지 않습니다.

```bash
uv run vla-subtask-phase-probe replay \
  --actions fixtures/pi05/actions.npy --probe outputs/pi05/probe.npz \
  --phases picking,placing,complete \
  --task-id 00000000-0000-4000-8000-000000000001 \
  --subtask-ids 00000000-0000-4000-8000-000000000002,00000000-0000-4000-8000-000000000003,00000000-0000-4000-8000-000000000004
```

## GPU: 원본 GR00T hidden 다시 추출

아래는 **기존 Isaac-GR00T CUDA 환경이 준비된 사용자용**입니다. 이번 Mac CPU 재현 범위와 별개입니다.
외부 원본 dataset,카메라3개 및16D dual-arm embodiment와 일치하는 checkpoint가 필요합니다.
LIBERO 공식 checkpoint를 그대로 NEW_EMBODIMENT 입력에 넣을 수 있다는 뜻이 아닙니다.
가중치 접근권한 및 외부 패키지 버전은 해당 학습 환경에서 확인해야 합니다.

```bash
# 모두 자신의 실제 경로로 설정. 토큰은 코드/문서에 적지 않습니다.
LAB_ROOT="$PWD"
ISAAC_GROOT_ROOT=/path/to/Isaac-GR00T
INCLINE_DATA=/path/to/Incline_new_20260902_203009
GR00T_CHECKPOINT=/path/to/compatible/checkpoint-300
PYTHONPATH="$LAB_ROOT/packages/groot/src" \
uv run --project "$ISAAC_GROOT_ROOT" --no-sync python \
  "$LAB_ROOT/packages/groot/src/groot_subtask_phase_probe/incline_groot_hidden.py" \
  --isaac-root "$ISAAC_GROOT_ROOT" --dataset "$INCLINE_DATA" \
  --checkpoint "$GR00T_CHECKPOINT" --split "$LAB_ROOT/fixtures/incline50/split.json" \
  --sample-fps 5 --batch-size 4 --output-dir "$LAB_ROOT/outputs/new-hidden"
```

원본 camera 영상으로 경계 영상을 만들 때는 시스템 ffmpeg가 추가로 필요합니다.
`packages/groot/scripts/evaluate_rollout_incline_vision_hidden.py --help`의
`--hidden --boundaries --video --output-dir`를 명시하세요. 과거 default 경로는 현재 폴더에 없습니다.

## 보존한 코드와 별도 준비물

| 영역 | 코드 | 필요한 외부 자료 | 이번 재검증 |
|---|---|---|---|
| LIBERO R0~R6 | packages/groot/src/*collect*,*evaluate*,*hooks* |Isaac-GR00T,LIBERO,checkpoint,60episode features|unit tests; full GPU 재실행 아님|
| streaming Transformer | streaming_transformer.py |LIBERO per-episode arrays|archive 결과|
| AllTracker / TAPNext++ | packages/groot/scripts/extract_incline_* |각 외부 tracker repo/weights,원본영상|archive 결과|
| tracker Linear/MLP | train_tapnextpp_subtask_probe.py |tracker npz,split,boundaries; optional robot parquet|vision-only 경로만 CPU 재검증|
| hidden/tracker fusion | packages/groot/scripts/*fusion* |hidden,tracks,action/state|archive 결과|
| Backpack | archive/backpack/run_hidden_transformer_template.py |기존 causal Transformer checkpoint,features|template/결과 설명만|
| pause | packages/groot/src/*online_pause* |CUDA,simulator,compatible policy|원래 구현 보존; 실기 재검증 아님|

원본 패키지 README는 역사적 경로/환경을 포함합니다. **새 진입점은 루트 README와 이 문서**입니다.
모든 과거 GPU 실험이 clone 직후 실행된다고 주장하지 않습니다.

## 출처·권리·비공개

π0.5 코드 출처: andyjusa/vla-subtask-phase-probe, 당시 commit57b0b0398a3824c76748a5a15d6ab52113bf68fc.
GR00T 계측/분류 코드는 기존 로컬 groot-subtask-phase-probe의 first-party snapshot입니다.
외부 NVIDIA/LeRobot/tracker 모델 소스와 weights는 vendor하지 않았습니다.
dataset는 fixtures/manifest.json의 source를 확인하세요. 파생 특징도 비공개 연구자료로 취급합니다.
새 저장소에 임의 오픈소스 라이선스를 부여하지 않았으며 공개 재배포 시 각 권리관계를 검토해야 합니다.
