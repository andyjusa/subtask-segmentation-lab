# 재현과 디버깅

## 실행 계약

이 경로는 **저장된 실제 GR00T 특징 → 분류기 학습 → checkpoint 재로드 → 시점별 추론**입니다.
로봇·HTTP API·원본 GR00T 추론 서버는 호출하지 않습니다. 기본 MLP/Linear는 CPU에서 실행됩니다.

입력 NPZ는 `vision_mean` 또는 `hidden_mean`의 유한한 float `[N,2048]`,
엄격히 증가하는 정수 `sample_frames[N]`을 가져야 합니다. 모델 폴더에는
같은 학습에서 생성한 `results.json`, `scaler.npz`, `mlp.pt`/`linear.pt`가 필요합니다.
`hidden_key`는 results.json에서 자동으로 읽으며 다른 특징으로 임의 대체하지 않습니다.
체크포인트는 직접 학습했거나 신뢰하는 파일만 사용하세요.

## clone 후 첫 실행

```bash
uv sync --frozen --extra train
uv run --frozen --extra train python scripts/doctor.py
uv run --frozen --extra train pytest -q
uv run --frozen --extra train python scripts/reproduce.py incline50
uv run --frozen --extra train python scripts/infer.py \
  --model-dir outputs/incline50 \
  --features fixtures/incline50/validation/episode_000/groot_backbone_hidden.npz \
  --output outputs/episode_000.jsonl
```

doctor는 Python·패키지 버전과 fixture 55개 체크섬을 확인합니다. 토큰이나 환경변수 값은 출력하지 않습니다.
pytest의 `test_train_save_load_predict`는 임시 경로에서 **2 epoch 실제 학습** 후 Linear/MLP를 각각
로드하여 확률 유효성·재추론 일치를 검사합니다. 이 smoke 점수는 본 실험 결과가 아닙니다.

추론 표준 출력에는 입력 shape/dtype, clip 비율, 모델·특징 SHA-256, raw 단계 전환 목록이 나옵니다.
JSONL에는 모든 샘플의 `source_frame`, `stage_id`, `stage`, 5개 `probabilities`가 남습니다.
전체 영상을 사용하는 ordered decoding은 적용하지 않습니다. raw 예측이 앞뒤 단계로 튈 수 있습니다.
source_frame은 원본 frame 인덱스입니다. 특징의 5 FPS로 나눠 초로 바꾸지 마세요.
시연의 5단계 정의는 성공한 추 개수 0~4인 rollout 라벨과 다릅니다.

## breakpoint로 확인

### 그래프로 보기

```bash
uv run --frozen python scripts/visualize.py \
  --predictions outputs/episode_000.jsonl --output outputs/episode_000.html \
  --boundaries fixtures/incline50/boundaries.csv --episode 0 --source-fps 30
```

생성된 HTML을 브라우저에서 열면 raw 단계·참조 라벨 막대와 5개 확률 곡선이 보입니다.
외부 서버나 JavaScript 없이 파일 자체로 열립니다. `--source-fps`는 원본 영상 30 FPS이며
특징 샘플링의 5 FPS가 아닙니다. 참조 경계가 없으면 마지막 세 옵션을 모두 생략하세요.
원본 동영상은 저장소에 포함하지 않으며, 기존 tracker/SAM3 영상은 Notion 실험 페이지에 따로 제공합니다.

### 디버거로 보기

별도 debugger 패키지 설치 없이 표준 `pdb`로 실행할 수 있습니다.

```bash
uv run --frozen --extra train python -m pdb scripts/infer.py \
  --model-dir outputs/incline50 \
  --features fixtures/incline50/validation/episode_000/groot_backbone_hidden.npz \
  --output outputs/debug-episode_000.jsonl --traceback
```

`b prepare_features`, `c`로 전처리 진입을 멈추고 `n`, `p x.shape` 등으로 값을 확인합니다.
VS Code는 Python/Python Debugger 확장이 이미 있으면 `.venv` 인터프리터를 선택하고
포함된 launch 설정의 train smoke 또는 infer를 실행합니다. 확장 설치는 자동으로 하지 않습니다.
IDE UI 자체는 검증 범위 밖이며 CLI·pdb 경로를 기준으로 합니다.

확인 위치:

- `scripts/reproduce.py:incline`: split, train-only scaler, 모델 선택, 저장
- `scripts/infer.py:prepare_features`: 입력 검사·정규화·clip
- `scripts/infer.py:run`: metadata, 모델 로드, 확률·전환 출력
- `packages/groot/scripts/train_tapnextpp_subtask_probe.py:make_model`: Linear/MLP 구조

## 문제별 확인

| 증상 | 확인 / 해결 |
|---|---|
| torch 누락 | `uv sync --frozen --extra train` 후 동일 옵션으로 실행 |
| checksum 실패 | 데이터가 수정되거나 clone이 불완전함. manifest를 고쳐 숨기지 말고 입력 원본 확인 |
| output already exists | 기존 결과 보존. 새 `--output` 경로 지정 |
| hidden_key missing / shape 오류 | incline50 학습 산출물과 `[N,2048]` 특징을 사용하는지 확인 |
| checkpoint shape mismatch | 같은 학습의 Linear/MLP 파일·scaler·results.json을 함께 사용 |
| clip 비율이 크거나 단계가 계속 잘못됨 | 카메라·layer·text·LoRA 상태·샘플링·학습 데이터 분포 확인. 고정 임계값으로 정상 여부를 단정하지 않음 |
| raw 점수가 표보다 낮음 | 표의 ordered와 raw를 구분. `docs/ERRATA.md` 확인 |

환경 진단과 실패 traceback을 공유할 때도 로컬 경로·비공개 데이터가 포함됐는지 확인하세요.
추론 실패는 종료 코드 2, doctor 이상은 1, 정상은 0입니다.
입력 검사 단계에서 실패하면 결과 파일을 만들지 않고 기존 출력은 덮어쓰지 않습니다.
이 검증은 실시간 GR00T 연결·GPU 특징 추출·실기 안전성 검증을 대신하지 않습니다.
