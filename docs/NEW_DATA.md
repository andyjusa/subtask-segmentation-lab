# 새 hidden·라벨로 학습하고 확인하기

기존 `incline50` 명령이 외부 경로를 받을 수 있습니다. **경사면의 다음 동작 5단계** 전용입니다.
성공 적재 개수나 다른 과업 라벨을 같은 stage 번호라는 이유로 넣으면 안 됩니다.
**MP4만 넣으면 자동 분리되는 API가 아닙니다.** 호환 hidden과 검수한 CSV·split을 먼저 준비해야 합니다.
원본부터 추출하려면 이 저장소에 없는 checkpoint와 호환 CUDA 환경도 필요합니다.

## 1. 준비

1. [라벨링 가이드](LABELING.md)의 명령으로 새 출력 폴더에 boundaries.csv와 split.json을 만듭니다.
2. [CUDA 특징 추출](REPRODUCTION.md)의 `--split`에 그 split.json을 넣습니다.
3. 추출한 hidden 경로, 라벨 CSV, split JSON을 아래 학습 명령에 함께 넣습니다.

```text
outputs/new-hidden/
  train/episode_003/groot_backbone_hidden.npz
  ...
  validation/episode_000/groot_backbone_hidden.npz
outputs/labels-new/boundaries.csv
outputs/labels-new/split.json
```

episode 번호는 예시이며 실제 split 목록과 일치해야 합니다.
NPZ에는 float `[N,2048]`의 `vision_mean` 또는 `hidden_mean`, integer `[N]`의 `sample_frames`가 필요합니다.
CSV 필수 열은 `episode,boundary_1_s,boundary_2_s,boundary_3_s,boundary_4_s`,
split JSON은 `{"train":[3,4],"validation":[0]}` 형태입니다. 이 작은 split은 실행 확인에만 씁니다.

frame은 원본 영상 기준으로0부터 일정 간격이어야 합니다. NaN·episode 중복·split 겹침·누락 라벨·
역순 경계·FPS 불일치는 거부합니다. 5개 단계가 모두 등장해야 하고, ordered decoder가 단계당 최소3초를
강제하므로 총 샘플도 최소15초 분량이어야 합니다. 이것이 실제 경계의 정확성을 검증해 주지는 않습니다.

## 2. 학습 → 추론 → 시각화

저장소 루트에서 실행합니다. 출력 경로는 이전 결과가 없는 새 경로를 사용하세요.
아래 episode 000은 validation에 포함된 경우의 예시입니다.

```bash
uv run --frozen --extra train python scripts/reproduce.py incline50 \
  --features-dir outputs/new-hidden \
  --boundaries outputs/labels-new/boundaries.csv \
  --split outputs/labels-new/split.json \
  --selection-episodes 8 --source-fps 30 --sample-fps 5 \
  --label-version incline-release-reviewed-v1 --output outputs/new-probe

uv run --frozen --extra train python scripts/infer.py \
  --model-dir outputs/new-probe \
  --features outputs/new-hidden/validation/episode_000/groot_backbone_hidden.npz \
  --output outputs/new-episode-000.jsonl

uv run --frozen python scripts/visualize.py \
  --predictions outputs/new-episode-000.jsonl \
  --boundaries outputs/labels-new/boundaries.csv --episode 0 --source-fps 30 \
  --output outputs/new-episode-000.html
```

`--label-version`은 실제 라벨 버전 이름으로 바꿉니다. 위 이름이 기존 수동 검수를 뜻하지 않습니다.
기본40개 train 중8개는 모델 선택에만 쓰고32개로 학습합니다. 작은 smoke split은
`--selection-episodes 1 --epochs 2`로 확인하세요. selection 수는 train 목록 수보다 작아야 합니다.
추론은 raw softmax 단계이며 학습 보고서의 비인과적 ordered 후처리와 다릅니다.

`results.json.inputs`에 라벨·split·각 특징 파일 SHA-256, 입력 FPS, 차원, 라벨 이름을 기록합니다.
dataset/backbone revision은 자동으로 알 수 없어 null입니다. 파일 해시는 사용한 데이터·모델 revision의 대체물이 아닙니다.

## 3. 별도 자료와 현재 검증 범위

| 준비물 | 제공 상태 / 해야 할 일 |
|---|---|
| 원본 Incline_new 데이터 | `leapshared/Incline_new_20260902_203009`; 원본은 미포함, 권한·로컬 경로 필요. 당시 revision 미기록 |
| GR00T checkpoint | 해당3카메라·16D embodiment와 일치하는 checkpoint 필요. 과거 checkpoint-300의 배포 가중치·revision 미포함 |
| Isaac-GR00T CUDA 환경 | 기존 호환 환경 필요. root uv 환경이 이를 설치하거나 CUDA 재현을 검증하지 않음 |
| tracker 가중치·원본 영상 | 미포함. vision-only 학습에는 tracker 파일 불필요 |
| 포함된 파생 특징 | clone 후 CPU 실행 가능. 55개 fixture 해시는 고정 |

이번에는 **포함 특징3개를 외부 폴더로 복사해2 epoch 학습→저장→재로드→290샘플 추론**을 검증했습니다.
원본 dataset에서 새 hidden을 추출하는 CUDA 전 과정이나 새로운 데이터의 정확도를 검증한 것은 아닙니다.
기존 `--exclude-tracker`도 hidden이 있으면 tracker 없이 읽도록 고쳤습니다.
hidden 없이 robot-only를 실행하는 legacy 경로까지 tracker 의존성을 없앤 것은 아닙니다.
실시간 적용은 [온라인 연결 범위](ONLINE_INTEGRATION.md)를 확인하세요.
