# Sub-task Segmentation Lab

로봇 행동을 단계로 분리한 실험 모음입니다. π0.5 action, GR00T 내부 표현,
point tracking과 action/state 융합을 비교합니다. **개인 비공개 연구 저장소**이며 원본 프로젝트는 변경하지 않았습니다.

[Notion 실험 정리](https://app.notion.com/p/3d4a84204fc581f8999ee5cc475d8c32)

## 바로 실행

Python 3.12와 uv가 필요합니다. 모델 다운로드·HF 로그인·GPU 없이 포함된 **실제 특징 데이터**로 실행합니다.

```bash
git clone git@github.com:andyjusa/subtask-segmentation-lab.git
cd subtask-segmentation-lab
uv sync --frozen --extra train
uv run --frozen --extra train python scripts/doctor.py
uv run --frozen python scripts/reproduce.py verify
uv run --frozen --extra train python scripts/reproduce.py pi05
uv run --frozen --extra train python scripts/reproduce.py incline50
uv run --frozen python scripts/reproduce.py rollout
uv run --frozen --extra train pytest -q -ra
```

학습한 GR00T 분류기로 저장된 특징을 추론합니다. 샘플마다 단계·확률을 JSONL로 남기며,
입력 규격과 모델 호환성을 검사합니다. 실시간 GR00T 서버나 로봇을 실행하는 명령은 아닙니다.

```bash
uv run --frozen --extra train python scripts/infer.py \
  --model-dir outputs/incline50 \
  --features fixtures/incline50/validation/episode_000/groot_backbone_hidden.npz \
  --output outputs/episode_000.jsonl
```

[디버깅 안내](docs/DEBUGGING.md): 환경 진단, 2 epoch 학습→로드→추론 테스트, pdb와 VS Code breakpoint 설정.
[그래프·실험 영상](docs/VIDEOS.md): raw 단계·확률 HTML 생성 및 Notion의 tracker/SAM3 대표 영상.
[새 데이터로 학습→추론](docs/NEW_DATA.md): 외부 hidden·라벨·split을 받는 실행 경로.
[GR00T 온라인 연결 범위](docs/ONLINE_INTEGRATION.md): 현재 되는 것과 추가 구현할 것.

기본 pytest는 root·π0.5·GR00T 테스트를 모두 수집합니다. `--extra train`을 포함하고
skip 없이 완료됐는지 확인하세요. CPU 테스트 통과가 CUDA나 로봇 동작 검증은 아닙니다.

결과는 `outputs/<실험>/results.json`에 저장됩니다. 재실행할 때는 `--output outputs/run2`처럼 새 경로를 사용합니다.
incline50은 Linear와 MLP를 둘 다 학습합니다. `--epochs 2`는 실행 확인용이며 본 실험 점수와 비교하지 않습니다.
`--hidden-key hidden_mean`으로 R3 전체 토큰 평균과 R3b vision 평균을 비교할 수 있습니다.

## 읽는 순서

라벨이 어떻게 만들어졌는지는 [라벨링 기준·예외·재생성 방법](docs/LABELING.md)을 먼저 참고하세요.

1. [실험표·결론](docs/EXPERIMENTS.md): 데이터, 라벨, 분할과 성능을 같은 표에서 확인
2. [구조·표현·평가 지표](docs/METHODS.md): 무엇을 추출하고 어떻게 경계를 찾는가
3. [실행 방법·재현 범위](docs/REPRODUCTION.md): CPU 재평가, GPU 원본 특징 추출, 외부 입력
4. [정정·한계](docs/ERRATA.md): 온라인/오프라인, 같은 영상/다른 영상 검증 구분
5. [검증 기록](docs/VALIDATION.md): 이번 환경에서 실제 실행한 결과

## 구성

```text
scripts/reproduce.py     통합 CPU 실행 진입점
fixtures/               실제 hidden/action, 라벨, 고정 split, SHA-256 manifest
packages/pi05/          action/hidden probe, detector, Task API adapter, 기존 테스트
packages/groot/         frozen probe, causal transformer, tracker/fusion, 계측·pause 코드
archive/                당시 결과 파일과 보고서; 현재 재검증 결과와 구분
docs/                   실험표, 구조, 실행, 한계
outputs/                실행 결과·새로 학습한 probe (git 제외)
```

**해석 주의:** incline의 `ordered F1`은 전체 영상을 본 오프라인 결과입니다.
실제 rollout 1개의 점수는 같은 영상에서 새 probe를 학습한 OOF 결과이며,
기존 50-episode 분류기의 무학습 전이 성능이 아닙니다.
가중치·원본 영상·비밀키·타사 모델 소스는 배포하지 않습니다.
외부 모델/데이터의 라이선스와 접근 권한은 각 제공자의 조건을 따릅니다.
