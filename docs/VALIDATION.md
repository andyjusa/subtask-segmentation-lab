# 이번 패키지 검증 (2026-09-07)

Mac arm64, 신규 `.venv`,Python3.12.13,NumPy1.26.4,scikit-learn1.9.0,
PyTorch2.14.0,CPU. 정확한 의존성은 uv.lock에 고정했습니다.

## 실제 실행 결과

| 실행 | 이번 결과 | 과거 결과와 비교 |
|---|---|---|
| π0.5 actual action Linear | accuracy .987805,경계170/350 | 일치 |
| incline50 R3b Linear | raw F1 .855657,ordered .899004,MAE1.028333초 | ordered 일치 |
| incline50 R3b MLP | raw .882351,ordered .901284,MAE1.026667초 | 과거 .9020/.997초와 소폭 차이 |
| rollout R3b native | raw .719549,ordered .866474,MAE2.833333초 | 일치 |
| rollout R3b PCA256 | raw .729466,ordered .868052,MAE2.783333초 | 일치 |
| rollout R3 native | raw .728008,ordered .868052,MAE2.783333초 | 일치 |
| rollout R3 PCA256 | raw .727925,ordered .868052,MAE2.783333초 | 과거 raw .733255와 차이;ordered 일치 |

모델/BLAS/의존성/플랫폼이 역사적 실행과 같다고 확인되지 않았으므로 차이의 원인을 단정하지 않습니다.
과거 수치를 덮어쓰지 않고 이번 결과는 `validation/2026-09-07/`에 별도 보존합니다.

## 테스트

- root fixture/checksum/split/OOF contract:4 passed
- π0.5 기존 단위 테스트:10 passed
- GR00T 기존 단위 테스트:70 passed
- CPU로 세 재현 명령 모두 완료. incline50 기본80epoch/early stopping;Linear best18,MLP best25.

GR00T pytest가 처음에는 pyarrow/gymnasium 누락으로 수집 실패하여 dev 의존성에 추가한 후70개 통과했습니다.
이 테스트가 CUDA 특징 추출,실제 로봇 제어,모든 외부 tracker 설치까지 검증했다는 뜻은 아닙니다.
새 runner/tests는 Ruff 검사 대상이며 보존한 과거 코드 전체를 일괄 재포맷하지 않았습니다.

## 전달용 추론·디버깅 경로 추가 검증

- root 19개 + 기존 π0.5 10개 + GR00T 70개: 총 99개 통과.
- 새 테스트가 실제 2 epoch 학습 → Linear/MLP 저장 → 재로드 → 추론을 실행한다.
- 입력 shape·NaN·frame 정렬·scaler·출력 덮어쓰기 방지를 검사한다.
- `doctor.py`: Python 3.12.13, fixture 55개 정상. 비밀 환경값은 출력하지 않는다.
- `infer.py`: 기존 기본 MLP로 episode 000의 `[290,2048]` 특징 추론 완료, clip 비율 0.
- 동일 명령을 표준 pdb로 실행 완료. VS Code UI는 별도 검증하지 않았다.
- `visualize.py`: raw 단계·참조 라벨·확률 HTML 생성 및 데이터 계약 테스트 통과.
  자동 브라우저의 로컬 파일 접근 제한으로 HTML의 브라우저 화면 확인은 하지 못했다.
- 새 코드 Ruff format/check 및 `git diff --check` 통과.

새 테스트는 기능 확인이며 본 실험의 80 epoch 학습 점수를 다시 갱신한 것이 아니다.
