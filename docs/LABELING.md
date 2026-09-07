# 기존 라벨을 어떻게 만들었나

## 먼저 구분할 것

**정답 라벨을 만드는 규칙과 모델이 경계를 예측하는 방법은 다릅니다.**
경사면 시연에서는 gripper action으로 참조 경계를 만들고, 그 라벨을 맞히도록
vision hidden·tracker·action/state 기반 분류기를 학습했습니다.
SAM3나 tracker의 점이 정답 라벨을 만들어 준 것은 아닙니다.

| 실험 | 라벨의 의미 | 라벨 생성 근거 | 보존 위치 |
|---|---|---|---|
| 경사면 시연 50개 | 지금 수행하는 단계: 추 1~4 추가, 마지막 pointing | 오른쪽 gripper release 후보 4개를 자동 선택한 약한 라벨 | [생성 코드](../packages/groot/scripts/prepare_incline_50_split.py), [경계 CSV](../fixtures/incline50/boundaries.csv), [split](../fixtures/incline50/split.json) |
| 실제 경사면 rollout 1개 | 카트에 성공적으로 올린 추의 누적 개수 0~4를 의도 | 당시 opening으로 기록했으나 이후 참조 지연 오류 발견; 해석 보류 | [보존 JSON](../fixtures/rollout/boundaries.json), [재검수](CAMERA_UPDATE.md#rollout-참조-라벨-정정) |
| GR00T LIBERO 3종 | 첫 조건 전 / 첫 조건 후 / 전체 성공 | 시뮬레이터 내부 predicate·success | [predicate 코드](../packages/groot/src/groot_subtask_phase_probe/predicates.py), [task 조건](../packages/groot/src/groot_subtask_phase_probe/tasks.py) |
| π0.5 pick-and-place | picking / placing / complete | 저장된 기준 경계 170·354프레임 | [실험 기록](EXPERIMENTS.md), [학습 진입점](../scripts/reproduce.py) |

숫자가 같은 stage라도 실험 사이에서 의미가 다릅니다. 특히 시연의 stage 0은
‘첫 추를 옮기는 구간’, rollout의 stage 0은 ‘성공적으로 놓인 추가 아직 없음’입니다.

## 1. 경사면 시연 50개: gripper 동작으로 자동 분할

대상은 `leapshared/Incline_new_20260902_203009`입니다.
추를 한 개씩 놓을 때 오른쪽 gripper가 열리는 움직임이 반복되는 점을 이용했습니다.
**이 데이터에서** 그 동작은 `action[:, 15]` 값이 감소하는 방향입니다.
관측 state가 아니라 action 명령을 사용하며, 다른 로봇의 채널·단위·부호에 그대로 적용하면 안 됩니다.

### 실제 코드의 처리 순서

| 순서 | 처리 | 코드에 들어 있는 값과 예외 |
|---|---|---|
| 1 | episode별 오른쪽 gripper action을 frame 순으로 읽음 | 16차원 action 중 index 15. 원본 FPS는 `meta/info.json` 사용 |
| 2 | 각 시점 전·후 중앙값의 감소량 계산 | 약 0.40초 전후 창에서 중앙 약 0.07초를 제외. 앞뒤 2초는 후보 탐색 제외 |
| 3 | 충분히 큰 국소 감소를 후보로 선택 | 감소량 ≥ `max(3.0, 0.08 × (P90−P10))`, 약 ±0.35초 이웃의 최대 감소량 |
| 4 | 한 번의 열림에서 나온 중복 후보 병합 | 코드상 직전 보존 후보와 1.5초 미만이면 더 큰 감소량을 보존 |
| 5 | 최종 경계 4개 선택 | 후보가 정확히 4개면 그대로 사용. 4개 초과일 때 간격 ≥4초인 조합 중 감소량과 간격 규칙성을 함께 고려 |
| 6 | 경계 사이를 5개 stage로 지정 | 시작→경계1→경계2→경계3→경계4→종료 |

`P90−P10`은 episode 전체 action의 변동 폭입니다. 후보가 4개 초과일 때의 점수는
`평균(감소량 / 변동 폭) − std(경계 간격) / mean(경계 간격)`입니다.
마지막 손 복귀 동작이 네 번째 release 대신 선택되는 것을 줄이기 위해 간격의 규칙성을 사용했습니다.
**4초 최소 간격 검사는 후보가 정확히 4개인 분기에는 적용되지 않습니다.**

4개를 고를 수 없으면 해당 episode를 건너뛰고 실패 진단을 기록한 뒤 마지막에 오류를 냅니다.
남은 CSV가 존재한다고 전체 50개 라벨 생성이 성공한 것은 아닙니다.
`validation_report.json`의 `failures`, `episodes_valid`와 프로세스 종료 코드를 함께 봐야 합니다.

### 실제 episode 000 예시

| 구간: 시작 포함, 끝 제외 | stage ID | 저장된 이름 | 의미 |
|---|---:|---|---|
| 0 ~ 15.800초 | 0 | `place_weight_1` | 첫 추를 옮기는 구간 |
| 15.800 ~ 25.633초 | 1 | `place_weight_2` | 첫 release 후 두 번째 추 구간 |
| 25.633 ~ 36.100초 | 2 | `place_weight_3` | 두 번째 release 후 세 번째 추 구간 |
| 36.100 ~ 47.833초 | 3 | `place_weight_4` | 세 번째 release 후 네 번째 추 구간 |
| 47.833 ~ 58.000초 | 4 | `final_pointing` | 네 번째 release부터 영상 끝까지 |

`final_pointing`은 pointing 동작 시작을 별도로 검출한 결과가 아닙니다.
네 번째 추를 놓은 뒤 팔을 빼거나 기다리는 시간도 이 라벨에 포함될 수 있습니다.
또한 후보는 action 감소량의 대표 시점이므로 물리적으로 추를 완전히 놓은 첫 frame과 반드시 같지는 않습니다.

### 프레임 단위 학습 라벨로 바꾸는 법

CSV의 초 단위 경계를 원본 FPS로 frame에 대응시키고,
각 `sample_frames`까지 지난 경계 개수를 stage ID로 사용합니다.
학습 코드의 `np.searchsorted(boundaries, sample_frames, side="right")`에 따라
경계 frame 자체는 새 stage에 들어갑니다.

예를 들어 원본 30 FPS에서 15.8초는 frame 474입니다.
frame 473은 stage 0, frame 474는 stage 1입니다.
hidden은 5 FPS로 추출해도 `sample_frames`는 원본 frame 인덱스입니다.
경계 시간을 5 FPS로 변환하면 정렬이 어긋납니다. legacy tracker 학습 loader는 여전히 원본 30 FPS를 가정합니다.
새 루트 학습 경로는 `--source-fps`와 `--sample-fps`를 받아 frame 간격을 검증합니다.
[새 데이터 실행](NEW_DATA.md)과 [보존 라벨 버전 차이](LABEL_VERSIONS.md)를 확인하세요.

### 학습·검증 분리

episode 0~49를 seed 42로 섞어 40개 train / 10개 validation으로 고정했습니다.
CPU 학습은 train 40개 중 32개로 학습하고 8개로 모델을 선택합니다.
같은 episode의 일부 frame을 train, 나머지를 validation으로 섞지 않습니다.

### 무엇을 보장하지 못하나

- gripper를 열어도 추가 카트 밖으로 떨어질 수 있으므로 **release 라벨 ≠ 적재 성공 라벨**입니다.
- 앞뒤 창과 전체 episode 통계를 사용하고 경계 4개를 고르므로 온라인 감지기가 아닌 오프라인 라벨러입니다.
- 규칙적인 4회 수행을 가정해 실패·재시도·비정상적으로 느린 수행에서 잘못 고를 수 있습니다.
- action으로 만든 라벨을 action 입력 모델로 맞히면 같은 신호를 읽는 지름길 학습이 가능합니다.
  gripper 채널을 제외해도 다른 관절과 상관이 남으므로 이 문제가 완전히 사라지지는 않습니다.

## 2. 실제 rollout: 성공 장면을 확인한 별도 라벨

**2026-09-07 재검수 정정:** 아래는 당시 기록을 보존한 설명입니다. 손목 영상·action을 대조하니
기존 4개 경계가 release보다 늦었으며, 아래 ‘첫 opening frame’ 설명은 실제 영상과 맞지 않았습니다.
원본 fixture와 과거 결과는 유지하지만 성공 시점 평가로의 해석은 보류합니다.
검수용 후보 46.000/56.167/86.900/100.133초도 확정 정답이 아닙니다.
[영상 근거·후보 계산·남은 검수](CAMERA_UPDATE.md#rollout-참조-라벨-정정)를 우선해서 읽으세요.

대상은 `leapshared/rollout_Incline_20260903_20260903_190043`, episode 000입니다.
실패·재시도가 있어 release 횟수를 성공 횟수로 쓰지 않았습니다.
기존 기록의 기준은 **영상에서 성공 적재를 확인한 뒤, 해당 동작의 첫 오른쪽 gripper opening frame**입니다.
이 문서 작업에서 영상을 새로 재판정한 것이 아니라 기존 JSON의 근거를 설명한 것입니다.

| 성공 후 누적 개수 | 경계 frame | 시간 |
|---:|---:|---:|
| 1 | 1533 | 51.100초 |
| 2 | 1960 | 65.333초 |
| 3 | 2841 | 94.700초 |
| 4 | 3086 | 102.867초 |

22.067·40.800·81.633초 시도는 적재 성공으로 세지 않았습니다.
사람이 원래 위치의 추를 재배치한 구간도 JSON에 따로 기록했습니다.
‘안정적으로 놓였다’를 판단하는 고정 유지시간·가림 처리 규칙·독립 검수자 간 일치도는
기존 기록에 정량적으로 명시돼 있지 않으므로 완성된 객관적 annotation protocol로 보아서는 안 됩니다.
카트에 놓인 추가 이후 떨어지는 상황을 다루는 감소 라벨도 현재 누적 0→4 정의에는 없습니다.

시연의 마지막 stage는 pointing이지만 여기서는 ‘추 4개 적재’입니다.
따라서 시연에서 학습한 5-class 모델과 바로 비교하거나 같은 의미의 라벨로 합치면 안 됩니다.

## 3. LIBERO와 π0.5의 기준

LIBERO는 영상이나 VLM 판단 대신 환경의 참/거짓 조건을 읽었습니다.

| task | 첫 경계 조건 | 전체 성공 조건 설명 |
|---|---|---|
| stove / moka pot | stove on | moka pot on stove |
| bowl / drawer | bowl in bottom drawer | drawer closed 및 task success |
| mug / microwave | mug in microwave | microwave closed 및 task success |

첫 predicate가 아직 한 번도 참이 아니면 stage 0, 처음 참이 된 뒤 성공 전까지 stage 1입니다.
전체 성공 후 샘플은 `stage=-1`, `terminal=1`로 별도 저장합니다.
정확한 첫 경계는 raw simulator step 단위 `exact_boundary_env_step`이고,
policy 특징 샘플의 `boundary`는 그 경계가 포함된 action chunk를 표시합니다. 두 시간축을 혼동하면 안 됩니다.
첫 조건을 끝내 만족하지 못한 실패 episode는 경계가 없고, 이때 경계 예측은 오탐입니다.

π0.5 CPU 재현은 이미 지정된 170·354프레임을 학습 함수에 전달합니다.
action에서 새 정답 경계를 자동으로 찾는 라벨러가 아닙니다.
410프레임 성공 episode와 기존 경계 기록은 보존했지만, 그 두 frame을 선정한 상세 검수 절차까지
이 패키지에서 다시 검증한 것은 아닙니다.

## 4. 기존 자동 라벨을 다시 생성하는 명령

원본 dataset은 GitHub fixture에 포함하지 않았습니다. 접근 권한이 있는 사람이 실제 dataset 경로를 준비해야 합니다.
현재 loader는 `meta/info.json`, `meta/episodes/chunk-000/file-000.parquet`,
`data/chunk-000/file-000.parquet`와 16차원 action, 연속 episode ID 0~49를 전제로 합니다.
여러 parquet chunk를 순회하는 범용 loader는 아닙니다.
영상 clip 생성에는 `videos/observation.images.follower_d455f/chunk-000/` 아래 원본 MP4와 ffmpeg가 추가로 필요합니다.

저장소 루트에서 실행합니다. **기존 스크립트는 결과를 덮어쓸 수 있으므로 반드시 새 출력 경로를 쓰세요.**

```bash
uv sync --frozen --extra train
# /path/to/...를 본인이 준비한 원본 dataset 경로로 변경
uv run --frozen python packages/groot/scripts/prepare_incline_50_split.py \
  /path/to/Incline_new_20260902_203009 \
  --output-dir outputs/labels-new --episodes 50 --train-episodes 40 --seed 42 --analyze-only
```

`--analyze-only`는 경계 계산·split·clip 메타데이터만 기록합니다. 영상 파일은 만들지 않지만
영상의 file_index/from_timestamp가 있는 episode metadata는 여전히 필요합니다.

산출물은 `boundaries.csv`, `split.json`, `manifest.jsonl`, `validation_report.json`입니다.
원본 영상까지 250개 구간으로 잘라 보려면 **다른 새 출력 경로**로 위 명령을 실행하면서
`--analyze-only`만 빼고, 생성 후 아래 검사를 실행합니다.

```bash
uv run --frozen python packages/groot/scripts/verify_incline_split.py outputs/clips-new
```

여기서 `outputs/clips-new`는 실제로 clip 생성에 지정한 경로여야 합니다.
이 검사는 파일·길이·frame 수·구간 연결·40/10 split을 확인하지, 적재 성공 의미를 검수하지는 않습니다.
재생성한 CSV와 `fixtures/incline50/boundaries.csv`를 비교해 차이가 있으면 입력 버전부터 조사합니다.
저장된 fixture나 manifest 체크섬을 덮어써서 차이를 숨기면 안 됩니다.

## 5. 검수와 현재 남은 일

다음은 권장 검수 절차이며, 기존 50개가 이 절차로 전부 검수됐다는 뜻은 아닙니다.

1. 각 후보 전후 영상을 보고 실제 놓기·재시도·손 복귀를 구분합니다.
2. ‘다음 동작 시작’, ‘release 대표 시점’, ‘안정적 적재 성공’ 중 어느 사건을 정답으로 삼을지 먼저 고정합니다.
3. pointing의 실제 시작을 평가하려면 별도로 annotation하고 마지막 release와 혼용하지 않습니다.
4. 수정 전후 frame, 이유, 검수자, 원본 데이터 버전을 기록하고 별도 라벨 버전으로 보존합니다.
5. 수정한 라벨로 모든 비교 모델을 같은 split에서 다시 평가합니다. 모델의 높은 점수에 맞춰 라벨을 바꾸지 않습니다.

이번 문서화에서는 코드·fixture·기존 기록을 대조했습니다. **원본 데이터 전체 재라벨링이나 독립 수동 검수는 새로 실행하지 않았습니다.**
