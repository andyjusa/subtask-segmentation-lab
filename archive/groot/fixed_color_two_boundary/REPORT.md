# GR00T 고정색 vs 무작위색 2경계 C2 실험

## 한 줄 결론

색을 파랑→노랑으로 고정하면 color boundary F1@±1 sample은 `0.902 ± 0.102`,
no-change FPR은 `0.000`이었다. 색 조합을 무작위화하면 color F1은 `0.211 ± 0.149`,
FPR은 `0.600 ± 0.480`으로 무너졌다. 다만 stove와 color 경계를 순서대로 모두 맞히는
전체 정확도는 고정색에서도 `0.571 ± 0.266`이므로, 현재 구조를 바로 제어에 연결하기에는
stove 경계 안정성이 부족하다.

## 실험 구조

- 환경: 실제 LIBERO `KITCHEN_SCENE3` stove+moka task
- 경계 1: 로봇이 knob를 돌려 `turnon flat_stove_1` predicate가 최초 true가 된 sample
- 경계 2: 경계 1 이후 무작위 2~6 policy sample 뒤 액체색이 처음 바뀐 sample
- Fixed 조건: 파랑 `(42,111,219)` → 노랑 `(224,180,52)`
- Random 조건: 5색 팔레트에서 서로 다른 초기색·변경색을 pair마다 선택
- 총 24개 simulator seed를 두 조건에 paired 적용해 48 episode 수집
- mode별 change 18개, no-change 6개
- stove 이후에는 GR00T action을 버리고 zero-motion hold를 적용하되 physics·camera·추론은 계속 진행
- 로봇 제어 action은 오버레이 없는 raw camera에서 생성하고, probe feature만 색 오버레이 forward에서 추출
- 모든 episode에서 stove 성공; fixed/random의 stove sample과 delay가 pair별로 동일함을 감사

## 모델과 평가

- 표현: GR00T R3b vision token hidden, 2048차원
- train-only StandardScaler + PCA-256
- C2 설정: window 16, d=256, 4 heads, 2 layers, FFN512
- stage loss 없음, 정확한 경계 1 sample만 positive
- stove와 color에 각각 독립된 C2 probe를 학습
- 미래 frame, smoothing, Viterbi, backtracking 없이 첫 rising event만 사용
- paired episode split 60/20/20, seeds `17,29,41,53,67`
- 표의 ±는 5-seed 95% CI

## 최종 결과

| 조건 | 두 경계 순서 포함 정확도@±1 | Change episode 정확도 | Stove F1@±1 | Color F1@±1 | Color no-change FPR |
|---|---:|---:|---:|---:|---:|
| Fixed 파랑→노랑 | **0.571 ± 0.266** | **0.680 ± 0.266** | **0.811 ± 0.120** | **0.902 ± 0.102** | **0.000 ± 0.000** |
| Random 색 조합 | 0.114 ± 0.105 | 0.080 ± 0.157 | 0.684 ± 0.147 | 0.211 ± 0.149 | 0.600 ± 0.480 |

±2 sample로 허용 범위를 넓혀도 고정색 결과는 같았고, random color F1만 `0.231 ± 0.153`으로
소폭 올랐다. 즉 대부분의 실패가 1~2 sample의 작은 시간 오차가 아니라 완전 누락 또는 여러 번의
오정지였다.

## Paired 차이

| 지표 | Fixed − Random 평균 차이 | 95% CI |
|---|---:|---:|
| 두 경계 순서 포함 정확도 | **+0.457** | ±0.299 |
| Color F1 | **+0.692** | ±0.227 |
| Stove F1 | +0.127 | ±0.172 |
| Color no-change FPR | **−0.600** | ±0.480 |

같은 simulator seed와 같은 변화 지연을 사용했으므로 fixed와 random의 주된 차이는 색 조건이다.
Color F1의 paired 차이는 모든 seed에서 fixed가 높았다. 반면 Stove F1 차이는 CI가 0을 포함하므로
고정색이 stove 검출 자체를 개선했다고 단정할 수 없다.

## 해석

### 초기색과 변경색이 고정된 경우

- 색 변화만 보면 F1 90.2%, no-change 오정지 0%로 잘 작동했다.
- color timing error median은 모든 seed에서 0 sample이었다. 검출했을 때는 정확한 변화 sample을 맞혔다.
- 실패는 주로 color보다 stove boundary probe에서 발생해 전체 2경계 정확도를 57.1%로 낮췄다.

### 색을 무작위화한 경우

- 18개 change episode만으로 여러 초기색→변경색 조합을 일반화하지 못했다.
- 어떤 seed에서는 색 변화를 전부 놓쳤고, 다른 seed에서는 no-change 대조군마다 오정지를 냈다.
- ±2 sample에서도 거의 회복되지 않으므로 threshold 미세조정보다는 데이터 다양성과 색 불변 표현이 필요하다.
- 현재 random 모델은 color FPR gate 0.25를 넘으므로 배포 부적합이다.

## 구조상 제한

- 두 boundary head는 encoder를 공유하지 않는 독립 C2 probe 두 개다. 총 probe 파라미터는 약 225만 개다.
- seed별 test는 7 episode이고 그중 no-change가 2개라 FPR이 0, 0.5, 1 단위로 거칠게 변한다.
- 색은 합성 beaker overlay이며 실제 액체·조명·반사 domain shift는 포함하지 않는다.
- random 조건은 색 조합당 표본이 적어, 이 결과는 색 일반화 실패를 보여주지만 충분한 random-color 데이터의 상한은 아니다.

## 다음 실험

1. random 조건을 최소 100 pair로 늘리고 색 조합별 train/test를 분리한다.
2. 하나의 causal encoder에 stove/color 두 head를 붙여 파라미터와 latency를 줄인다.
3. 색 차이 특징 또는 색상 augmentation을 추가하되 동일 evaluator로 비교한다.
4. 실제 카메라 액체 영상으로 domain shift를 검증한다.

## 대표 영상

- Fixed 성공: seed 29, stove `11→11`, color `14→14`, 추가 event 없음
- Random 실패: seed 41, stove `12→12`, color GT `18`을 완전히 누락

원시 결과는 `remote_results/two_boundary_20260828/reports/fixed_color_two_boundary/results.json`,
요약은 같은 폴더의 `summary.json`에 있다. 전체 feature와 checkpoint는 원격 WSL
`/root/projects/groot-subtask-phase-probe`에 보관한다.
