# 보존 라벨 버전과 차이

아래 이름은 2026-09-07에 **식별용으로 새로 붙인 이름**이며 당시 버전 이름을 복원한 것이 아닙니다.
원본 결과와 라벨 파일을 수정하지 않았습니다.

| 식별 이름 | 경계(초) | 의미 / 원본 |
|---|---|---|
| point-pilot-ep000-preserved |15.77 /25.73 /36.23 /48.03|[point pilot 결과](../archive/groot/incline_new/point_only_subtask_episode_000/results.json)의 참조값 |
| incline-release50-preserved |15.8 /25.633333 /36.1 /47.833333|[현재 CSV](../fixtures/incline50/boundaries.csv)의 episode000, 기본 CPU 재현 라벨 |
| rollout-success-ep000-preserved |51.1 /65.333333 /94.7 /102.866667|[rollout JSON](../fixtures/rollout/boundaries.json)의 성공 누적 경계 |

pilot 대비 현재 ep000은 +0.03/−0.096667/−0.13/−0.196667초 차이가 납니다.
**언제·누가·어떤 이유로 바꿨는지는 보존 기록으로 확인하지 못했습니다.** 수동 보정이었다고 추정하지 않습니다.
pilot의 경계 MAE6.97초는 pilot 참조값 기준입니다. 모든 과거 artifact가 현재 CSV와 같다고 보장하지 않습니다.
rollout은 별도 영상이고 성공 개수 라벨이므로 위 두 시연 버전과 수치 대응 비교를 하면 안 됩니다.

## 파일 SHA-256

```text
point pilot results.json
e9f177fd12bc46126bd6f85f147fdba53604d762ffa45c1fb204cf7a42c7f1b0
fixtures/incline50/boundaries.csv
c03e98e6b41907f7a327c6f85904f1ecf8703889c86e6671810e377908f617cc
fixtures/rollout/boundaries.json
afa7dbce1ea7f4c7c58340626fbe94cd78092fda0bc53d43245a1271ab765c2e
```

새 학습은 `results.json.inputs`에 라벨 버전 이름·라벨 해시·split 해시·각 특징 해시를 함께 저장합니다.
향후 수동 검수로 변경할 때는 새 파일과 변경 이유를 남기고 기존 점수는 기존 라벨 버전과 함께 보존하세요.
