# 정정과 해석 한계

**최신 중요 정정:** 실제 rollout의 기존 51.100/65.333/94.700/102.867초 경계는 영상의 첫 release와 맞지 않았습니다.
과거 성공 판정 점수·디버그 영상은 해당 보존 라벨 기준입니다. [검수 기록](CAMERA_UPDATE.md#rollout-참조-라벨-정정)의 후보도 최종 정답은 아니며 원본 fixture는 유지합니다.

1. **경사면50개**를 “추 제거·카트 복귀를 포함한 과업”으로 설명한 것은 오류입니다. 50개는 추4개와 pointing 과업입니다. 별도로 수행한 **SciEdu 73초 영상**에는 추 제거·카트 복귀가 실제로 있으며, 이 실험을 취소하거나 부정하는 정정은 아닙니다. [별도 실험](EXPERIMENTS.md#7-sciedu-73초--cosmos-시간창fps)을 구분해서 읽으세요.
2. 다만 라벨 의미는 다릅니다. 시연은 다음 수행 단계(첫째~넷째,pointing), rollout은 성공한 추 개수(0~4)입니다. 같은 라벨이라고 간주하면 안 됩니다.
3. rollout hidden 평가에서는 **기존 backbone 표현을 재사용하되 새 Logistic Regression을 같은 영상에서 학습**했습니다. 기존50개 probe를 그대로 옮긴 실험이 아닙니다.
4. 과거 OOF helper의 `causal one-second gap` 표현을 정정했습니다. 실제로는 같은 stage에만 embargo를 적용하고 미래 프레임도 train에 포함합니다. 알고리즘은 보존했습니다.
5. ordered F1은 비인과적 후처리 결과입니다. 과거 보고서의 “온라인 사용 가능”처럼 읽히는 문장보다 이 정정이 우선합니다.
6. π0.5 98.78%는 accuracy이며 F1이 아닙니다. GR00T Stage F1과 경계 event F1도 다른 지표입니다.
7. peak6536.55MiB는 약6.38GiB입니다. 과거6.54GiB 표기는 단위 혼동입니다.
8. 가중치/원본 dataset revision이 남아 있지 않은 실험은 완전한 bitwise 재현을 주장하지 않습니다. 포함 fixture는 SHA-256로 고정합니다.
9. 약한 gripper 라벨은 tracker/action 입력과 상관이 있으므로 label leakage/shortcut 우려가 있습니다. action gripper 제거 비교만으로 완전히 해소되지 않습니다.
10. validation을 여러 번 보며 설계를 골랐으므로 새 test set이 필요합니다. 단일 실패와 단일 rollout로 실제 실패 FPR을 일반화할 수 없습니다.

`archive/`는 당시 보고서를 보존한 것입니다. 이 문서와 새 검증 기록이 해석상 우선합니다.
