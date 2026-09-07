# 영상으로 보는 실험

[Notion 실험 페이지](https://app.notion.com/p/3d4a84204fc581f8999ee5cc475d8c32)의
대표 영상 항목에서 재생할 수 있습니다. 원본은 경사면 시연 episode 000이며
각 방법의 출력 모습을 보여주는 예시입니다. 같은 해상도·길이·점 수로 통제한 성능 비교표가 아닙니다.

| 영상 | 길이 | 무엇을 보여주나 |
|---|---:|---|
| 참조 subtask 경계 | 58초 | 정해둔 경계와 단계 표시. 모델 예측 영상이 아님 |
| SAM3 검출 + optical flow | 5초 | 카트·추 검출과 keyframe 사이 흐름 추적. 매 프레임 SAM3 추론이 아님 |
| 일반 point tracking | 58초 | Shi-Tomasi 특징점 + Lucas-Kanade optical flow |
| SAM3 재검출 + point tracking | 58초 | 2초마다 SAM3로 카트·추·실린더 점을 다시 잡고 사이 프레임은 KLT |
| CoTracker3 quasi-dense | 58초 | 400개 격자 점 추적. 모든 pixel의 dense flow와 다름 |
| AllTracker dense | 13.33초 | dense 결과에서 3,072개 점을 골라 표시 |
| TAPNext++ | 13.33초 | 같은 400프레임 pilot에서 400개 점 추적 |

재생본은 H.264/yuv420p MP4로 변환하고 원본 frame rate(30 FPS)를 유지했습니다.
비디오 재생 속도는 모델의 실제 추론 속도를 뜻하지 않습니다.
점이 잘 따라간다고 subtask 성공·경계를 정확히 안다는 의미도 아닙니다.
SAM3의 cylinder 표기는 검출 프롬프트/추적 대상이며 물리량을 측정한 결과는 아닙니다.

분류기 출력은 `scripts/visualize.py`로 HTML을 다시 만들 수 있습니다.
입력 JSONL부터 생성하는 명령은 [디버깅 안내](DEBUGGING.md)를 참고하세요.
