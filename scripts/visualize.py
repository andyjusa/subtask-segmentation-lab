"""Create an offline HTML/SVG view of raw stage predictions, without extra packages."""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
from pathlib import Path

COLORS = ["#2563eb", "#008568", "#d17c00", "#a548ba", "#d94a43"]
NAMES = ["추 1 추가", "추 2 추가", "추 3 추가", "추 4 추가", "마지막 pointing"]


def render(records: list[dict], boundaries: list[float] | None = None) -> str:
    if not records:
        raise ValueError("predictions must not be empty")
    frames = [row["source_frame"] for row in records]
    if any(type(f) is not int or f < 0 for f in frames):
        raise ValueError("source_frame must be nonnegative integers")
    if any(b <= a for a, b in itertools.pairwise(frames)):
        raise ValueError("source_frame must strictly increase")
    for row in records:
        p = row["probabilities"]
        if type(row["stage_id"]) is not int or row["stage_id"] not in range(5):
            raise ValueError("stage_id must be 0..4")
        if len(p) != 5 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p):
            raise ValueError("expected five finite probabilities in [0,1]")
        if not math.isclose(sum(p), 1, abs_tol=1e-5):
            raise ValueError("probabilities must sum to one")
    if boundaries is not None and (
        len(boundaries) != 4
        or any(not math.isfinite(v) or v < 0 for v in boundaries)
        or any(b <= a for a, b in itertools.pairwise(boundaries))
    ):
        raise ValueError("expected four finite increasing reference boundaries")
    start, stop = frames[0], frames[-1]
    span = max(1, stop - start)

    def x(frame):
        return 100 + 940 * (frame - start) / span

    def band(values, y):
        parts = []
        first = 0
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[first]:
                left = x(frames[first])
                right = x(frames[i]) if i < len(values) else 1040
                label = values[first]
                parts.append(
                    f'<rect x="{left:.2f}" y="{y}" width="{max(0.5, right - left):.2f}" '
                    f'height="24" fill="{COLORS[label]}"><title>{NAMES[label]}: '
                    f"frame {frames[first]}</title></rect>"
                )
                first = i
        return "".join(parts)

    pieces = [
        '<svg viewBox="0 0 1080 445" role="img" aria-label="시점별 단계와 확률">',
        '<text x="10" y="38">Raw 예측</text>',
        band([row["stage_id"] for row in records], 20),
    ]
    if boundaries is not None:
        truth = [sum(frame >= boundary for boundary in boundaries) for frame in frames]
        pieces += ['<text x="10" y="80">참조 라벨</text>', band(truth, 62)]
        for boundary in boundaries:
            if start <= boundary <= stop:
                pos = x(boundary)
                pieces.append(
                    f'<line x1="{pos:.2f}" x2="{pos:.2f}" y1="100" y2="360" '
                    'stroke="#64748b" stroke-dasharray="4 4"/>'
                )
    for value in (0, 0.5, 1):
        y = 360 - value * 240
        pieces += [
            f'<line x1="100" x2="1040" y1="{y}" y2="{y}" stroke="#dce3eb"/>',
            f'<text x="65" y="{y + 5}">{value}</text>',
        ]
    for label, color in enumerate(COLORS):
        points = " ".join(
            f"{x(frame):.2f},{360 - row['probabilities'][label] * 240:.2f}"
            for frame, row in zip(frames, records)
        )
        pieces.append(
            f'<polyline points="{points}" stroke="{color}" fill="none" stroke-width="1.8"/>'
        )
    for i in range(6):
        frame = start + span * i / 5
        pieces.append(f'<text x="{x(frame):.1f}" y="390" text-anchor="middle">{frame:.0f}</text>')
    pieces += ['<text x="570" y="425" text-anchor="middle">원본 frame 인덱스</text>', "</svg>"]
    legend = " ".join(f'<span style="color:{c}">● {n}</span>' for c, n in zip(COLORS, NAMES))
    reference = (
        "점선은 제공한 참조 경계입니다. gripper release 기반 약한 라벨이며 성공 판정과 다릅니다."
        if boundaries is not None
        else "참조 라벨을 넣지 않아 예측만 표시합니다."
    )
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Subtask prediction debug view</title><style>
body{{font:16px/1.6 system-ui,sans-serif;background:#f3f6fa;color:#172b40;margin:32px auto;padding:0 24px;max-width:1120px}}
section{{background:white;padding:24px;border-radius:16px}}svg{{width:100%;font:14px system-ui,sans-serif}}.legend span{{margin-right:18px;white-space:nowrap}}code{{background:#e9eef5;padding:2px 6px}}
</style><h1>단계 예측 디버그 뷰</h1><p>{len(records)}개 샘플 · 전체 영상 순서 보정 없는 raw 추론</p>
<section><div class="legend">{legend}</div>{"".join(pieces)}</section>
<p>{html.escape(reference)}</p><p>위 막대는 단계, 아래 곡선은 5개 softmax 확률입니다. 색이 바뀌는 시점과 확률이 불안정한 구간을 확인하세요. 확률은 보정된 성공 확률이 아닙니다.</p>
<p>프레임 축은 초 단위가 아닙니다. 이 자료는 저장된 특징의 관찰용 결과이며 실시간 GR00T 또는 로봇 제어를 실행하지 않습니다.</p></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path)
    parser.add_argument("--episode", type=int)
    parser.add_argument(
        "--source-fps", type=float, help="original video FPS, NOT hidden sample FPS"
    )
    args = parser.parse_args()
    options = (args.boundaries, args.episode, args.source_fps)
    if any(v is not None for v in options) and not all(v is not None for v in options):
        parser.error("boundaries, episode, source-fps must be provided together")
    try:
        records = [
            json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()
        ]
        boundaries = None
        if args.boundaries:
            if not math.isfinite(args.source_fps) or args.source_fps <= 0:
                raise ValueError("source-fps must be positive and finite")
            with args.boundaries.open(newline="") as source:
                rows = [r for r in csv.DictReader(source) if int(r["episode"]) == args.episode]
            if len(rows) != 1:
                raise ValueError("expected exactly one matching episode in boundaries CSV")
            boundaries = [float(rows[0][f"boundary_{i}_s"]) * args.source_fps for i in range(1, 5)]
        document = render(records, boundaries)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as target:
            target.write(document)
        print(args.output.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
