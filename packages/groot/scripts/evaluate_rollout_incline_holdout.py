"""Export reviewed sub-task labels and videos for one real Incline rollout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

STAGE_NAMES = (
    "zero_weights",
    "one_weight",
    "two_weights",
    "three_weights",
    "four_weights_terminal",
)
STAGE_COLORS = ("red", "orange", "yellow", "lime", "deepskyblue")


@dataclass(frozen=True)
class Boundary:
    stage: int
    frame: int
    time_seconds: float


def load_boundaries(path: Path) -> tuple[dict, list[Boundary]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    boundaries = [
        Boundary(
            stage=int(item["stage"]),
            frame=int(item["frame"]),
            time_seconds=float(item["time_seconds"]),
        )
        for item in payload["successful_boundaries"]
    ]
    if [item.stage for item in boundaries] != [1, 2, 3, 4]:
        raise ValueError("successful boundaries must advance through stages 1, 2, 3, 4")
    if any(right.frame <= left.frame for left, right in pairwise(boundaries)):
        raise ValueError("successful boundary frames must be strictly increasing")
    return payload, boundaries


def stage_at_frame(frame: int, boundaries: list[Boundary]) -> int:
    return sum(frame >= item.frame for item in boundaries)


def stage_ranges(frame_count: int, boundaries: list[Boundary]) -> list[tuple[int, int]]:
    starts = [0, *(item.frame for item in boundaries)]
    ends = [*(item.frame for item in boundaries), frame_count]
    return list(zip(starts, ends, strict=True))


def write_frame_labels(
    path: Path,
    frame_count: int,
    source_fps: float,
    sample_fps: float,
    boundaries: list[Boundary],
) -> int:
    sample_count = math.ceil((frame_count / source_fps) * sample_fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_index", "source_frame", "time_seconds", "stage", "stage_name"),
        )
        writer.writeheader()
        for sample_index in range(sample_count):
            time_seconds = sample_index / sample_fps
            source_frame = min(frame_count - 1, round(time_seconds * source_fps))
            stage = stage_at_frame(source_frame, boundaries)
            writer.writerow(
                {
                    "sample_index": sample_index,
                    "source_frame": source_frame,
                    "time_seconds": f"{time_seconds:.6f}",
                    "stage": stage,
                    "stage_name": STAGE_NAMES[stage],
                }
            )
    return sample_count


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def render_stage_videos(
    video_path: Path,
    output_dir: Path,
    source_fps: float,
    frame_count: int,
    boundaries: list[Boundary],
) -> list[Path]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to render stage videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for stage, (start_frame, end_frame) in enumerate(stage_ranges(frame_count, boundaries)):
        output = output_dir / f"stage-{stage}-{STAGE_NAMES[stage]}.mp4"
        start = start_frame / source_fps
        duration = (end_frame - start_frame) / source_fps
        run_command(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.6f}",
                "-i",
                str(video_path),
                "-t",
                f"{duration:.6f}",
                "-vf",
                f"drawbox=x=0:y=0:w=iw:h=ih:color={STAGE_COLORS[stage]}:t=10",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(output),
            ]
        )
        outputs.append(output)
    concat_file = output_dir / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{output.name}'\n" for output in outputs), encoding="utf-8"
    )
    review_video = output_dir.parent / "rollout-subtasks-color-bordered.mp4"
    run_command(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(review_video),
        ]
    )
    return [*outputs, review_video]


def write_report(
    path: Path,
    payload: dict,
    boundaries: list[Boundary],
    sample_count: int,
) -> None:
    failed = payload["failed_attempts"]
    assisted = payload["human_repositioning"]
    rows = []
    for stage, (start_frame, end_frame) in enumerate(
        stage_ranges(int(payload["frame_count"]), boundaries)
    ):
        start = start_frame / float(payload["fps"])
        end = end_frame / float(payload["fps"])
        rows.append(
            f"| {stage} | {STAGE_NAMES[stage]} | {start:.2f}–{end:.2f}초 | "
            f"{start_frame}–{end_frame - 1} | {STAGE_COLORS[stage]} |"
        )
    boundary_rows = [
        f"| {item.stage} | {item.time_seconds:.3f}초 | {item.frame} | "
        f"{payload['successful_boundaries'][index]['evidence']} |"
        for index, item in enumerate(boundaries)
    ]
    path.write_text(
        "\n".join(
            [
                "# 실제 GR00T Incline rollout sub-task 분리",
                "",
                f"- 데이터: `{payload['dataset']}` episode 0",
                f"- 지시문: `{payload['task']}`",
                "- 기준: 수레 위에 안정적으로 남아 있는 금속 추 개수",
                "- 판정: 실제 외부 카메라 영상 + 오른쪽 그리퍼 상태를 함께 검토",
                "",
                "## 분리 결과",
                "",
                "| stage | 의미 | 구간 | 원본 프레임 | 영상 테두리 |",
                "|---:|---|---:|---:|---|",
                *rows,
                "",
                "## 성공 경계",
                "",
                "| 진입 stage | 시각 | 프레임 | 근거 |",
                "|---:|---:|---:|---|",
                *boundary_rows,
                "",
                "## 실패와 보조 개입",
                "",
                f"- 실패한 grasp/release 후보: {len(failed)}회 — "
                + ", ".join(f"{item['time_seconds']:.2f}초" for item in failed),
                f"- 사람이 남은 추를 다시 잡기 쉬운 위치로 옮긴 구간: {len(assisted)}회 — "
                + ", ".join(f"{item['time_seconds']:.2f}초" for item in assisted),
                "- 사람은 수레에 추를 넣지 않았다. 성공한 네 번의 적재는 정책 팔이 수행했다.",
                "- 따라서 이 episode는 완전 무개입 rollout이 아니라 source-object reposition 보조가 있는 rollout이다.",
                "",
                "## 사용 범위",
                "",
                f"- 5 FPS 학습/평가용 label {sample_count}개를 CSV로 내보냈다.",
                "- 기존 5단계 Incline probe와는 label 의미가 달라 점수를 직접 비교하지 않는다.",
                "- 같은 지시문의 rollout을 더 모은 뒤 episode 단위 train/validation split으로 평가해야 한다.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--boundaries",
        type=Path,
        default=Path("reports/rollout_incline_20260903/boundaries.json"),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(
            "data/rollout_Incline_20260903_20260903_190043/videos/"
            "observation.images.follower_d455f/chunk-000/file-000.mp4"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/rollout_incline_20260903"))
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload, boundaries = load_boundaries(args.boundaries)
    frame_count = int(payload["frame_count"])
    source_fps = float(payload["fps"])
    sample_count = write_frame_labels(
        args.output_dir / "frame-labels-5fps.csv",
        frame_count,
        source_fps,
        args.sample_fps,
        boundaries,
    )
    if not args.metadata_only:
        if not args.video.is_file():
            raise FileNotFoundError(args.video)
        render_stage_videos(
            args.video,
            args.output_dir / "segments",
            source_fps,
            frame_count,
            boundaries,
        )
    write_report(args.output_dir / "REPORT.md", payload, boundaries, sample_count)
    print(
        json.dumps(
            {
                "successful_boundary_count": len(boundaries),
                "failed_attempt_count": len(payload["failed_attempts"]),
                "human_reposition_event_count": len(payload["human_repositioning"]),
                "sample_count_5fps": sample_count,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
