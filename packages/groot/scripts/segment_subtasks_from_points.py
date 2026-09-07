from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

GRID_ROWS = 3
GRID_COLS = 4
MOTION_FEATURES_PER_CELL = 5
GLOBAL_MOTION_FEATURES = 8
GLOBAL_POSITION_FEATURES = 4
FEATURE_DIM = (
    GRID_ROWS * GRID_COLS * (MOTION_FEATURES_PER_CELL + 1)
    + GLOBAL_MOTION_FEATURES
    + GLOBAL_POSITION_FEATURES
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--reference-boundaries", default="")
    parser.add_argument("--overlay-input", type=Path)
    return parser.parse_args()


def detect_points(gray: np.ndarray, existing: np.ndarray, limit: int) -> np.ndarray:
    mask = np.full_like(gray, 255)
    for point in existing:
        cv2.circle(mask, tuple(np.round(point).astype(int)), 9, 0, -1)
    points = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        maxCorners=limit,
        qualityLevel=0.015,
        minDistance=7,
        blockSize=7,
    )
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    return points.reshape(-1, 2).astype(np.float32)


def flow_features(
    old_points: np.ndarray,
    new_points: np.ndarray,
    width: int,
    height: int,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> np.ndarray:
    if len(new_points) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    displacement = new_points - old_points
    speed = np.linalg.norm(displacement, axis=1)
    moving = speed >= 0.2
    selected_points = new_points[moving]
    selected_flow = displacement[moving]
    selected_speed = speed[moving]
    if len(selected_points) == 0:
        selected_points = np.empty((0, 2), dtype=np.float32)
        selected_flow = np.empty((0, 2), dtype=np.float32)
        selected_speed = np.empty(0, dtype=np.float32)

    values = []
    for row in range(rows):
        for col in range(cols):
            inside = (
                (selected_points[:, 0] >= col * width / cols)
                & (selected_points[:, 0] < (col + 1) * width / cols)
                & (selected_points[:, 1] >= row * height / rows)
                & (selected_points[:, 1] < (row + 1) * height / rows)
            )
            if not inside.any():
                values.extend((0.0, 0.0, 0.0, 0.0, 0.0))
                continue
            cell_flow = selected_flow[inside]
            cell_speed = selected_speed[inside]
            values.extend(
                (
                    float(inside.mean()),
                    float(np.median(cell_flow[:, 0])),
                    float(np.median(cell_flow[:, 1])),
                    float(np.median(cell_speed)),
                    float(np.percentile(cell_speed, 90)),
                )
            )
    if len(selected_points):
        weights = selected_speed / max(float(selected_speed.sum()), 1e-6)
        centroid = (selected_points * weights[:, None]).sum(axis=0)
        values.extend(
            (
                float(len(selected_points) / max(len(new_points), 1)),
                float(np.median(selected_flow[:, 0])),
                float(np.median(selected_flow[:, 1])),
                float(np.median(selected_speed)),
                float(np.percentile(selected_speed, 90)),
                float(centroid[0] / width),
                float(centroid[1] / height),
                float(selected_speed.sum()),
            )
        )
    else:
        values.extend((0.0,) * GLOBAL_MOTION_FEATURES)
    for row in range(rows):
        for col in range(cols):
            inside = (
                (new_points[:, 0] >= col * width / cols)
                & (new_points[:, 0] < (col + 1) * width / cols)
                & (new_points[:, 1] >= row * height / rows)
                & (new_points[:, 1] < (row + 1) * height / rows)
            )
            values.append(float(inside.mean()))
    normalized_positions = new_points / np.asarray([width, height], dtype=np.float32)
    values.extend(
        (
            float(normalized_positions[:, 0].mean()),
            float(normalized_positions[:, 1].mean()),
            float(normalized_positions[:, 0].std()),
            float(normalized_positions[:, 1].std()),
        )
    )
    return np.asarray(values, dtype=np.float32)


def extract_point_flow(video: Path) -> tuple[np.ndarray, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first = capture.read()
    if not ok:
        raise RuntimeError("Could not read first frame")
    previous_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    points = detect_points(previous_gray, np.empty((0, 2)), 300)
    features = [np.zeros(FEATURE_DIM, dtype=np.float32)]
    frame_index = 1
    lk_params = {
        "winSize": (31, 31),
        "maxLevel": 4,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    }
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            old = points.copy()
            if len(old):
                new, status, _ = cv2.calcOpticalFlowPyrLK(
                    previous_gray, gray, old.reshape(-1, 1, 2), None, **lk_params
                )
                backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, previous_gray, new, None, **lk_params
                )
                new = new.reshape(-1, 2)
                forward_backward = np.linalg.norm(
                    old.reshape(-1, 1, 2) - backward, axis=2
                ).reshape(-1)
                valid = (
                    status.reshape(-1).astype(bool)
                    & backward_status.reshape(-1).astype(bool)
                    & (forward_backward < 1.5)
                    & (new[:, 0] >= 0)
                    & (new[:, 0] < width)
                    & (new[:, 1] >= 0)
                    & (new[:, 1] < height)
                )
                old_valid, new_valid = old[valid], new[valid]
                features.append(flow_features(old_valid, new_valid, width, height))
                points = new_valid
            else:
                features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            if frame_index % 15 == 0 and len(points) < 300:
                fresh = detect_points(gray, points, 300 - len(points))
                if len(fresh):
                    points = np.concatenate([points, fresh])
            previous_gray = gray
            frame_index += 1
    finally:
        capture.release()
    return np.stack(features), fps, (width, height)


def novelty_curve(features: np.ndarray, fps: float) -> np.ndarray:
    smooth = gaussian_filter1d(features, sigma=max(1.0, fps * 0.15), axis=0)
    median = np.median(smooth, axis=0)
    scale = np.median(np.abs(smooth - median), axis=0) * 1.4826
    usable = scale > 1e-4
    normalized = np.zeros_like(smooth)
    normalized[:, usable] = (smooth[:, usable] - median[usable]) / scale[usable]

    window = max(1, round(fps))
    cumulative = np.vstack([np.zeros((1, normalized.shape[1])), normalized.cumsum(0)])
    novelty = np.zeros(len(features), dtype=np.float32)
    for index in range(window, len(features) - window):
        before = (cumulative[index] - cumulative[index - window]) / window
        after = (cumulative[index + window] - cumulative[index]) / window
        novelty[index] = np.linalg.norm(after - before) / np.sqrt(normalized.shape[1])
    novelty = gaussian_filter1d(novelty, sigma=max(1.0, fps * 0.2))
    return novelty


def select_boundaries(
    novelty: np.ndarray, fps: float, segments: int
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(novelty)
    boundary_count = segments - 1
    min_segment = round(6.0 * fps)
    candidates = []
    for index in range(min_segment, frame_count - min_segment):
        radius = max(1, round(0.5 * fps))
        local = novelty[index - radius : index + radius + 1]
        if novelty[index] >= local.max():
            candidates.append(index)
    candidates = np.asarray(candidates, dtype=np.int32)
    if len(candidates) < boundary_count:
        raise RuntimeError("Not enough point-motion boundary candidates")

    expected = frame_count / segments
    states: dict[tuple[int, int], tuple[float, list[int]]] = {}
    for candidate_index, frame in enumerate(candidates):
        duration_penalty = abs(frame - expected) / expected
        states[(1, candidate_index)] = (
            float(novelty[frame] - 0.25 * duration_penalty),
            [int(frame)],
        )
    for count in range(2, boundary_count + 1):
        for candidate_index, frame in enumerate(candidates):
            best = None
            for previous_index in range(candidate_index):
                previous = states.get((count - 1, previous_index))
                if previous is None:
                    continue
                previous_frame = previous[1][-1]
                if frame - previous_frame < min_segment:
                    continue
                duration_penalty = abs((frame - previous_frame) - expected) / expected
                score = previous[0] + float(novelty[frame]) - 0.25 * duration_penalty
                if best is None or score > best[0]:
                    best = (score, [*previous[1], int(frame)])
            if best is not None:
                states[(count, candidate_index)] = best

    valid = []
    for (count, _), state in states.items():
        if count != boundary_count:
            continue
        final_duration = frame_count - state[1][-1]
        if final_duration < min_segment:
            continue
        penalty = abs(final_duration - expected) / expected
        valid.append((state[0] - 0.25 * penalty, state[1]))
    if not valid:
        raise RuntimeError("Could not select a valid boundary sequence")
    best = max(valid, key=lambda item: item[0])
    return np.asarray(best[1], dtype=np.int32), candidates


def evaluate(predicted_s: np.ndarray, reference_s: np.ndarray) -> dict:
    if len(reference_s) == 0:
        return {}
    if len(predicted_s) != len(reference_s):
        raise ValueError("Predicted and reference boundary counts differ")
    errors = np.abs(predicted_s - reference_s)
    return {
        "reference_s": reference_s.tolist(),
        "absolute_errors_s": errors.tolist(),
        "mean_absolute_error_s": float(errors.mean()),
        "median_absolute_error_s": float(np.median(errors)),
        "within_1s": int((errors <= 1.0).sum()),
        "within_2s": int((errors <= 2.0).sum()),
    }


def render_video(
    source: Path,
    output: Path,
    boundaries: np.ndarray,
    fps: float,
    segments: int,
) -> None:
    capture = cv2.VideoCapture(str(source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    temporary = output.with_suffix(".tracking.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    colors = [
        (70, 190, 255),
        (70, 230, 100),
        (230, 190, 60),
        (230, 90, 160),
        (180, 100, 240),
    ]
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            stage = int(np.searchsorted(boundaries, frame_index, side="right"))
            color = colors[stage % len(colors)]
            cv2.rectangle(frame, (0, 0), (width, 38), color, -1)
            cv2.putText(
                frame,
                f"Point-only predicted subtask {stage + 1}/{segments}",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )
            y = height - 18
            cv2.line(frame, (12, y), (width - 12, y), (255, 255, 255), 3)
            for boundary in boundaries:
                x = 12 + round((width - 24) * boundary / frame_count)
                cv2.line(frame, (x, y - 8), (x, y + 8), (20, 20, 20), 2)
            cursor = 12 + round((width - 24) * frame_index / frame_count)
            cv2.circle(frame, (cursor, y), 5, color, -1)
            distance = np.min(np.abs(boundaries - frame_index)) if len(boundaries) else 9999
            if distance <= round(0.35 * fps):
                cv2.putText(
                    frame,
                    "POINT-MOTION BOUNDARY",
                    (width // 2 - 135, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(temporary),
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    temporary.unlink()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, fps, frame_size = extract_point_flow(args.input)
    novelty = novelty_curve(features, fps)
    boundaries, candidates = select_boundaries(novelty, fps, args.segments)
    predicted_s = boundaries / fps
    equal_s = np.arange(1, args.segments) * (len(features) / fps / args.segments)
    reference_s = np.asarray(
        [float(value) for value in args.reference_boundaries.split(",") if value],
        dtype=np.float32,
    )
    np.savez_compressed(
        args.output_dir / "point_flow_features.npz",
        features=features,
        novelty=novelty,
        boundaries=boundaries,
        candidates=candidates,
        fps=np.asarray(fps),
    )
    with (args.output_dir / "novelty.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frame", "timestamp_s", "novelty", "predicted_boundary"))
        boundary_set = set(boundaries.tolist())
        for frame, value in enumerate(novelty):
            writer.writerow((frame, frame / fps, float(value), frame in boundary_set))
    result = {
        "method": "point tracks only: 4x3 flow grid + causal-free change-point novelty",
        "input": str(args.input),
        "frames": len(features),
        "fps": fps,
        "frame_size": list(frame_size),
        "segments": args.segments,
        "predicted_boundaries_s": predicted_s.tolist(),
        "equal_duration_baseline_s": equal_s.tolist(),
        "point_method_evaluation": evaluate(predicted_s, reference_s),
        "equal_duration_evaluation": evaluate(equal_s, reference_s),
        "limitations": [
            "Single-episode pilot; no cross-episode generalization claim.",
            "The number of stages is supplied in advance.",
            "Change-point scoring is offline because it compares one second before and after.",
        ],
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    overlay_source = args.overlay_input or args.input
    render_video(
        overlay_source,
        args.output_dir / "point_only_subtask_boundaries.mp4",
        boundaries,
        fps,
        args.segments,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
