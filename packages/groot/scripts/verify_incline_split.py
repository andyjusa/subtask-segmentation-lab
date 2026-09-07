from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> None:
    args = parse_args()
    split = json.loads((args.output_dir / "split.json").read_text())
    records = [
        json.loads(line) for line in (args.output_dir / "manifest.jsonl").read_text().splitlines()
    ]
    train = set(split["train"])
    validation = set(split["validation"])
    assert len(train) == 40
    assert len(validation) == 10
    assert not train & validation
    assert train | validation == set(range(50))
    assert len(records) == 250

    by_episode: dict[int, list[dict]] = defaultdict(list)
    codecs: Counter[str] = Counter()
    frame_rates: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    duration_errors = []
    frame_errors = []
    total_bytes = 0
    for record in records:
        by_episode[record["episode"]].append(record)
        path = args.output_dir / record["output_video"]
        assert path.is_file(), path
        total_bytes += path.stat().st_size
        details = probe(path)
        stream = details["streams"][0]
        codecs[stream["codec_name"]] += 1
        frame_rates[stream["avg_frame_rate"]] += 1
        resolutions[f"{stream['width']}x{stream['height']}"] += 1
        actual_duration = float(details["format"]["duration"])
        duration_errors.append(abs(actual_duration - record["duration_s"]))
        if stream.get("nb_frames"):
            actual_frames = int(stream["nb_frames"])
            expected_frames = record["end_frame"] - record["start_frame"]
            frame_errors.append(abs(actual_frames - expected_frames))

    for episode, episode_records in by_episode.items():
        episode_records.sort(key=lambda record: record["stage"])
        assert [record["stage"] for record in episode_records] == list(range(5))
        expected_split = "train" if episode in train else "validation"
        assert {record["split"] for record in episode_records} == {expected_split}
        assert episode_records[0]["start_frame"] == 0
        for left, right in itertools.pairwise(episode_records):
            assert left["end_frame"] == right["start_frame"]
            assert abs(left["end_s"] - right["start_s"]) < 1e-9
        assert all(record["duration_s"] > 0 for record in episode_records)

    assert len(by_episode) == 50
    assert codecs == {"h264": 250}
    assert frame_rates == {"30/1": 250}
    assert resolutions == {"640x480": 250}
    assert max(duration_errors) <= 0.05
    assert not frame_errors or max(frame_errors) <= 1
    report = {
        "episodes": len(by_episode),
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "clips": len(records),
        "train_clips": sum(record["split"] == "train" for record in records),
        "validation_clips": sum(record["split"] == "validation" for record in records),
        "codec": dict(codecs),
        "frame_rate": dict(frame_rates),
        "resolution": dict(resolutions),
        "max_duration_error_s": max(duration_errors),
        "max_frame_error": max(frame_errors) if frame_errors else None,
        "total_size_bytes": total_bytes,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
