#!/usr/bin/env python3
"""Check consistency between raw, translated, and embedded datasets.

The checker understands the formats produced by ``offline_preprocessing.py``:
raw JSONL files, raw SocialED NumPy arrays, translation checkpoints, and
model-specific embedding checkpoints.

Examples:
    python examples/data/check_consistency.py
    python examples/data/check_consistency.py --details --strict
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


SUPPORTED_SUFFIXES = {".json", ".npy"}


@dataclass
class DatasetInfo:
    """Identifiers and metadata extracted from one dataset file."""

    ids: list[str]
    timestamps: list[Any]
    clusters: list[Any]

    @property
    def tweet_count(self) -> int:
        return len(self.ids)

    @property
    def duplicate_ids(self) -> list[str]:
        return sorted(identifier for identifier, count in Counter(self.ids).items() if count > 1)

    @property
    def cluster_count(self) -> int:
        return len({str(cluster) for cluster in self.clusters if cluster is not None and str(cluster) != ""})


def canonical_id(value: Any) -> str:
    """Use the same representation as the preprocessing pipeline."""
    return str(value)


def parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return datetime.fromtimestamp(float(value))
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    for parser in (
        lambda item: datetime.fromisoformat(item.replace("Z", "+00:00")),
        lambda item: datetime.strptime(item.replace("/", "-"), "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            return parser(text)
        except ValueError:
            pass
    return None


def json_dataset(path: Path) -> DatasetInfo:
    ids: list[str] = []
    timestamps: list[Any] = []
    clusters: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            if "tweetID" in record:
                ids.append(canonical_id(record["tweetID"]))
            else:
                ids.append("")
            timestamps.append(record.get("timestamp"))
            clusters.append(record.get("clusterNo") or record.get("topic"))
    return DatasetInfo(ids, timestamps, clusters)


def numpy_dataset(path: Path) -> DatasetInfo:
    data = np.load(path, allow_pickle=True)
    if data.ndim != 2 or data.shape[1] <= 7:
        raise ValueError(f"{path}: expected a 2-D SocialED array with at least 8 columns, got {data.shape}")
    # SocialED columns: tweet_id, text, event_id, ..., created_at at index 7.
    return DatasetInfo(
        [canonical_id(value) for value in data[:, 0]],
        data[:, 7].tolist(),
        data[:, 2].tolist(),
    )


def load_raw(path: Path) -> DatasetInfo:
    if path.suffix.lower() == ".json":
        return json_dataset(path)
    if path.suffix.lower() == ".npy":
        return numpy_dataset(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_checkpoint(path: Path) -> tuple[list[str], dict[str, Any]]:
    value = np.load(path, allow_pickle=True)
    if value.shape != ():
        raise ValueError(f"{path}: expected a scalar NumPy object checkpoint, got shape {value.shape}")
    payload = value.item()
    if not isinstance(payload, dict) or "tweet_ids" not in payload:
        raise ValueError(f"{path}: expected a checkpoint dictionary containing 'tweet_ids'")
    return [canonical_id(identifier) for identifier in payload["tweet_ids"]], payload


def files_by_stem(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES}


def compare_ids(expected: Iterable[str], actual: Iterable[str]) -> tuple[bool, list[str], list[str]]:
    expected_set, actual_set = set(expected), set(actual)
    expected_counts = Counter(expected)
    duplicates = set(identifier for identifier, count in expected_counts.items() if count > 1)
    expected_set = expected_set - duplicates # Duplicates are ignored in the comparison, as they are not valid identifiers.
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    return expected_set == actual_set, missing, extra


def format_span(info: DatasetInfo) -> str:
    dates = [parsed for value in info.timestamps if (parsed := parse_timestamp(value)) is not None]
    if not dates:
        return "n/a"
    return f"{min(dates).isoformat(sep=' ')} .. {max(dates).isoformat(sep=' ')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).parent / "raw_data")
    parser.add_argument("--translated-dir", type=Path, default=Path(__file__).parent / "translated_no_emojis")
    parser.add_argument("--embedded-dir", type=Path, default=Path(__file__).parent / "embedded")
    parser.add_argument("--details", action="store_true", help="show missing/extra IDs for failed comparisons")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any consistency check fails")
    args = parser.parse_args()

    args.details = False
    
    raw_files = files_by_stem(args.raw_dir)
    translated_files = files_by_stem(args.translated_dir) if args.translated_dir.is_dir() else {}
    models = sorted(path.name for path in args.embedded_dir.iterdir() if path.is_dir()) if args.embedded_dir.is_dir() else []
    failures = 0
    failed_datasets: set[str] = set()

    print(f"Raw datasets: {args.raw_dir}")
    print(f"Embedding models discovered: {', '.join(models) or 'none'}")
    print()
    print("DATASET STATISTICS")
    print("dataset | tweets | time span | clusters | duplicate tweet_ids")
    raw_infos: dict[str, DatasetInfo] = {}
    for name, path in sorted(raw_files.items()):
        try:
            info = raw_infos[name] = load_raw(path)
            duplicates = len(info.duplicate_ids)
            print(f"{name} | {info.tweet_count} | {format_span(info)} | {info.cluster_count} | {duplicates}")
        except (OSError, ValueError, TypeError) as error:
            failures += 1
            failed_datasets.add(name)
            print(f"{name} | ERROR: {error}")

    print("\nCONSISTENCY")
    for name, raw in sorted(raw_infos.items()):
        translated = translated_files.get(name)
        if translated is None:
            failures += 1
            failed_datasets.add(name)
            print(f"{name}: translated MISSING")
        else:
            try:
                translated_ids, _ = load_checkpoint(translated)
                same, missing, extra = compare_ids(raw.ids, translated_ids)
                status = "OK" if same else "FAIL"
                if not same:
                    failures += 1
                    failed_datasets.add(name)
                print(f"{name}: translated {status} (translated_tweets={len(translated_ids)}, num_missing_ids={len(missing)}, num_extra_ids={len(extra)})")
                if args.details and not same:
                    print(f"  missing={missing[:10]} extra={extra[:10]}")
            except (OSError, ValueError, TypeError, AttributeError) as error:
                failures += 1
                failed_datasets.add(name)
                print(f"{name}: translated ERROR: {error}")

        for model in models:
            embedding_path = args.embedded_dir / model / f"{name}.npy"
            if not embedding_path.exists():
                failures += 1
                failed_datasets.add(name)
                print(f"{name}: embedding/{model} MISSING")
                continue
            try:
                embedding_ids, payload = load_checkpoint(embedding_path)
                same, missing, extra = compare_ids(raw.ids, embedding_ids)
                has_embeddings = len(payload.get("embeddings", [])) == len(embedding_ids)
                status = "OK" if same and has_embeddings else "FAIL"
                if status == "FAIL":
                    failures += 1
                    failed_datasets.add(name)
                print(f"{name}: embedding/{model} {status} (embedded_tweets={len(embedding_ids)}, num_missing_ids={len(missing)}, num_extra_ids={len(extra)}")
                if args.details and not same:
                    print(f"  missing={missing[:10]} extra={extra[:10]}" )
            except (OSError, ValueError, TypeError, AttributeError) as error:
                failures += 1
                failed_datasets.add(name)
                print(f"{name}: embedding/{model} ERROR: {error}")

    orphan_translated = sorted(set(translated_files) - set(raw_files))
    if orphan_translated:
        print(f"\nTranslated datasets without raw source: {', '.join(orphan_translated)}")
    if failures:
        print(f"\nChecks with failures: {failures}")
        print(f"Failed datasets: {', '.join(sorted(set(failed_datasets)))}")
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
