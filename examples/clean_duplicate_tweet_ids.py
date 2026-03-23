#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def deduplicate_tweet_ids(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    seen_ids: set[str] = set()
    total_lines = 0
    kept_lines = 0
    removed_lines = 0

    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for raw_line in infile:
            total_lines += 1
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                outfile.write(raw_line)
                kept_lines += 1
                continue

            tweet_id = record.get("tweetID")

            if tweet_id is None:
                outfile.write(raw_line)
                kept_lines += 1
                continue

            if tweet_id in seen_ids:
                removed_lines += 1
                continue

            seen_ids.add(tweet_id)
            outfile.write(raw_line)
            kept_lines += 1

    return total_lines, kept_lines, removed_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicate tweetIDs from a JSONL file while keeping the first occurrence."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="examples/data/NTSO-T.json",
        help="Path to the input JSONL file (default: examples/data/NTSO-T.json)",
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Path to write cleaned output. If omitted, writes '<input>.dedup'.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the input file directly.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.in_place:
        temp_output_path = input_path.with_suffix(input_path.suffix + ".tmp")
        total, kept, removed = deduplicate_tweet_ids(input_path, temp_output_path)
        temp_output_path.replace(input_path)
        output_path = input_path
    else:
        output_path = Path(args.output_file) if args.output_file else input_path.with_suffix(input_path.suffix + ".dedup")
        total, kept, removed = deduplicate_tweet_ids(input_path, output_path)

    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total lines: {total}")
    print(f"Kept lines:  {kept}")
    print(f"Removed:     {removed}")


if __name__ == "__main__":
    main()