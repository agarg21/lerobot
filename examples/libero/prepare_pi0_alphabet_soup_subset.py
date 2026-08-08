#!/usr/bin/env python

"""Materialize the pinned pi0 alphabet-soup episodes with corrected local shard metadata."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


TARGET_TASK_INDEX = 24
TARGET_TASK = "pick up the alphabet soup and place it in the basket"
EPISODE_TO_FILE = {
    813: 213,
    818: 214,
    825: 215,
    850: 221,
    852: 221,
    857: 222,
    860: 223,
    876: 226,
    877: 226,
    887: 229,
    905: 232,
    908: 233,
    934: 238,
    945: 241,
    978: 248,
    983: 249,
    984: 249,
    985: 249,
    1002: 253,
    1011: 255,
    1020: 256,
    1024: 257,
    1035: 259,
    1074: 268,
    1075: 268,
    1100: 273,
    1102: 274,
    1108: 275,
    1131: 280,
    1143: 283,
    1148: 284,
    1151: 285,
    1152: 285,
    1166: 288,
    1171: 289,
    1187: 293,
    1205: 296,
    1219: 299,
    1224: 300,
    1225: 301,
    1233: 302,
    1242: 304,
    1243: 304,
    1245: 305,
}


def _replace_int_column(table: pa.Table, name: str, updates: dict[int, int]) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise KeyError(f"Missing metadata column: {name}")
    values = table[name].to_pylist()
    for row_index, value in updates.items():
        values[row_index] = value
    return table.set_column(column_index, name, pa.array(values, type=table[name].type))


def prepare_subset(repo_id: str, revision: str, output_root: Path) -> dict:
    data_paths = sorted(
        {f"data/chunk-000/file-{file_index:03d}.parquet" for file_index in EPISODE_TO_FILE.values()}
    )
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=output_root,
        allow_patterns=["meta/**", *data_paths],
    )

    observed: dict[int, int] = {}
    for relative_path in data_paths:
        table = pq.read_table(
            output_root / relative_path,
            columns=["episode_index", "task_index"],
            filters=[("task_index", "=", TARGET_TASK_INDEX)],
        )
        for episode_index in set(table["episode_index"].to_pylist()):
            if episode_index in EPISODE_TO_FILE:
                observed[episode_index] = int(relative_path[-11:-8])

    if observed != EPISODE_TO_FILE:
        missing = sorted(set(EPISODE_TO_FILE) - set(observed))
        mismatched = {
            episode: (EPISODE_TO_FILE[episode], observed.get(episode))
            for episode in observed
            if observed[episode] != EPISODE_TO_FILE[episode]
        }
        raise RuntimeError(f"Pinned task shard verification failed; missing={missing}, mismatched={mismatched}")

    episodes_path = output_root / "meta/episodes/chunk-000/file-000.parquet"
    episodes_table = pq.read_table(episodes_path)
    episode_rows = episodes_table["episode_index"].to_pylist()
    row_by_episode = {episode: row for row, episode in enumerate(episode_rows)}
    missing_metadata = sorted(set(EPISODE_TO_FILE) - set(row_by_episode))
    if missing_metadata:
        raise RuntimeError(f"Pinned episodes missing from metadata: {missing_metadata}")

    tasks = episodes_table["tasks"].to_pylist()
    bad_tasks = [episode for episode in EPISODE_TO_FILE if tasks[row_by_episode[episode]] != [TARGET_TASK]]
    if bad_tasks:
        raise RuntimeError(f"Pinned episodes no longer have the expected task text: {bad_tasks}")

    file_updates = {row_by_episode[episode]: file_index for episode, file_index in EPISODE_TO_FILE.items()}
    chunk_updates = {row_by_episode[episode]: 0 for episode in EPISODE_TO_FILE}
    episodes_table = _replace_int_column(episodes_table, "data/file_index", file_updates)
    episodes_table = _replace_int_column(episodes_table, "data/chunk_index", chunk_updates)
    pq.write_table(episodes_table, episodes_path)

    payload = {
        "dataset": repo_id,
        "dataset_revision": revision,
        "target_task_index": TARGET_TASK_INDEX,
        "target_task": TARGET_TASK,
        "episodes": sorted(EPISODE_TO_FILE),
        "data_files": data_paths,
        "episode_to_verified_file_index": {str(k): v for k, v in EPISODE_TO_FILE.items()},
        "metadata_repair": "local data/file_index values replaced for the 44 selected episodes",
    }
    (output_root / "subset_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare_subset(args.repo_id, args.revision, args.output_root)
    print(
        f"Prepared {len(payload['episodes'])} verified episodes across "
        f"{len(payload['data_files'])} data files in {args.output_root}"
    )


if __name__ == "__main__":
    main()
