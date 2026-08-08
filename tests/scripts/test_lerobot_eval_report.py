# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from lerobot.eval_report import build_comparison_report, load_episodes, write_comparison_report


def _write_eval(
    path,
    successes,
    *,
    seeds=True,
    task_group="libero_spatial",
    task_id=0,
    protocol=None,
    video_paths=None,
):
    metrics = {
        "successes": successes,
        "sum_rewards": [float(success) for success in successes],
        "max_rewards": [float(success) for success in successes],
        "video_paths": video_paths
        if video_paths is not None
        else [f"videos/episode_{index}.mp4" for index in range(min(2, len(successes)))],
        "predicted_video_paths": [],
    }
    if seeds:
        metrics["seeds"] = list(range(1000, 1000 + len(successes)))
    payload = {
        "per_task": [{"task_group": task_group, "task_id": task_id, "metrics": metrics}],
        "per_group": {},
        "overall": {},
    }
    path.mkdir()
    (path / "eval_info.json").write_text(json.dumps(payload))
    if protocol is not None:
        (path / "eval_manifest.json").write_text(json.dumps({"schema_version": 1, "protocol": protocol}))


def test_build_report_detects_clear_improvement(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_eval(baseline, [False] * 20)
    _write_eval(candidate, [True] * 20)

    report = build_comparison_report(baseline, candidate, bootstrap_samples=200)

    assert report["decision"]["status"] == "improved"
    assert report["summary"]["delta"]["success_rate"] == 1.0
    assert report["summary"]["delta"]["success_rate_ci"] == [1.0, 1.0]
    assert report["pairing"]["modes"] == {"seed": 20}
    assert report["episodes"][0]["baseline"]["video_path"] == "videos/episode_0.mp4"


def test_old_artifacts_fall_back_to_episode_index(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_eval(baseline, [False, True], seeds=False)
    _write_eval(candidate, [True, True], seeds=False)

    report = build_comparison_report(baseline, candidate, bootstrap_samples=100, min_episodes=2)

    assert report["pairing"]["modes"] == {"episode_index": 2}
    assert report["summary"]["matched_episodes"] == 2


def test_unmatched_episode_sets_do_not_pass_gate(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_eval(baseline, [False] * 20)
    _write_eval(candidate, [True] * 19)

    report = build_comparison_report(baseline, candidate, bootstrap_samples=100, min_episodes=10)

    assert report["decision"]["status"] == "not_comparable"
    assert len(report["pairing"]["unmatched"]) == 1


def test_different_protocols_do_not_pass_gate(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_eval(baseline, [False] * 20, protocol={"environment": "libero_spatial", "seed": 1000})
    _write_eval(candidate, [True] * 20, protocol={"environment": "libero_object", "seed": 1000})

    report = build_comparison_report(baseline, candidate, bootstrap_samples=100)

    assert report["decision"]["status"] == "not_comparable"
    assert report["provenance"]["protocol_comparison"]["status"] == "different"


def test_legacy_single_task_schema_is_supported(tmp_path):
    artifact = tmp_path / "eval_info.json"
    artifact.write_text(
        json.dumps(
            {
                "per_episode": [
                    {
                        "episode_ix": 0,
                        "sum_reward": 1.0,
                        "max_reward": 1.0,
                        "success": True,
                        "seed": 42,
                    }
                ],
                "aggregated": {},
            }
        )
    )

    _, episodes = load_episodes(artifact)

    assert episodes[0].task_group == "default"
    assert episodes[0].seed == 42


def test_video_paths_are_made_relative_to_the_eval_artifact(tmp_path):
    artifact = tmp_path / "baseline"
    _write_eval(artifact, [True], video_paths=["old/output/prefix/eval_episode_0.mp4"])
    video = artifact / "videos" / "libero_spatial_0" / "eval_episode_0.mp4"
    video.parent.mkdir(parents=True)
    video.touch()

    _, episodes = load_episodes(artifact)

    assert episodes[0].video_path == "videos/libero_spatial_0/eval_episode_0.mp4"


def test_report_writer_emits_json_and_markdown(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    output = tmp_path / "report"
    _write_eval(baseline, [False, True])
    _write_eval(candidate, [True, True])
    report = build_comparison_report(baseline, candidate, bootstrap_samples=100, min_episodes=2)

    json_path, markdown_path = write_comparison_report(report, output)

    assert json.loads(json_path.read_text())["schema_version"] == 1
    assert "# Simulation evaluation comparison" in markdown_path.read_text()
