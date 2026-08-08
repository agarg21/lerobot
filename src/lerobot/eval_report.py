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

"""Build reproducible, paired comparisons from LeRobot simulation evaluations."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist, fmean
from typing import Any


@dataclass(frozen=True)
class EpisodeOutcome:
    """One policy outcome for one task episode."""

    task_group: str
    task_id: str
    episode_ix: int
    seed: int | None
    success: bool
    sum_reward: float
    max_reward: float
    video_path: str | None = None


@dataclass(frozen=True)
class EpisodePair:
    """Matched baseline and candidate outcomes for the same episode."""

    baseline: EpisodeOutcome
    candidate: EpisodeOutcome
    pairing: str


def resolve_eval_info(path: str | Path) -> Path:
    """Resolve an evaluation directory or JSON filename to ``eval_info.json``."""
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "eval_info.json"
    if not resolved.is_file():
        raise FileNotFoundError(f"Evaluation artifact does not exist: {resolved}")
    return resolved


def _display_eval_info(path: str | Path) -> Path:
    """Return the input path without forcing it to become machine-specific."""
    display_path = Path(path).expanduser()
    if display_path.is_dir():
        display_path = display_path / "eval_info.json"
    return display_path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_eval_manifest(eval_path: Path) -> tuple[Path, dict[str, Any]] | None:
    """Load the optional protocol manifest stored beside an evaluation result."""
    manifest_path = eval_path.parent / "eval_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open() as source:
        manifest = json.load(source)
    if "protocol" not in manifest:
        raise ValueError(f"Evaluation manifest has no protocol section: {manifest_path}")
    return manifest_path, manifest


def _coerce_episode(
    *,
    task_group: Any,
    task_id: Any,
    episode_ix: int,
    success: Any,
    sum_reward: Any,
    max_reward: Any,
    seed: Any = None,
    video_path: Any = None,
) -> EpisodeOutcome:
    return EpisodeOutcome(
        task_group=str(task_group),
        task_id=str(task_id),
        episode_ix=int(episode_ix),
        seed=None if seed is None else int(seed),
        success=bool(success),
        sum_reward=float(sum_reward),
        max_reward=float(max_reward),
        video_path=None if video_path is None else str(video_path),
    )


def _extract_task_episodes(payload: dict[str, Any]) -> list[EpisodeOutcome]:
    outcomes: list[EpisodeOutcome] = []
    for task in payload["per_task"]:
        metrics = task["metrics"]
        successes = metrics["successes"]
        sum_rewards = metrics["sum_rewards"]
        max_rewards = metrics["max_rewards"]
        seeds = metrics.get("seeds")
        video_paths = metrics.get("video_paths", [])

        lengths = {len(successes), len(sum_rewards), len(max_rewards)}
        if len(lengths) != 1:
            raise ValueError(
                f"Task {task['task_group']}/{task['task_id']} has inconsistent episode metric lengths."
            )
        if seeds is not None and len(seeds) != len(successes):
            raise ValueError(f"Task {task['task_group']}/{task['task_id']} has inconsistent seed count.")

        for episode_ix, (success, sum_reward, max_reward) in enumerate(
            zip(successes, sum_rewards, max_rewards, strict=True)
        ):
            outcomes.append(
                _coerce_episode(
                    task_group=task["task_group"],
                    task_id=task["task_id"],
                    episode_ix=episode_ix,
                    seed=None if seeds is None else seeds[episode_ix],
                    success=success,
                    sum_reward=sum_reward,
                    max_reward=max_reward,
                    video_path=video_paths[episode_ix] if episode_ix < len(video_paths) else None,
                )
            )
    return outcomes


def _extract_legacy_episodes(payload: dict[str, Any]) -> list[EpisodeOutcome]:
    return [
        _coerce_episode(
            task_group="default",
            task_id="0",
            episode_ix=episode.get("episode_ix", episode_ix),
            seed=episode.get("seed"),
            success=episode["success"],
            sum_reward=episode["sum_reward"],
            max_reward=episode["max_reward"],
            video_path=(payload.get("video_paths") or [])[episode_ix]
            if episode_ix < len(payload.get("video_paths") or [])
            else None,
        )
        for episode_ix, episode in enumerate(payload["per_episode"])
    ]


def load_episodes(path: str | Path) -> tuple[Path, list[EpisodeOutcome]]:
    """Load episode outcomes from current or legacy LeRobot evaluation output."""
    eval_path = resolve_eval_info(path)
    with eval_path.open() as source:
        payload = json.load(source)

    if "per_task" in payload:
        outcomes = _extract_task_episodes(payload)
    elif "per_episode" in payload:
        outcomes = _extract_legacy_episodes(payload)
    else:
        raise ValueError(f"Unsupported evaluation schema in {eval_path}")
    if not outcomes:
        raise ValueError(f"Evaluation artifact contains no episodes: {eval_path}")
    return eval_path, outcomes


def _group_episodes(outcomes: list[EpisodeOutcome]) -> dict[tuple[str, str], list[EpisodeOutcome]]:
    grouped: dict[tuple[str, str], list[EpisodeOutcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[(outcome.task_group, outcome.task_id)].append(outcome)
    return grouped


def _index_by_seed(outcomes: list[EpisodeOutcome]) -> dict[int, EpisodeOutcome]:
    indexed: dict[int, EpisodeOutcome] = {}
    for outcome in outcomes:
        if outcome.seed is None:
            raise ValueError("Cannot index an episode without a seed.")
        if outcome.seed in indexed:
            raise ValueError(f"Duplicate seed {outcome.seed} in task {outcome.task_group}/{outcome.task_id}.")
        indexed[outcome.seed] = outcome
    return indexed


def pair_episodes(
    baseline: list[EpisodeOutcome], candidate: list[EpisodeOutcome]
) -> tuple[list[EpisodePair], list[dict[str, Any]], dict[str, int]]:
    """Pair episodes by task and seed, falling back to episode index for older outputs."""
    baseline_groups = _group_episodes(baseline)
    candidate_groups = _group_episodes(candidate)
    pairs: list[EpisodePair] = []
    unmatched: list[dict[str, Any]] = []
    pairing_modes: dict[str, int] = defaultdict(int)

    for task_key in sorted(set(baseline_groups) | set(candidate_groups)):
        baseline_task = baseline_groups.get(task_key, [])
        candidate_task = candidate_groups.get(task_key, [])
        has_seeds = (
            baseline_task
            and candidate_task
            and all(outcome.seed is not None for outcome in baseline_task + candidate_task)
        )

        if has_seeds:
            mode = "seed"
            baseline_index = _index_by_seed(baseline_task)
            candidate_index = _index_by_seed(candidate_task)
        else:
            mode = "episode_index"
            baseline_index = {outcome.episode_ix: outcome for outcome in baseline_task}
            candidate_index = {outcome.episode_ix: outcome for outcome in candidate_task}

        shared = sorted(set(baseline_index) & set(candidate_index))
        for match_key in shared:
            pairs.append(
                EpisodePair(
                    baseline=baseline_index[match_key],
                    candidate=candidate_index[match_key],
                    pairing=mode,
                )
            )
            pairing_modes[mode] += 1

        for side, index, other in (
            ("baseline", baseline_index, candidate_index),
            ("candidate", candidate_index, baseline_index),
        ):
            for match_key in sorted(set(index) - set(other)):
                unmatched.append(
                    {
                        "side": side,
                        "task_group": task_key[0],
                        "task_id": task_key[1],
                        "pairing": mode,
                        "match_key": match_key,
                    }
                )

    return pairs, unmatched, dict(pairing_modes)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap_interval(
    deltas: list[float], *, confidence_level: float, samples: int, seed: int
) -> list[float]:
    """Return a deterministic percentile interval for the mean paired delta."""
    if not deltas:
        raise ValueError("At least one paired delta is required.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if samples < 1:
        raise ValueError("bootstrap_samples must be at least 1.")

    generator = random.Random(seed)
    estimates = [fmean(generator.choice(deltas) for _ in range(len(deltas))) for _ in range(samples)]
    tail = (1 - confidence_level) / 2
    return [_quantile(estimates, tail), _quantile(estimates, 1 - tail)]


def wilson_interval(successes: int, total: int, confidence_level: float) -> list[float]:
    """Return a Wilson score interval for a binomial success rate."""
    if total < 1:
        raise ValueError("total must be at least 1.")
    z = NormalDist().inv_cdf(1 - (1 - confidence_level) / 2)
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _policy_summary(outcomes: list[EpisodeOutcome], confidence_level: float) -> dict[str, Any]:
    success_count = sum(outcome.success for outcome in outcomes)
    return {
        "successes": success_count,
        "success_rate": success_count / len(outcomes),
        "success_rate_ci": wilson_interval(success_count, len(outcomes), confidence_level),
        "avg_sum_reward": fmean(outcome.sum_reward for outcome in outcomes),
        "avg_max_reward": fmean(outcome.max_reward for outcome in outcomes),
    }


def summarize_pairs(
    pairs: list[EpisodePair], *, confidence_level: float, bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    """Summarize success and reward changes for matched episodes."""
    baseline = [pair.baseline for pair in pairs]
    candidate = [pair.candidate for pair in pairs]
    success_deltas = [float(pair.candidate.success) - float(pair.baseline.success) for pair in pairs]
    return {
        "matched_episodes": len(pairs),
        "baseline": _policy_summary(baseline, confidence_level),
        "candidate": _policy_summary(candidate, confidence_level),
        "delta": {
            "success_rate": fmean(success_deltas),
            "success_rate_ci": paired_bootstrap_interval(
                success_deltas,
                confidence_level=confidence_level,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "avg_sum_reward": fmean(pair.candidate.sum_reward - pair.baseline.sum_reward for pair in pairs),
            "avg_max_reward": fmean(pair.candidate.max_reward - pair.baseline.max_reward for pair in pairs),
        },
    }


def _decision(
    summary: dict[str, Any],
    unmatched: list[dict[str, Any]],
    protocol_comparison: dict[str, Any],
    *,
    min_episodes: int,
    minimum_improvement_pp: float,
    maximum_regression_pp: float,
) -> dict[str, str]:
    matched = summary["matched_episodes"]
    lower, upper = (value * 100 for value in summary["delta"]["success_rate_ci"])
    if protocol_comparison["status"] == "different":
        return {
            "status": "not_comparable",
            "reason": "The baseline and candidate evaluation protocols differ.",
        }
    if unmatched:
        return {
            "status": "not_comparable",
            "reason": f"{len(unmatched)} episode(s) could not be paired across the two runs.",
        }
    if matched < min_episodes:
        return {
            "status": "inconclusive",
            "reason": f"Only {matched} matched episodes; the gate requires at least {min_episodes}.",
        }
    if lower > minimum_improvement_pp:
        return {
            "status": "improved",
            "reason": (
                f"The lower confidence bound ({lower:.2f} pp) exceeds the required improvement "
                f"({minimum_improvement_pp:.2f} pp)."
            ),
        }
    if upper < -maximum_regression_pp:
        return {
            "status": "regressed",
            "reason": (
                f"The upper confidence bound ({upper:.2f} pp) is below the allowed regression floor "
                f"(-{maximum_regression_pp:.2f} pp)."
            ),
        }
    return {
        "status": "inconclusive",
        "reason": "The confidence interval crosses the configured improvement and regression gates.",
    }


def build_comparison_report(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    min_episodes: int = 20,
    minimum_improvement_pp: float = 0.0,
    maximum_regression_pp: float = 0.0,
) -> dict[str, Any]:
    """Build a machine-readable comparison report from two evaluation artifacts."""
    if min_episodes < 1:
        raise ValueError("min_episodes must be at least 1.")
    if minimum_improvement_pp < 0 or maximum_regression_pp < 0:
        raise ValueError("Improvement and regression thresholds cannot be negative.")

    baseline_file, baseline = load_episodes(baseline_path)
    candidate_file, candidate = load_episodes(candidate_path)
    baseline_display = _display_eval_info(baseline_path)
    candidate_display = _display_eval_info(candidate_path)
    baseline_manifest = load_eval_manifest(baseline_file)
    candidate_manifest = load_eval_manifest(candidate_file)
    if baseline_manifest is None or candidate_manifest is None:
        protocol_comparison = {
            "status": "unavailable",
            "reason": "One or both runs predate eval_manifest.json; protocol equality was not verified.",
        }
    elif baseline_manifest[1]["protocol"] == candidate_manifest[1]["protocol"]:
        protocol_comparison = {"status": "matched", "reason": "Evaluation protocols match exactly."}
    else:
        protocol_comparison = {
            "status": "different",
            "reason": "Evaluation protocol manifests differ.",
            "baseline": baseline_manifest[1]["protocol"],
            "candidate": candidate_manifest[1]["protocol"],
        }
    pairs, unmatched, pairing_modes = pair_episodes(baseline, candidate)
    if not pairs:
        raise ValueError("The evaluation artifacts do not contain any comparable episodes.")

    summary = summarize_pairs(
        pairs,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    by_task: list[dict[str, Any]] = []
    grouped_pairs: dict[tuple[str, str], list[EpisodePair]] = defaultdict(list)
    for pair in pairs:
        grouped_pairs[(pair.baseline.task_group, pair.baseline.task_id)].append(pair)
    for task_key, task_pairs in sorted(grouped_pairs.items()):
        by_task.append(
            {
                "task_group": task_key[0],
                "task_id": task_key[1],
                **summarize_pairs(
                    task_pairs,
                    confidence_level=confidence_level,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
            }
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "decision": _decision(
            summary,
            unmatched,
            protocol_comparison,
            min_episodes=min_episodes,
            minimum_improvement_pp=minimum_improvement_pp,
            maximum_regression_pp=maximum_regression_pp,
        ),
        "configuration": {
            "confidence_level": confidence_level,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "min_episodes": min_episodes,
            "minimum_improvement_pp": minimum_improvement_pp,
            "maximum_regression_pp": maximum_regression_pp,
        },
        "summary": summary,
        "by_task": by_task,
        "pairing": {
            "modes": pairing_modes,
            "unmatched": unmatched,
        },
        "episodes": [
            {
                "task_group": pair.baseline.task_group,
                "task_id": pair.baseline.task_id,
                "pairing": pair.pairing,
                "baseline": asdict(pair.baseline),
                "candidate": asdict(pair.candidate),
            }
            for pair in pairs
        ],
        "provenance": {
            "baseline": {
                "path": str(baseline_display),
                "sha256": sha256_file(baseline_file),
                "manifest": None
                if baseline_manifest is None
                else {
                    "path": str(baseline_display.parent / "eval_manifest.json"),
                    "sha256": sha256_file(baseline_manifest[0]),
                },
            },
            "candidate": {
                "path": str(candidate_display),
                "sha256": sha256_file(candidate_file),
                "manifest": None
                if candidate_manifest is None
                else {
                    "path": str(candidate_display.parent / "eval_manifest.json"),
                    "sha256": sha256_file(candidate_manifest[0]),
                },
            },
            "protocol_comparison": protocol_comparison,
        },
    }


def _format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable Markdown report."""
    summary = report["summary"]
    delta = summary["delta"]
    ci = delta["success_rate_ci"]
    decision = report["decision"]
    lines = [
        "# Simulation evaluation comparison",
        "",
        f"**Decision: {decision['status'].upper()}** — {decision['reason']}",
        "",
        "## Overall result",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Success rate | {_format_rate(summary['baseline']['success_rate'])} | "
            f"{_format_rate(summary['candidate']['success_rate'])} | "
            f"{delta['success_rate'] * 100:+.1f} pp |"
        ),
        (
            f"| Average summed reward | {summary['baseline']['avg_sum_reward']:.3f} | "
            f"{summary['candidate']['avg_sum_reward']:.3f} | {delta['avg_sum_reward']:+.3f} |"
        ),
        (
            f"| Average max reward | {summary['baseline']['avg_max_reward']:.3f} | "
            f"{summary['candidate']['avg_max_reward']:.3f} | {delta['avg_max_reward']:+.3f} |"
        ),
        "",
        (
            f"Matched episodes: **{summary['matched_episodes']}**. "
            f"Paired success-rate {report['configuration']['confidence_level']:.0%} CI: "
            f"**[{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}] percentage points**."
        ),
        "",
        "## Results by task",
        "",
        "| Suite | Task | Episodes | Baseline | Candidate | Delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for task in report["by_task"]:
        lines.append(
            f"| {task['task_group']} | {task['task_id']} | {task['matched_episodes']} | "
            f"{_format_rate(task['baseline']['success_rate'])} | "
            f"{_format_rate(task['candidate']['success_rate'])} | "
            f"{task['delta']['success_rate'] * 100:+.1f} pp |"
        )

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Pairing modes: `{json.dumps(report['pairing']['modes'], sort_keys=True)}`",
            f"- Unmatched episodes: `{len(report['pairing']['unmatched'])}`",
            f"- Protocol comparison: `{report['provenance']['protocol_comparison']['status']}`",
            f"- Bootstrap seed: `{report['configuration']['bootstrap_seed']}`",
            f"- Bootstrap samples: `{report['configuration']['bootstrap_samples']}`",
            f"- Baseline SHA-256: `{report['provenance']['baseline']['sha256']}`",
            f"- Candidate SHA-256: `{report['provenance']['candidate']['sha256']}`",
            "",
            "The JSON report contains the complete per-episode evidence and input paths.",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Write JSON and Markdown forms of a comparison report."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "comparison.json"
    markdown_path = destination / "comparison.md"
    with json_path.open("w") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    markdown_path.write_text(render_markdown(report))
    return json_path, markdown_path
