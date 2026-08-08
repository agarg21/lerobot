#!/usr/bin/env python

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

"""Compare baseline and candidate ``lerobot-eval`` artifacts."""

import argparse

from lerobot.eval_report import build_comparison_report, write_comparison_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a paired evidence report from two LeRobot simulation evaluations."
    )
    parser.add_argument("--baseline", required=True, help="Baseline eval directory or eval_info.json")
    parser.add_argument("--candidate", required=True, help="Candidate eval directory or eval_info.json")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison.json and comparison.md")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--min-episodes", type=int, default=20)
    parser.add_argument("--minimum-improvement-pp", type=float, default=0.0)
    parser.add_argument("--maximum-regression-pp", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_comparison_report(
        args.baseline,
        args.candidate,
        confidence_level=args.confidence_level,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        min_episodes=args.min_episodes,
        minimum_improvement_pp=args.minimum_improvement_pp,
        maximum_regression_pp=args.maximum_regression_pp,
    )
    json_path, markdown_path = write_comparison_report(report, args.output_dir)
    print(f"Decision: {report['decision']['status']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
