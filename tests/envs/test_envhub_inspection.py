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

import hashlib
from types import SimpleNamespace

import pytest

from lerobot.envs.envhub_inspection import EnvHubInspectionError, inspect_envhub_target

PINNED_SHA = "bdf2d67ccfb5d73e9bff12927249bc2cef408c21"


class FakeApi:
    def __init__(self, info):
        self.info = info
        self.calls = []

    def model_info(self, repo_id, **kwargs):
        self.calls.append((repo_id, kwargs))
        return self.info


def make_info(*, sha=PINNED_SHA, files=((".gitattributes", 0), ("env.py", 33)), card_data=None, tags=None):
    return SimpleNamespace(
        sha=sha,
        siblings=[SimpleNamespace(rfilename=filename, size=size) for filename, size in files],
        card_data=card_data,
        tags=tags or [],
    )


def test_inspect_envhub_target_records_pinned_provenance(tmp_path):
    entry = tmp_path / "env.py"
    content = b"def make_env():\n    return None\n"
    entry.write_bytes(content)
    api = FakeApi(
        make_info(
            files=((".gitattributes", 0), ("env.py", len(content))),
            card_data={"license": "apache-2.0"},
            tags=["robotics", "robotics"],
        )
    )
    download_calls = []

    def download_file(**kwargs):
        download_calls.append(kwargs)
        return str(entry)

    result = inspect_envhub_target(
        f"lerobot/cartpole-env@{PINNED_SHA}",
        api=api,
        download_file=download_file,
    )

    assert result.repository_id == "lerobot/cartpole-env"
    assert result.requested_revision == PINNED_SHA
    assert result.resolved_revision == PINNED_SHA
    assert result.canonical_target == f"lerobot/cartpole-env@{PINNED_SHA}:env.py"
    assert result.entry_file == "env.py"
    assert result.entry_sha256 == hashlib.sha256(content).hexdigest()
    assert result.declared_entry_size_bytes == len(content)
    assert result.entry_size_bytes == len(content)
    assert result.is_commit_pinned is True
    assert result.declared_license == "apache-2.0"
    assert result.tags == ("robotics",)
    assert api.calls == [("lerobot/cartpole-env", {"revision": PINNED_SHA, "files_metadata": True})]
    assert download_calls == [
        {
            "repo_id": "lerobot/cartpole-env",
            "filename": "env.py",
            "revision": PINNED_SHA,
            "cache_dir": None,
        }
    ]
    assert result.to_dict()["tags"] == ["robotics"]


def test_inspect_envhub_target_resolves_branch_but_marks_it_unpinned(tmp_path):
    entry = tmp_path / "custom.py"
    entry.write_text("pass\n")
    api = FakeApi(make_info(files=(("custom.py", 5),), card_data=SimpleNamespace(license="mit")))

    result = inspect_envhub_target(
        "user/repo@main:custom.py",
        api=api,
        download_file=lambda **_: str(entry),
    )

    assert result.requested_revision == "main"
    assert result.resolved_revision == PINNED_SHA
    assert result.canonical_target == f"user/repo@{PINNED_SHA}:custom.py"
    assert result.entry_file == "custom.py"
    assert result.is_commit_pinned is False
    assert result.declared_license == "mit"


def test_inspect_envhub_target_rejects_missing_entry_file(tmp_path):
    api = FakeApi(make_info(files=(("README.md", 10),)))

    with pytest.raises(FileNotFoundError, match="env.py"):
        inspect_envhub_target(
            f"user/repo@{PINNED_SHA}",
            api=api,
            download_file=lambda **_: str(tmp_path / "unused"),
        )


@pytest.mark.parametrize("invalid_sha", [None, "main", "abc123", "g" * 40])
def test_inspect_envhub_target_requires_full_resolved_commit(invalid_sha):
    api = FakeApi(make_info(sha=invalid_sha))

    with pytest.raises(EnvHubInspectionError, match="40-character commit SHA"):
        inspect_envhub_target("user/repo", api=api)


def test_inspect_envhub_target_rejects_requested_commit_mismatch(tmp_path):
    requested_sha = "a" * 40
    api = FakeApi(make_info())

    with pytest.raises(EnvHubInspectionError, match="unexpected commit"):
        inspect_envhub_target(
            f"user/repo@{requested_sha}",
            api=api,
            download_file=lambda **_: str(tmp_path / "unused"),
        )


def test_inspect_envhub_target_rejects_oversized_entry_before_download():
    api = FakeApi(make_info(files=(("env.py", 101),)))
    download_calls = []

    with pytest.raises(EnvHubInspectionError, match="exceeding the 100-byte inspection limit"):
        inspect_envhub_target(
            f"user/repo@{PINNED_SHA}",
            max_entry_size_bytes=100,
            api=api,
            download_file=lambda **kwargs: download_calls.append(kwargs),
        )

    assert download_calls == []


def test_inspect_envhub_target_rejects_missing_size_metadata():
    info = make_info()
    info.siblings[-1].size = None

    with pytest.raises(EnvHubInspectionError, match="valid size"):
        inspect_envhub_target(f"user/repo@{PINNED_SHA}", api=FakeApi(info))


def test_inspect_envhub_target_rejects_download_size_mismatch(tmp_path):
    entry = tmp_path / "env.py"
    entry.write_text("pass\n")
    api = FakeApi(make_info(files=(("env.py", 4),)))

    with pytest.raises(EnvHubInspectionError, match="metadata declared 4 bytes"):
        inspect_envhub_target(
            f"user/repo@{PINNED_SHA}",
            api=api,
            download_file=lambda **_: str(entry),
        )


def test_inspect_envhub_target_requires_positive_size_limit():
    with pytest.raises(ValueError, match="must be positive"):
        inspect_envhub_target("user/repo", max_entry_size_bytes=0)
