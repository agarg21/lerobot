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

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from .utils import _parse_hub_url

_FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
DEFAULT_MAX_ENTRY_SIZE_BYTES = 1024 * 1024


class EnvHubInspectionError(RuntimeError):
    """Raised when an EnvHub target cannot produce trustworthy static provenance."""


@dataclass(frozen=True)
class EnvHubProvenance:
    """Static, non-executing provenance for one EnvHub entry point."""

    repository_id: str
    requested_revision: str | None
    resolved_revision: str
    canonical_target: str
    entry_file: str
    entry_sha256: str
    declared_entry_size_bytes: int
    entry_size_bytes: int
    is_commit_pinned: bool
    declared_license: str | None
    tags: tuple[str, ...]
    schema_version: str = "0.1"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with stable field names."""
        result = asdict(self)
        result["tags"] = list(self.tags)
        return result


def _declared_card_license(card_data: Any) -> str | None:
    if card_data is None:
        return None
    value = card_data.get("license") if isinstance(card_data, dict) else getattr(card_data, "license", None)
    return str(value) if value else None


def _entry_metadata(info: Any, entry_file: str) -> Any | None:
    return next(
        (
            sibling
            for sibling in (getattr(info, "siblings", None) or [])
            if getattr(sibling, "rfilename", None) == entry_file
        ),
        None,
    )


def _sha256_file(path: str | Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def inspect_envhub_target(
    target: str,
    *,
    cache_dir: str | None = None,
    max_entry_size_bytes: int = DEFAULT_MAX_ENTRY_SIZE_BYTES,
    api: Any | None = None,
    download_file: Callable[..., str] = hf_hub_download,
) -> EnvHubProvenance:
    """Inspect an EnvHub target without importing or executing its code.

    The target uses LeRobot's ``repo[@revision][:path]`` syntax. Metadata is
    resolved first, then the entry file is downloaded by the resolved commit
    SHA to prevent a moving branch from changing between inspection steps.

    This function does not establish that remote code is safe. It records the
    exact artifact that a user can review before a later, explicit trust step.
    Later execution must use ``canonical_target`` rather than the original
    target so a moving branch cannot substitute different code.
    """
    if max_entry_size_bytes <= 0:
        raise ValueError("max_entry_size_bytes must be positive.")

    repository_id, requested_revision, entry_file = _parse_hub_url(target)
    hub_api = api if api is not None else HfApi()
    info = hub_api.model_info(repository_id, revision=requested_revision, files_metadata=True)

    resolved_revision = getattr(info, "sha", None)
    if not isinstance(resolved_revision, str) or not _FULL_COMMIT_SHA.fullmatch(resolved_revision):
        raise EnvHubInspectionError(
            f"Hub metadata for '{target}' did not provide a full 40-character commit SHA."
        )

    requested_is_commit = bool(requested_revision and _FULL_COMMIT_SHA.fullmatch(requested_revision))
    if requested_is_commit and requested_revision.lower() != resolved_revision.lower():
        raise EnvHubInspectionError(
            f"Requested commit {requested_revision} resolved to unexpected commit {resolved_revision}."
        )

    entry_metadata = _entry_metadata(info, entry_file)
    if entry_metadata is None:
        raise FileNotFoundError(
            f"Could not find EnvHub entry file '{entry_file}' in {repository_id}@{resolved_revision}."
        )

    declared_entry_size_bytes = getattr(entry_metadata, "size", None)
    if not isinstance(declared_entry_size_bytes, int) or declared_entry_size_bytes < 0:
        raise EnvHubInspectionError(f"Hub metadata did not provide a valid size for '{entry_file}'.")
    if declared_entry_size_bytes > max_entry_size_bytes:
        raise EnvHubInspectionError(
            f"EnvHub entry file '{entry_file}' is {declared_entry_size_bytes} bytes, exceeding the "
            f"{max_entry_size_bytes}-byte inspection limit."
        )

    local_file = download_file(
        repo_id=repository_id,
        filename=entry_file,
        revision=resolved_revision,
        cache_dir=cache_dir,
    )
    entry_sha256, entry_size_bytes = _sha256_file(local_file)
    if entry_size_bytes != declared_entry_size_bytes:
        raise EnvHubInspectionError(
            f"Downloaded '{entry_file}' is {entry_size_bytes} bytes, but Hub metadata declared "
            f"{declared_entry_size_bytes} bytes."
        )

    tags = tuple(sorted({str(tag) for tag in (getattr(info, "tags", None) or [])}))
    canonical_target = f"{repository_id}@{resolved_revision}:{entry_file}"

    return EnvHubProvenance(
        repository_id=repository_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        canonical_target=canonical_target,
        entry_file=entry_file,
        entry_sha256=entry_sha256,
        declared_entry_size_bytes=declared_entry_size_bytes,
        entry_size_bytes=entry_size_bytes,
        is_commit_pinned=requested_is_commit,
        declared_license=_declared_card_license(getattr(info, "card_data", None)),
        tags=tags,
    )
