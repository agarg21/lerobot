#!/usr/bin/env bash

# Run a bounded pi0/LIBERO smoke evaluation inside a Hugging Face Job.
# The caller is responsible for enforcing the hardware flavor and timeout.

set -euo pipefail

ARTIFACT_REPO=${ARTIFACT_REPO:?Set ARTIFACT_REPO to a private Hugging Face dataset repository.}
MODEL_REVISION=${MODEL_REVISION:-1dc27a57cf4b54c6fb138ed80a97da150e812e76}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
SOURCE_IMAGE=${SOURCE_IMAGE:-unknown}
N_EPISODES=${N_EPISODES:-2}

if [[ ! "$N_EPISODES" =~ ^[1-9][0-9]*$ ]] || (( N_EPISODES > 20 )); then
    echo "N_EPISODES must be an integer from 1 through 20." >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
LEROBOT_REVISION=$(git -C "$REPO_ROOT" rev-parse HEAD)
RUN_ROOT="/tmp/pi0-libero-smoke/$RUN_ID"
OUTPUT_DIR="$RUN_ROOT/eval"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/job.log") 2>&1

export ARTIFACT_REPO MODEL_REVISION RUN_ID SOURCE_IMAGE LEROBOT_REVISION RUN_ROOT N_EPISODES

write_status() {
    local exit_code=$1
    export JOB_EXIT_CODE=$exit_code
    python - <<'PY'
import json
import os
from datetime import UTC, datetime
from pathlib import Path

payload = {
    "run_id": os.environ["RUN_ID"],
    "status": "completed" if os.environ["JOB_EXIT_CODE"] == "0" else "failed",
    "exit_code": int(os.environ["JOB_EXIT_CODE"]),
    "finished_at": datetime.now(UTC).isoformat(),
    "policy": "lerobot/pi0_libero_base",
    "policy_revision": os.environ["MODEL_REVISION"],
    "lerobot_revision": os.environ["LEROBOT_REVISION"],
    "source_image": os.environ["SOURCE_IMAGE"],
    "environment": "libero_object",
    "task_ids": [0],
    "seed": 1000,
    "episodes": int(os.environ["N_EPISODES"]),
}
Path(os.environ["RUN_ROOT"], "job_status.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY
}

retain_artifacts() {
    local exit_code=$?
    set +e
    write_status "$exit_code"
    hf repo create "$ARTIFACT_REPO" --type dataset --private --exist-ok
    hf upload "$ARTIFACT_REPO" "$RUN_ROOT" "runs/$RUN_ID" \
        --repo-type dataset \
        --commit-message "Retain pi0 LIBERO smoke $RUN_ID"
    local upload_exit=$?
    if [[ $upload_exit -ne 0 ]]; then
        echo "Artifact upload failed with exit code $upload_exit."
    fi
    exit "$exit_code"
}

trap retain_artifacts EXIT

echo "Run ID: $RUN_ID"
echo "LeRobot revision: $LEROBOT_REVISION"
echo "Policy: lerobot/pi0_libero_base@$MODEL_REVISION"
echo "Protocol: libero_object task 0, seed 1000, $N_EPISODES episodes"

uv pip install -e "$REPO_ROOT[evaluation,pi,libero]"

LIBERO_USER_DIR=$(python - <<'PY'
from pathlib import Path
print(Path.home() / ".libero")
PY
)
LIBERO_PACKAGE_DIR=$(python - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("libero")
if spec is None or spec.origin is None:
    raise RuntimeError("The hf-libero package is not installed")
print(Path(spec.origin).parent / "libero")
PY
)

export LIBERO_USER_DIR LIBERO_PACKAGE_DIR
mkdir -p "$LIBERO_USER_DIR"
python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lerobot/libero-assets",
    repo_type="dataset",
    local_dir=os.path.join(os.environ["LIBERO_USER_DIR"], "assets"),
)
PY

python - <<'PY'
import os
from pathlib import Path

user_dir = Path(os.environ["LIBERO_USER_DIR"])
package_dir = Path(os.environ["LIBERO_PACKAGE_DIR"])
config = "\n".join(
    [
        f"assets: {user_dir / 'assets'}",
        f"bddl_files: {package_dir / 'bddl_files'}",
        f"datasets: {package_dir.parent / 'datasets'}",
        f"init_states: {package_dir / 'init_files'}",
        "",
    ]
)
(user_dir / "config.yaml").write_text(config, encoding="utf-8")
PY

export MUJOCO_GL=egl

lerobot-eval \
    --policy.path=lerobot/pi0_libero_base \
    --policy.pretrained_revision="$MODEL_REVISION" \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=libero_object \
    --env.task_ids='[0]' \
    --env.init_states=true \
    --env.hard_reset=true \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=1 \
    --eval.n_episodes="$N_EPISODES" \
    --eval.use_async_envs=false \
    --seed=1000 \
    --job_name=pi0_libero_object_task0_smoke \
    --output_dir="$OUTPUT_DIR"
