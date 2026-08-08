#!/usr/bin/env bash

# Run a bounded, task-specialized pi0 post-training pilot inside a Hugging Face Job.
# The caller is responsible for enforcing the hardware flavor and timeout.

set -euo pipefail

ARTIFACT_REPO=${ARTIFACT_REPO:?Set ARTIFACT_REPO to a private Hugging Face dataset repository.}
MODEL_REPO=${MODEL_REPO:?Set MODEL_REPO to the private Hugging Face model repository for the selected checkpoint.}
BASE_MODEL_ID=${BASE_MODEL_ID:-lerobot/pi0_libero_finetuned_v044}
BASE_MODEL_REVISION=${BASE_MODEL_REVISION:-45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f}
DATASET_ID=${DATASET_ID:-HuggingFaceVLA/libero}
DATASET_REVISION=${DATASET_REVISION:-86958911c0f959db2bbbdb107eb3e17c5f9c798e}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
SOURCE_IMAGE=${SOURCE_IMAGE:-unknown}

for repo in "$ARTIFACT_REPO" "$MODEL_REPO" "$BASE_MODEL_ID" "$DATASET_ID"; do
    if [[ ! "$repo" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
        echo "Repository IDs must use the form owner/repository: $repo" >&2
        exit 2
    fi
done
for revision in "$BASE_MODEL_REVISION" "$DATASET_REVISION"; do
    if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
        echo "Model and dataset revisions must be full lowercase commit hashes." >&2
        exit 2
    fi
done

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
LEROBOT_REVISION=$(git -C "$REPO_ROOT" rev-parse HEAD)
RUN_ROOT="/tmp/pi0-post-train/$RUN_ID"
OUTPUT_DIR="$RUN_ROOT/output"
SELECTION_FILE="$RUN_ROOT/selection.json"

# These 44 pinned episodes all have the target instruction. With eval_split=0.2,
# LeRobot deterministically uses the first 35 for training and the final 9 for evaluation.
TARGET_EPISODES='[813,818,825,850,852,857,860,876,877,887,905,908,934,945,978,983,984,985,1002,1011,1020,1024,1035,1074,1075,1100,1102,1108,1131,1143,1148,1151,1152,1166,1171,1187,1205,1219,1224,1225,1233,1242,1243,1245]'

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/job.log") 2>&1

export ARTIFACT_REPO MODEL_REPO BASE_MODEL_ID BASE_MODEL_REVISION DATASET_ID DATASET_REVISION
export RUN_ID SOURCE_IMAGE LEROBOT_REVISION RUN_ROOT SELECTION_FILE TARGET_EPISODES

write_manifest() {
    python - <<'PY'
import json
import os
from datetime import UTC, datetime
from pathlib import Path

episodes = json.loads(os.environ["TARGET_EPISODES"])
payload = {
    "run_id": os.environ["RUN_ID"],
    "created_at": datetime.now(UTC).isoformat(),
    "base_model": os.environ["BASE_MODEL_ID"],
    "base_model_revision": os.environ["BASE_MODEL_REVISION"],
    "dataset": os.environ["DATASET_ID"],
    "dataset_revision": os.environ["DATASET_REVISION"],
    "target_task": "pick up the alphabet soup and place it in the basket",
    "episodes": episodes,
    "train_episodes": episodes[:-9],
    "validation_episodes": episodes[-9:],
    "steps": 500,
    "checkpoint_steps": [100, 200, 300, 400, 500],
    "selection_rule": "minimum held-out demonstration loss",
    "train_expert_only": True,
    "learning_rate": 5e-6,
    "precision": "bfloat16",
    "compile_model": False,
    "lerobot_revision": os.environ["LEROBOT_REVISION"],
    "source_image": os.environ["SOURCE_IMAGE"],
    "hardware": "a100-large",
    "timeout_hours": 2,
    "maximum_compute_charge_usd": 5.0,
}
Path(os.environ["RUN_ROOT"], "manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY
}

write_status() {
    local exit_code=$1
    export JOB_EXIT_CODE=$exit_code
    python - <<'PY'
import json
import os
from datetime import UTC, datetime
from pathlib import Path

selection_path = Path(os.environ["SELECTION_FILE"])
selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
payload = {
    "run_id": os.environ["RUN_ID"],
    "status": "completed" if os.environ["JOB_EXIT_CODE"] == "0" else "failed",
    "exit_code": int(os.environ["JOB_EXIT_CODE"]),
    "finished_at": datetime.now(UTC).isoformat(),
    "selected_model_repo": os.environ["MODEL_REPO"] if selection else None,
    "selection": selection,
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
        --exclude 'output/**' \
        --commit-message "Retain pi0 post-training pilot $RUN_ID"
    local upload_exit=$?
    if [[ $upload_exit -ne 0 ]]; then
        echo "Artifact upload failed with exit code $upload_exit."
    fi
    exit "$exit_code"
}

trap retain_artifacts EXIT
write_manifest

echo "Run ID: $RUN_ID"
echo "LeRobot revision: $LEROBOT_REVISION"
echo "Base policy: $BASE_MODEL_ID@$BASE_MODEL_REVISION"
echo "Dataset: $DATASET_ID@$DATASET_REVISION"
echo "Protocol: 35 train episodes, 9 held-out episodes, 500 expert-only steps"

uv pip install -e "$REPO_ROOT[training,pi]"

lerobot-train \
    --policy.path="$BASE_MODEL_ID" \
    --policy.pretrained_revision="$BASE_MODEL_REVISION" \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.train_expert_only=true \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=false \
    --policy.push_to_hub=false \
    --policy.repo_id="$MODEL_REPO" \
    --policy.private=true \
    --policy.optimizer_lr=0.000005 \
    --policy.scheduler_warmup_steps=50 \
    --policy.scheduler_decay_steps=500 \
    --policy.scheduler_decay_lr=0.0000005 \
    --dataset.repo_id="$DATASET_ID" \
    --dataset.revision="$DATASET_REVISION" \
    --dataset.episodes="$TARGET_EPISODES" \
    --dataset.eval_split=0.2 \
    --dataset.video_backend=torchcodec \
    --steps=500 \
    --batch_size=32 \
    --num_workers=4 \
    --log_freq=10 \
    --eval_steps=100 \
    --max_eval_samples=0 \
    --env_eval_freq=0 \
    --save_checkpoint=true \
    --save_freq=100 \
    --save_checkpoint_to_hub=false \
    --wandb.enable=false \
    --seed=1000 \
    --cudnn_deterministic=true \
    --job_name=pi0_alphabet_soup_expert500 \
    --output_dir="$OUTPUT_DIR"

python - <<'PY'
import json
import os
import re
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
matches = re.findall(
    r"step\s+(100|200|300|400|500):\s+eval_loss=([0-9.eE+-]+)",
    (run_root / "job.log").read_text(encoding="utf-8"),
)
losses = {int(step): float(loss) for step, loss in matches}
expected = {100, 200, 300, 400, 500}
if set(losses) != expected:
    raise RuntimeError(f"Expected held-out losses at {sorted(expected)}, observed {sorted(losses)}")
selected_step = min(losses, key=lambda step: (losses[step], step))
selected_dir = run_root / "output" / "checkpoints" / f"{selected_step:06d}" / "pretrained_model"
if not (selected_dir / "config.json").is_file():
    raise FileNotFoundError(f"Selected checkpoint is incomplete: {selected_dir}")
payload = {
    "selection_rule": "minimum held-out demonstration loss",
    "eval_losses": {str(step): losses[step] for step in sorted(losses)},
    "selected_step": selected_step,
    "selected_eval_loss": losses[selected_step],
    "selected_checkpoint": str(selected_dir),
}
(run_root / "selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"Selected step {selected_step} with held-out loss {losses[selected_step]:.4f}")
PY

SELECTED_STEP=$(python -c 'import json, os; print(json.load(open(os.environ["SELECTION_FILE"]))["selected_step"])')
SELECTED_DIR="$OUTPUT_DIR/checkpoints/$(printf '%06d' "$SELECTED_STEP")/pretrained_model"
cp "$SELECTION_FILE" "$SELECTED_DIR/post_training_selection.json"

hf repo create "$MODEL_REPO" --type model --private --exist-ok
hf upload "$MODEL_REPO" "$SELECTED_DIR" . \
    --repo-type model \
    --commit-message "Publish selected pi0 alphabet-soup post-training checkpoint"

echo "Selected checkpoint published privately to https://huggingface.co/$MODEL_REPO"
