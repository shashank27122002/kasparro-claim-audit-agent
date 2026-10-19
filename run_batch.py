
from __future__ import annotations

import asyncio
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from agent.db import init_db, save_run
from agent.evaluate import evaluate
from agent.facts import BrandFacts
from agent.llm import Agent


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

ANSWERS_FILE = DATA_DIR / "answers.csv"
FACTS_FILE = DATA_DIR / "brand_facts.csv"
GOLD_FILE = DATA_DIR / "development_gold.csv"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env", override=True)

VERSION = os.getenv("VERSION", "v1").strip().lower()
SPLIT = os.getenv("SPLIT", "development").strip().lower()

VALID_VERSIONS = {"v1", "v2"}
VALID_SPLITS = {"development", "check"}

if VERSION not in VALID_VERSIONS:
    raise ValueError(
        f"Invalid VERSION={VERSION!r}. Choose v1 or v2."
    )

if SPLIT not in VALID_SPLITS:
    raise ValueError(
        f"Invalid SPLIT={SPLIT!r}. "
        "Choose development or check."
    )


# ---------------------------------------------------------
# JSON and checkpoint helpers
# ---------------------------------------------------------

def json_safe(value: Any) -> Any:
    """Convert common Python/Pydantic values into JSON-safe data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if hasattr(value, "to_dict"):
        return value.to_dict()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON atomically to avoid leaving a partial checkpoint."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(temporary_path, path)


def prediction_to_dict(prediction: Any) -> dict[str, Any]:
    """Convert an agent prediction to a dictionary."""
    if hasattr(prediction, "model_dump"):
        result = prediction.model_dump(mode="json")
    elif isinstance(prediction, dict):
        result = prediction
    else:
        raise TypeError(
            "Agent.predict() must return a Pydantic model "
            "or dictionary."
        )

    if not isinstance(result, dict):
        raise TypeError("Prediction serialization did not return a dictionary.")

    return result


# ---------------------------------------------------------
# Batch execution
# ---------------------------------------------------------

async def main() -> None:
    print(
        f"Starting run: version={VERSION}, "
        f"split={SPLIT}"
    )

    # Validate files before making API calls.
    for path in (ANSWERS_FILE, FACTS_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file does not exist: {path}"
            )

    answers_df = pd.read_csv(ANSWERS_FILE)

    # IMPORTANT: the source CSV uses ai_answer, not answer_text.
    required_columns = {
        "answer_id",
        "split",
        "brand_id",
        "question",
        "ai_answer",
    }

    missing_columns = required_columns - set(answers_df.columns)

    if missing_columns:
        raise ValueError(
            "answers.csv is missing columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: {answers_df.columns.tolist()}"
        )

    # Keep the source data unchanged and select only the requested split.
    subset = answers_df[
        answers_df["split"].astype(str).str.strip().str.lower()
        == SPLIT
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No answers found for split={SPLIT!r}."
        )

    if subset["answer_id"].duplicated().any():
        duplicates = subset.loc[
            subset["answer_id"].duplicated(),
            "answer_id",
        ].tolist()

        raise ValueError(
            f"Duplicate answer IDs in selected split: {duplicates}"
        )

    subset["answer_id"] = subset["answer_id"].astype(str)

    expected_total = 40 if SPLIT == "development" else 10

    print(f"Answers in selected split: {len(subset)}")

    if len(subset) != expected_total:
        print(
            f"WARNING: expected {expected_total} answers "
            f"for split={SPLIT}, found {len(subset)}."
        )

    checkpoint_path = (
        ARTIFACTS_DIR
        / f"checkpoint-{VERSION}-{SPLIT}.json"
    )

    predictions: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    run_id: str

    # Restore progress when the checkpoint matches this run.
    if checkpoint_path.exists():
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )

        if (
            checkpoint.get("version") != VERSION
            or checkpoint.get("split") != SPLIT
        ):
            raise ValueError(
                "Checkpoint configuration does not match "
                "the requested version and split."
            )

        run_id = checkpoint["run_id"]
        predictions = checkpoint.get("predictions", [])

        valid_ids = set(subset["answer_id"])

        restored_ids = {
            str(item["answer_id"])
            for item in predictions
            if "answer_id" in item
        }

        unexpected_ids = restored_ids - valid_ids

        if unexpected_ids:
            raise ValueError(
                "Checkpoint contains answer IDs not found "
                f"in this split: {sorted(unexpected_ids)}"
            )

        completed_ids = restored_ids

        print(
            f"Resuming {checkpoint_path.name}: "
            f"{len(completed_ids)}/{len(subset)} "
            "predictions already saved."
        )
    else:
        run_id = str(uuid.uuid4())

    def save_checkpoint() -> None:
        write_json_atomic(
            checkpoint_path,
            {
                "run_id": run_id,
                "version": VERSION,
                "split": SPLIT,
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "total_answers": len(subset),
                "completed_count": len(completed_ids),
                "predictions": predictions,
            },
        )

    # Save even an initially empty checkpoint before API calls.
    save_checkpoint()

    facts = BrandFacts(FACTS_FILE)
    agent = Agent(facts)

    total = len(subset)

    try:
        for index, (_, row) in enumerate(
            subset.iterrows(),
            start=1,
        ):
            answer_id = str(row["answer_id"])

            if answer_id in completed_ids:
                print(
                    f"[{index}/{total}] {answer_id}: "
                    "restored from checkpoint"
                )
                continue

            print(
                f"[{index}/{total}] {answer_id}: "
                "processing...",
                flush=True,
            )

            # Agent.predict uses the actual CSV row, including ai_answer.
            agent_row = row.copy()
            agent_row["answer_text"] = row["ai_answer"]
            prediction = await agent.predict(agent_row, VERSION)
            prediction_dict = prediction_to_dict(prediction)

            prediction_id = str(
                prediction_dict.get("answer_id", answer_id)
            )

            if prediction_id != answer_id:
                raise ValueError(
                    f"Prediction ID mismatch: expected "
                    f"{answer_id}, received {prediction_id}."
                )

            prediction_dict["answer_id"] = answer_id
            prediction_dict["version"] = VERSION

            predictions.append(prediction_dict)
            completed_ids.add(answer_id)

            # Persist every successful prediction immediately.
            save_checkpoint()

            print(
                f"[{index}/{total}] {answer_id}: completed",
                flush=True,
            )

    except Exception:
        # Preserve all successful predictions before propagating the error.
        save_checkpoint()

        print(
            "\nBatch stopped before completion. "
            f"Saved {len(completed_ids)}/{total} predictions."
        )
        print(f"Checkpoint retained: {checkpoint_path}")

        raise

    # Check that every selected answer has a prediction.
    if completed_ids != set(subset["answer_id"]):
        missing_ids = (
            set(subset["answer_id"]) - completed_ids
        )

        save_checkpoint()

        raise RuntimeError(
            "Batch is incomplete. Missing answer IDs: "
            f"{sorted(missing_ids)}"
        )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    metrics = None

    if SPLIT == "development":
        if not GOLD_FILE.exists():
            raise FileNotFoundError(
                f"Development gold file not found: {GOLD_FILE}"
            )

        metrics = evaluate(
            predictions,
            GOLD_FILE,
            ANSWERS_FILE,
        )

        print("\nDevelopment evaluation:")
        print(json.dumps(json_safe(metrics), indent=2))

    else:
        print(
            "\nCheck split completed. "
            "Do not use check-set labels to tune prompts."
        )

    # -----------------------------------------------------
    # Final artifact
    # -----------------------------------------------------

    completed_at = datetime.now(timezone.utc).isoformat()

    artifact = {
        "run_id": run_id,
        "version": VERSION,
        "split": SPLIT,
        "model": getattr(agent, "model", None),
        "started_or_resumed_at": None,
        "completed_at": completed_at,
        "total_answers": total,
        "completed_count": len(predictions),
        "metrics": json_safe(metrics),
        "predictions": predictions,
    }

    artifact_path = (
        ARTIFACTS_DIR
        / f"{VERSION}-{SPLIT}-{run_id}.json"
    )

    write_json_atomic(artifact_path, artifact)

    print(f"\nArtifact saved: {artifact_path}")

    # -----------------------------------------------------
    # Database persistence
    # -----------------------------------------------------

    try:
        init_db()

        save_run(
            run_id,
            VERSION,
            SPLIT,
            getattr(agent, "model", ""),
            json_safe(metrics),
            subset.to_dict("records"),
            predictions,
        )

        print("Run saved to PostgreSQL.")
    except Exception as exc:
        # Do not discard the completed JSON artifact if the DB is unavailable.
        print(
            "WARNING: JSON artifact was saved, but PostgreSQL "
            f"persistence failed: {exc}"
        )

    print(
        f"\nRun complete: version={VERSION}, "
        f"split={SPLIT}, predictions={len(predictions)}/{total}"
    )

    print(
        "Checkpoint retained at: "
        f"{checkpoint_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Kasparro audit batch or all four required batches."
    )
    parser.add_argument(
        "--version",
        choices=("v1", "v2"),
        help="Agent version to run (defaults to VERSION in .env).",
    )
    parser.add_argument(
        "--split",
        choices=("development", "check"),
        help="Dataset split to run (defaults to SPLIT in .env).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run v1 development, v2 development, v1 check, and v2 check.",
    )
    parser.add_argument(
        "--budget-seconds",
        type=int,
        default=600,
        help="Maximum wall-clock time for --all (default: 600 seconds).",
    )
    return parser.parse_args()


def run_all_batches(budget_seconds: int = 600) -> int:
    """Run each required batch with its own resumable checkpoint."""
    batches = [
        ("v1", "development"),
        ("v2", "development"),
        ("v1", "check"),
        ("v2", "check"),
    ]
    deadline = time.monotonic() + max(1, budget_seconds)
    failures: list[str] = []

    print(
        f"Running {len(batches)} batches with a total time budget "
        f"of {max(1, budget_seconds)} seconds.",
        flush=True,
    )

    for version, split in batches:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failures.append(
                f"{version}/{split}: not started; overall time budget expired"
            )
            continue

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--version",
            version,
            "--split",
            split,
        ]
        print(
            f"\n=== Starting {version} / {split} "
            f"(about {int(remaining)} seconds remain) ===",
            flush=True,
        )

        try:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                timeout=max(1, remaining),
                check=False,
            )
            if completed.returncode == 0:
                print(
                    f"=== Completed {version} / {split} ===",
                    flush=True,
                )
            else:
                failures.append(
                    f"{version}/{split}: process exited "
                    f"with code {completed.returncode}"
                )
                print(
                    f"=== FAILED {version} / {split}; checkpoint retained "
                    "if the batch saved progress ===",
                    flush=True,
                )
        except subprocess.TimeoutExpired:
            failures.append(
                f"{version}/{split}: stopped at the overall time limit; "
                "checkpoint should retain completed predictions"
            )
            print(
                "\nOverall time budget reached. Stopping the active batch; "
                "its checkpoint can be resumed later.",
                flush=True,
            )
            break

    if failures:
        print("\nBATCH SUMMARY: one or more batches are incomplete.", flush=True)
        for failure in failures:
            print(f"- {failure}", flush=True)
        print(
            "Do not treat missing batches as completed. Re-run --all to resume "
            "from their checkpoints.",
            flush=True,
        )
        return 1

    print("\nBATCH SUMMARY: all four batches completed.", flush=True)
    return 0


if __name__ == "__main__":
    args = parse_args()

    if args.all:
        raise SystemExit(run_all_batches(args.budget_seconds))

    # CLI values take precedence over values loaded from .env.
    if args.version:
        VERSION = args.version
    if args.split:
        SPLIT = args.split

    asyncio.run(main())
