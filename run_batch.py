
import asyncio
import json
import os
import uuid
from pathlib import Path

import pandas as pd

from agent.facts import BrandFacts
from agent.llm import Agent
from agent.evaluate import evaluate
from agent.db import init_db, save_run

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"


def write_json_atomic(path: Path, payload: dict) -> None:
    """Safely persist JSON without leaving a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def get_answer_id(row, index: int) -> str:
    value = getattr(row, "answer_id", None)
    return str(value) if value is not None else f"row-{index}"


async def main() -> None:
    split = os.getenv("SPLIT", "development")
    version = os.getenv("VERSION", "v1")

    if version not in {"v1", "v2"}:
        raise ValueError("VERSION must be 'v1' or 'v2'.")

    answers_path = ROOT / "data" / "answers.csv"
    gold_path = ROOT / "data" / "development_gold.csv"

    if not answers_path.exists():
        raise FileNotFoundError(f"Dataset not found: {answers_path}")

    answers = pd.read_csv(answers_path).fillna("")

    required_columns = {"answer_id", "answer_text", "split"}
    missing_columns = required_columns - set(answers.columns)

    if missing_columns:
        raise ValueError(
            f"answers.csv is missing columns: {sorted(missing_columns)}"
        )

    subset = answers.loc[answers["split"] == split].copy()

    if subset.empty:
        print(f"No answers found for split={split!r}.")
        print("Available splits:", answers["split"].unique().tolist())
        return

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Separate checkpoint files prevent v1 and v2 from sharing predictions.
    checkpoint_path = (
        ARTIFACTS_DIR / f"checkpoint-{version}-{split}.json"
    )

    facts = BrandFacts(ROOT / "data" / "brand_facts.csv")
    agent = Agent(facts)

    if checkpoint_path.exists():
        checkpoint = json.loads(
            checkpoint_path.read_text(encoding="utf-8")
        )

        if (
            checkpoint.get("version") != version
            or checkpoint.get("split") != split
        ):
            raise ValueError(
                "Checkpoint configuration does not match this run."
            )

        run_id = checkpoint["run_id"]
        predictions = checkpoint.get("predictions", [])

        expected_ids = {
            str(value) for value in subset["answer_id"].tolist()
        }
        saved_ids = {
            str(item["answer_id"]) for item in predictions
        }

        if not saved_ids.issubset(expected_ids):
            raise ValueError(
                "Checkpoint contains answer IDs not found in this split."
            )

        print(
            f"Resuming {checkpoint_path.name}: "
            f"{len(saved_ids)}/{len(subset)} predictions saved.",
            flush=True,
        )
    else:
        run_id = f"{version}-{split}-{uuid.uuid4().hex[:8]}"
        predictions = []

    completed_ids = {
        str(item["answer_id"]) for item in predictions
    }

    def save_checkpoint() -> None:
        write_json_atomic(
            checkpoint_path,
            {
                "run_id": run_id,
                "version": version,
                "split": split,
                "model": agent.model,
                "predictions": predictions,
            },
        )

    total = len(subset)

    print(
        f"Starting run: version={version}, split={split}, "
        f"total={total}, model={agent.model}",
        flush=True,
    )

    for index, row in enumerate(
        subset.itertuples(index=False), start=1
    ):
        answer_id = get_answer_id(row, index)

        if answer_id in completed_ids:
            print(
                f"[{index}/{total}] {answer_id}: restored from checkpoint",
                flush=True,
            )
            continue

        try:
            prediction = await agent.predict(row, version)

            prediction_data = (
                prediction.model_dump()
                if hasattr(prediction, "model_dump")
                else prediction
            )

            prediction_data["answer_id"] = str(
                prediction_data["answer_id"]
            )

            predictions.append(prediction_data)
            completed_ids.add(answer_id)

            # Save after every successful answer.
            save_checkpoint()

            print(
                f"[{index}/{total}] {answer_id}: saved "
                f"({len(predictions)}/{total})",
                flush=True,
            )

        except Exception as exc:
            # Preserve successful predictions before stopping.
            save_checkpoint()

            print(
                f"[{index}/{total}] {answer_id}: FAILED - "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            print(
                f"Checkpoint retained: {checkpoint_path}",
                flush=True,
            )
            raise

    if len(predictions) != total:
        save_checkpoint()
        raise RuntimeError(
            f"Incomplete run: expected {total} predictions, "
            f"found {len(predictions)}."
        )

    metrics = {}

    if split == "development":
        metrics = evaluate(
            predictions,
            gold_path,
            answers_path,
        )

    artifact = {
        "run_id": run_id,
        "version": version,
        "split": split,
        "model": agent.model,
        "metrics": metrics,
        "predictions": predictions,
    }

    # Keep final artifacts distinct from checkpoints.
    output_path = ARTIFACTS_DIR / f"{run_id}.json"
    write_json_atomic(output_path, artifact)

    print(f"\nFinal artifact saved: {output_path}", flush=True)
    print(
        "Metrics:",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        flush=True,
    )

    # The JSON artifact remains available if database persistence fails.
    try:
        init_db()
        save_run(
            run_id,
            version,
            split,
            agent.model,
            metrics,
            subset.to_dict("records"),
            predictions,
        )
        print("Database persistence completed.", flush=True)
    except Exception as exc:
        print(
            "WARNING: JSON artifact saved, but database persistence "
            f"failed: {type(exc).__name__}: {exc}",
            flush=True,
        )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "artifact": str(output_path),
                "checkpoint": str(checkpoint_path),
                "predictions": len(predictions),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
