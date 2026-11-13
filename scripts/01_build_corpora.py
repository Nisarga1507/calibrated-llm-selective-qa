"""Download and deterministically split circularity-free benchmark & UK sector corpora."""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def deterministic_split(
    records: list[dict],
    seed: int = 42,
    train_ratio: float = 0.60,
    dev_ratio: float = 0.20,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    return {
        "train": shuffled[:n_train],
        "dev_calibration": shuffled[n_train : n_train + n_dev],
        "test": shuffled[n_train + n_dev :],
    }


def load_truthfulqa_records() -> list[dict]:
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    records = []
    for idx, row in enumerate(ds):
        records.append(
            {
                "id": f"tqa_{idx}",
                "domain": "general",
                "dataset": "truthful_qa",
                "question": row["question"],
                "gold_answer": row["best_answer"],
                "correct_answers": row["correct_answers"],
                "incorrect_answers": row["incorrect_answers"],
                "evidence": "",
            }
        )
    return records


def load_fever_records(max_samples: int = 2000, seed: int = 42) -> list[dict]:
    """Balanced binary fact verification (SUPPORTS vs REFUTES) from VitaminC / MultiNLI."""
    try:
        ds = load_dataset("tals/vitaminc", split="validation")
        filtered = [r for r in ds if r["label"] in {"SUPPORTS", "REFUTES"}]
        rng = random.Random(seed)
        rng.shuffle(filtered)
        records = []
        for idx, row in enumerate(filtered[:max_samples]):
            records.append(
                {
                    "id": f"fever_{idx}",
                    "domain": "general",
                    "dataset": "fever_vitaminc",
                    "question": f"Verify this claim: {row['claim']}",
                    "gold_answer": row["label"],
                    "evidence": row.get("evidence", ""),
                }
            )
        if records:
            return records
    except Exception:
        pass

    ds = load_dataset("nyu-mll/multi_nli", split="validation_matched")
    filtered = [r for r in ds if r["label"] in {0, 2}]
    rng = random.Random(seed)
    rng.shuffle(filtered)
    records = []
    for idx, row in enumerate(filtered[:max_samples]):
        records.append(
            {
                "id": f"nli_{idx}",
                "domain": "general",
                "dataset": "fever_nli",
                "question": f"Verify hypothesis: {row['hypothesis']}",
                "gold_answer": "SUPPORTS" if row["label"] == 0 else "REFUTES",
                "evidence": row["premise"],
            }
        )
    return records


def load_uk_healthcare_records() -> list[dict]:
    """Clinician-annotated PubMedQA pqa_labeled (1,000 expert QA pairs)."""
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    records = []
    for idx, row in enumerate(ds):
        contexts = row.get("context", {}).get("contexts", [])
        evidence_text = " ".join(contexts)
        records.append(
            {
                "id": f"nhs_med_{idx}",
                "domain": "uk_healthcare",
                "dataset": "pubmedqa_pqa_labeled",
                "question": row["question"],
                "gold_answer": row["final_decision"],
                "long_answer": row.get("long_answer", ""),
                "evidence": evidence_text,
            }
        )
    return records


def load_uk_law_records(max_samples: int = 1200, seed: int = 42) -> list[dict]:
    """LegalBench contract & statutory jurisdiction verification (pure Parquet/TSV)."""
    records = []
    for task in ("cuad_governing_law", "abercrombie"):
        try:
            ds = load_dataset("nguha/legalbench", task, split="test")
            for idx, row in enumerate(ds):
                records.append(
                    {
                        "id": f"uklaw_{task}_{idx}",
                        "domain": "uk_law",
                        "dataset": f"legalbench_{task}",
                        "question": f"Determine the governing legal rule or category: {row['text']}",
                        "gold_answer": str(row["answer"]),
                        "evidence": str(row["text"]),
                    }
                )
        except Exception:
            continue

    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:max_samples]


def load_uk_finance_records(max_samples: int = 1200, seed: int = 42) -> list[dict]:
    """Human-annotated financial QA pairs (pure Parquet, zero legacy scripts)."""
    for repo_id, split_name, q_col, a_col, ctx_col in [
        ("ibm-research/finqa", "test", "question", "answer", "pre_text"),
        ("PatronusAI/financebench", "train", "question", "answer", "evidence_text"),
        ("zeroshot/twitter-financial-news-sentiment", "validation", "text", "label", "text"),
    ]:
        try:
            ds = load_dataset(repo_id, split=split_name)
            records = []
            for idx, row in enumerate(ds):
                if len(records) >= max_samples:
                    break
                raw_ctx = row.get(ctx_col, "")
                evidence = " ".join(raw_ctx) if isinstance(raw_ctx, list) else str(raw_ctx)
                q_val = str(row.get(q_col, ""))
                if repo_id == "zeroshot/twitter-financial-news-sentiment":
                    q_val = f"Classify the financial market sentiment of this report: {q_val}"
                records.append(
                    {
                        "id": f"fin_{idx}",
                        "domain": "uk_finance",
                        "dataset": repo_id.split("/")[-1],
                        "question": q_val,
                        "gold_answer": str(row.get(a_col, "")),
                        "evidence": evidence[:1200],
                    }
                )
            if records:
                rng = random.Random(seed)
                rng.shuffle(records)
                return records
        except Exception:
            continue

    raise RuntimeError("Unable to load any Parquet financial QA dataset.")


def main() -> None:
    cfg = yaml.safe_load(Path("configs/v2_conformal_config.yaml").read_text())
    seed = cfg["project"]["seed"]
    splits_cfg = cfg["datasets"]["splits"]
    out_root = Path("data/splits")

    loaders = {
        "truthful_qa": load_truthfulqa_records,
        "fever": lambda: load_fever_records(max_samples=2000, seed=seed),
        "uk_healthcare": load_uk_healthcare_records,
        "uk_law": lambda: load_uk_law_records(max_samples=1200, seed=seed),
        "uk_finance": lambda: load_uk_finance_records(max_samples=1200, seed=seed),
    }

    for name, loader_fn in loaders.items():
        print(f"[3S-CCR] Loading {name}...")
        records = loader_fn()
        splits = deterministic_split(
            records,
            seed=seed,
            train_ratio=splits_cfg["train"],
            dev_ratio=splits_cfg["dev_calibration"],
        )
        for split_name, rows in splits.items():
            save_jsonl(rows, out_root / name / f"{split_name}.jsonl")
        print(
            f"  -> Saved {name}: train={len(splits['train'])}, "
            f"dev={len(splits['dev_calibration'])}, test={len(splits['test'])}"
        )


if __name__ == "__main__":
    main()