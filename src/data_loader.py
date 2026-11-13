"""
Data loading and stratified split management for TruthfulQA (ID) and FEVER (OOD).
Enforces strict anti-leakage boundaries across Train (20%), Dev (30%), and Test (50%).
"""

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def create_truthfulqa_splits(
    output_dir: str = "data/splits",
    split_ratios: tuple[float, float, float] = (0.20, 0.30, 0.50),
    seed: int = 42,
    force_recreate: bool = False,
) -> dict[str, list[dict]]:
    """
    Loads TruthfulQA (generation validation split, N=817) and creates stratified
    train/dev/test JSONL files preserving category distributions.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "truthfulqa_train.jsonl"
    dev_file = out_path / "truthfulqa_dev.jsonl"
    test_file = out_path / "truthfulqa_test.jsonl"

    if (
        not force_recreate
        and train_file.exists()
        and dev_file.exists()
        and test_file.exists()
    ):
        return {
            "train": load_jsonl(train_file),
            "dev": load_jsonl(dev_file),
            "test": load_jsonl(test_file),
        }

    assert np.isclose(sum(split_ratios), 1.0), "Split ratios must sum to 1.0"
    train_r, dev_r, test_r = split_ratios

    ds = load_dataset("truthful_qa", "generation", split="validation")
    records = []
    for idx, row in enumerate(ds):
        records.append({
            "id": f"tqa_{idx:04d}",
            "question": row["question"],
            "category": row["category"],
            "best_answer": row["best_answer"],
            "correct_answers": row["correct_answers"],
            "incorrect_answers": row["incorrect_answers"],
        })

    categories = [r["category"] for r in records]
    # Rare categories (<3 items) are grouped into 'Misc' so stratification never fails
    cat_counts = {c: categories.count(c) for c in set(categories)}
    strat_labels = [c if cat_counts[c] >= 3 else "Misc" for c in categories]

    # First split: Train (20%) vs Temp (80%)
    train_data, temp_data, _, temp_labels = train_test_split(
        records,
        strat_labels,
        train_size=train_r,
        random_state=seed,
        stratify=strat_labels,
    )

    # Second split: Dev (30% of total = 37.5% of temp) vs Test (50% of total = 62.5% of temp)
    relative_dev = dev_r / (dev_r + test_r)
    dev_data, test_data = train_test_split(
        temp_data,
        train_size=relative_dev,
        random_state=seed,
        stratify=temp_labels,
    )

    # Verify zero overlap
    train_ids = {r["id"] for r in train_data}
    dev_ids = {r["id"] for r in dev_data}
    test_ids = {r["id"] for r in test_data}
    assert len(train_ids & dev_ids) == 0, "Leakage between Train and Dev!"
    assert len(train_ids & test_ids) == 0, "Leakage between Train and Test!"
    assert len(dev_ids & test_ids) == 0, "Leakage between Dev and Test!"

    save_jsonl(train_data, train_file)
    save_jsonl(dev_data, dev_file)
    save_jsonl(test_data, test_file)

    return {"train": train_data, "dev": dev_data, "test": test_data}


def create_fever_ood_split(
    output_dir: str = "data/splits",
    eval_size: int = 1500,
    seed: int = 42,
    force_recreate: bool = False,
) -> list[dict]:
    """
    Creates a balanced binary (SUPPORTS vs REFUTES) out-of-domain evaluation subset
    from FEVER v1.0 paper_dev for zero-shot transfer testing.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fever_file = out_path / "fever_test.jsonl"

    if not force_recreate and fever_file.exists():
        return load_jsonl(fever_file)

    ds = load_dataset("fever", "v1.0", split="paper_dev", trust_remote_code=True)
    supports, refutes = [], []
    seen_claims = set()

    for idx, row in enumerate(ds):
        claim = row["claim"].strip()
        label = row["label"]
        if claim in seen_claims or label not in ("SUPPORTS", "REFUTES"):
            continue
        seen_claims.add(claim)
        item = {
            "id": f"fever_{idx:05d}",
            "question": f"Is the following claim factually true? Claim: {claim}",
            "claim": claim,
            "gold_label": 1 if label == "SUPPORTS" else 0,
            "raw_label": label,
        }
        if label == "SUPPORTS":
            supports.append(item)
        else:
            refutes.append(item)

    rng = np.random.default_rng(seed)
    half = eval_size // 2
    sel_sup = rng.choice(supports, size=min(half, len(supports)), replace=False).tolist()
    sel_ref = rng.choice(refutes, size=min(half, len(refutes)), replace=False).tolist()

    fever_eval = sel_sup + sel_ref
    rng.shuffle(fever_eval)
    save_jsonl(fever_eval, fever_file)
    return fever_eval


def save_jsonl(records: list[dict], filepath: Path) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(filepath: Path) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]