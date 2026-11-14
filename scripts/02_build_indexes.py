"""Stage 2 CLI: Build and validate Hybrid RRF retrieval indexes across all 5 domains."""

from __future__ import annotations

import json
from pathlib import Path

from ccr_router.retrieval.hybrid_retriever import HybridRRFRetriever

DOMAINS = ("truthful_qa", "fever", "uk_healthcare", "uk_law", "uk_finance")


def main() -> None:
    for ds in DOMAINS:
        docs: list[str] = []
        doc_ids: list[str] = []
        for split in ("train.jsonl", "dev_calibration.jsonl", "test.jsonl"):
            path = Path("data/splits") / ds / split
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        ev = str(row.get("evidence", "")).strip()
                        if ev:
                            doc_ids.append(row["id"])
                            docs.append(ev)
        retriever = HybridRRFRetriever(use_dense=False)
        retriever.index(doc_ids, docs)
        print(f"[02_build_indexes] {ds}: indexed {len(docs)} passages.")


if __name__ == "__main__":
    main()
