"""
baseline_majority.py — Baseline de clase mayoritaria para GastroCorp NER 2026.

Para cada token, predice la etiqueta IOB2 más frecuente observada en el conjunto
de entrenamiento. Tokens desconocidos reciben la etiqueta O.

Uso:
    python baselines/baseline_majority.py \\
        --train menu_train.jsonl \\
        --eval  menu_dev.jsonl \\
        --output predictions/menu_dev_majority.csv

Resultado esperado:
    Este baseline establece el límite inferior de rendimiento. Cualquier
    modelo neuronal debería superarlo. F1 típico: 0.40-0.60.
"""

import json
import csv
import argparse
from collections import defaultdict, Counter
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train_majority(train_records: list[dict]) -> dict:
    """
    Construye el mapa token → etiqueta más frecuente a partir del entrenamiento.
    """
    token_tag_counts = defaultdict(Counter)

    for rec in train_records:
        tokens   = rec.get("tokens", [])
        ner_tags = rec.get("ner_tags", ["O"] * len(tokens))
        for token, tag in zip(tokens, ner_tags):
            token_tag_counts[token.lower()][tag] += 1

    majority_map = {
        token: counts.most_common(1)[0][0]
        for token, counts in token_tag_counts.items()
    }
    return majority_map


def predict_and_save(eval_records: list[dict],
                     majority_map: dict,
                     output_path: str,
                     default_tag: str = "O"):
    """
    Genera predicciones y las guarda en el formato CSV requerido por Codabench.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "token_index", "predicted_tag"])

        for rec in eval_records:
            seq_id = rec["id"]
            tokens = rec["tokens"]
            for idx, token in enumerate(tokens):
                tag = majority_map.get(token.lower(), default_tag)
                writer.writerow([seq_id, idx, tag])
                total_tokens += 1

    print(f"  Predictions written: {total_tokens:,} tokens")
    print(f"  Output file: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Majority-class baseline for GastroCorp NER 2026"
    )
    parser.add_argument("--train",  required=True, help="Training JSONL file")
    parser.add_argument("--eval",   required=True, help="Evaluation JSONL file")
    parser.add_argument("--output", required=True, help="Output CSV predictions file")
    args = parser.parse_args()

    print(f"\n  GastroCorp NER 2026 — Majority Baseline")
    print(f"  Train : {args.train}")
    print(f"  Eval  : {args.eval}\n")

    train_records = load_jsonl(args.train)
    eval_records  = load_jsonl(args.eval)

    print(f"  Training sequences : {len(train_records):,}")
    print(f"  Eval sequences     : {len(eval_records):,}")

    majority_map = train_majority(train_records)
    print(f"  Unique tokens seen : {len(majority_map):,}\n")

    predict_and_save(eval_records, majority_map, args.output)

    print(f"\n  Evaluate locally with:")
    print(f"  python evaluate.py --gold {args.eval} --pred {args.output}")


if __name__ == "__main__":
    main()
