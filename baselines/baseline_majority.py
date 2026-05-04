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

def extract_features(tokens, domain=None):
    features = []

    for i in range(len(tokens)):
        tok = tokens[i]
        tok_lower = tok.lower()

        prev_tok = tokens[i-1].lower() if i > 0 else "<START>"
        next_tok = tokens[i+1].lower() if i < len(tokens)-1 else "<END>"

        f = {
            "bias": 1.0,

            # Token actual
            "word.lower": tok_lower,
            "word.isupper": tok.isupper(),
            "word.istitle": tok.istitle(),
            "word.isdigit": tok.isdigit(),

            # Forma
            "has_digit": any(c.isdigit() for c in tok),
            "has_hyphen": "-" in tok,

            # Prefijos / sufijos
            "prefix_2": tok_lower[:2],
            "prefix_3": tok_lower[:3],
            "suffix_2": tok_lower[-2:],
            "suffix_3": tok_lower[-3:],

            # Contexto (CLAVE)
            "prev_word": prev_tok,
            "next_word": next_tok,

            "is_menu": 1 if domain == "menu" else 0,
            "is_recipe": 1 if domain == "recipe" else 0,


            # N-gramas
            "bigram_prev": prev_tok + "_" + tok_lower,
            "bigram_next": tok_lower + "_" + next_tok,
        }

        # Inicio
        if i == 0:
            f["BOS"] = True
        else:
            f["prev_word.isupper"] = tokens[i-1].isupper()

        # Fin
        if i == len(tokens) - 1:
            f["EOS"] = True
        else:
            f["next_word.istitle"] = tokens[i+1].istitle()

        features.append(f)

    return features

def load_two_datasets(menu_path, recipe_path):
    menu_data = load_jsonl(menu_path)
    recipe_data = load_jsonl(recipe_path)

    print(f"  Menu sequences   : {len(menu_data):,}")
    print(f"  Recipe sequences : {len(recipe_data):,}")

    combined = menu_data + recipe_data

    print(f"  Total training   : {len(combined):,}")

    return combined

def predict_and_save(eval_records, crf, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "token_index", "predicted_tag"])

        for rec in eval_records:
            seq_id = rec["id"]
            tokens = rec["tokens"]

            features = extract_features(tokens, domain="menu")
            tags = crf.predict([features])[0]

            for idx, tag in enumerate(tags):
                writer.writerow([seq_id, idx, tag])
                total_tokens += 1

    print(f"  Predictions written: {total_tokens:,} tokens")
    print(f"  Output file: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Majority-class baseline for GastroCorp NER 2026"
    )
    parser.add_argument("--train",  required=True, help="Training JSONL file (menu)")
    parser.add_argument("--eval",   required=True, help="Evaluation JSONL file")
    parser.add_argument("--output", required=True, help="Output CSV predictions file")
    args = parser.parse_args()

    print(f"\n  GastroCorp NER 2026 — CRF (menu + recipe)")
    print(f"  Train (menu): {args.train}")
    print(f"  Eval        : {args.eval}\n")

    # 🔹 Cargar datasets por separado
    menu_data   = load_jsonl(args.train)
    recipe_data = load_jsonl("recipe_train.jsonl")
    eval_records = load_jsonl(args.eval)

    print(f"  Menu sequences   : {len(menu_data):,}")
    print(f"  Recipe sequences : {len(recipe_data):,}")
    print(f"  Total training   : {len(menu_data) + len(recipe_data):,}")
    print(f"  Eval sequences   : {len(eval_records):,}")

    # 🔹 Preparar datos
    X_train = []
    y_train = []

    # MENU
    for rec in menu_data:
        tokens = rec["tokens"]
        tags = rec["ner_tags"]

        X_train.append(extract_features(tokens, domain="menu"))
        y_train.append(tags)

    # RECIPE
    for rec in recipe_data:
        tokens = rec["tokens"]
        tags = rec["ner_tags"]

        X_train.append(extract_features(tokens, domain="recipe"))
        y_train.append(tags)

    # 🔹 Entrenar CRF
    from sklearn_crfsuite import CRF

    print("\n  Entrenando CRF...")
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    print("  Model trained (CRF)\n")

    # 🔹 Predicción (IMPORTANTE: domain="menu")
    predict_and_save(eval_records, crf, args.output)

    print(f"\n  Evaluate locally with:")
    print(f"  python evaluate.py --gold menu_train.jsonl --pred predictions/menu_train_crf.csv")

if __name__ == "__main__":
    main()