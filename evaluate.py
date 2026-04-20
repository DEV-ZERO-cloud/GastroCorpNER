"""
evaluate.py — Evaluación local para GastroCorp NER 2026.

Compara predicciones contra el gold standard y reporta métricas span-level.
Úsalo para validar tus predicciones localmente antes de subir a Codabench.

Uso:
    python evaluate.py \\
        --gold menu_train.jsonl \\
        --pred predictions/my_predictions.csv

    # Evaluar un subconjunto (e.g., solo DISH y BRAND)
    python evaluate.py \\
        --gold menu_train.jsonl \\
        --pred predictions/my_predictions.csv \\
        --labels DISH BRAND

Nota: Para validar durante el desarrollo, evalúa sobre el conjunto de
entrenamiento (ya que dev no incluye etiquetas). El servidor de Codabench
usa los labels reales de dev/test para el leaderboard oficial.
"""

import json
import csv
import argparse
from collections import defaultdict
from pathlib import Path


# ── Lectura de archivos ───────────────────────────────────────────────────────

def load_gold(path: str) -> dict[str, list[str]]:
    """Carga el gold standard desde JSONL. Retorna {seq_id: [tags]}."""
    gold = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "ner_tags" in rec:
                gold[rec["id"]] = rec["ner_tags"]
    return gold


def load_predictions(path: str) -> dict[str, dict[int, str]]:
    """Carga predicciones desde CSV. Retorna {seq_id: {token_idx: tag}}."""
    preds = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id    = row["sequence_id"]
            token_idx = int(row["token_index"])
            tag       = row["predicted_tag"].strip()
            preds[seq_id][token_idx] = tag
    return dict(preds)


# ── Extracción de spans ───────────────────────────────────────────────────────

def extract_spans(tags: list[str]) -> set[tuple]:
    """Extrae spans (entity_type, start, end) de una secuencia IOB2."""
    spans = set()
    start, cur_type = None, None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                spans.add((cur_type, start, i - 1))
            start, cur_type = i, tag[2:]
        elif tag.startswith("I-"):
            if start is None:
                start, cur_type = i, tag[2:]
        else:  # O
            if start is not None:
                spans.add((cur_type, start, i - 1))
                start, cur_type = None, None

    if start is not None:
        spans.add((cur_type, start, len(tags) - 1))

    return spans


# ── Evaluación ────────────────────────────────────────────────────────────────

def evaluate(gold_map: dict, pred_map: dict,
             filter_labels: list[str] = None) -> dict:
    """
    Calcula métricas span-level (exact match).
    Retorna diccionario con métricas globales y por entidad.
    """
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    missing_seqs = []

    for seq_id, gold_tags in gold_map.items():
        if seq_id not in pred_map:
            missing_seqs.append(seq_id)
            continue

        pred_tags_dict = pred_map[seq_id]
        pred_tags = [pred_tags_dict.get(i, "O") for i in range(len(gold_tags))]

        gold_spans = extract_spans(gold_tags)
        pred_spans = extract_spans(pred_tags)

        # Filtrar por labels si se especificó
        if filter_labels:
            gold_spans = {s for s in gold_spans if s[0] in filter_labels}
            pred_spans = {s for s in pred_spans if s[0] in filter_labels}

        for span in gold_spans:
            if span in pred_spans:
                per_label[span[0]]["tp"] += 1
            else:
                per_label[span[0]]["fn"] += 1

        for span in pred_spans:
            if span not in gold_spans:
                per_label[span[0]]["fp"] += 1

    if missing_seqs:
        print(f"  ⚠️  {len(missing_seqs)} sequences missing from predictions")
        print(f"     First missing: {missing_seqs[:3]}")

    # Calcular métricas por label
    results = {}
    total_tp = total_fp = total_fn = 0

    for label, counts in sorted(per_label.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p  = tp / (tp + fp) if (tp + fp) else 0.0
        r  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        results[label] = {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "support":   tp + fn,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Micro-average
    p_micro  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    r_micro  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) else 0.0

    results["MICRO-AVG"] = {
        "precision": round(p_micro,  4),
        "recall":    round(r_micro,  4),
        "f1":        round(f1_micro, 4),
        "support":   total_tp + total_fn,
    }
    return results


def print_results(results: dict):
    print(f"\n{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 64)
    for label, m in results.items():
        if label == "MICRO-AVG":
            continue
        print(f"{label:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10}")
    print("-" * 64)
    m = results["MICRO-AVG"]
    print(f"{'MICRO-AVG':<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
          f"{m['f1']:>10.4f} {m['support']:>10}")
    print()
    print(f"  → Official score (Micro-F1): {m['f1']:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Local evaluation for GastroCorp NER 2026"
    )
    parser.add_argument("--gold",   required=True,
                        help="Gold JSONL file (with ner_tags)")
    parser.add_argument("--pred",   required=True,
                        help="Predictions CSV (sequence_id, token_index, predicted_tag)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Optional: evaluate only these entity types")
    args = parser.parse_args()

    if not Path(args.gold).exists():
        print(f"ERROR: Gold file not found: {args.gold}")
        return
    if not Path(args.pred).exists():
        print(f"ERROR: Predictions file not found: {args.pred}")
        return

    gold_map = load_gold(args.gold)
    pred_map = load_predictions(args.pred)

    if not gold_map:
        print("ERROR: Gold file has no ner_tags. "
              "Use a training file for local evaluation.")
        return

    print(f"\n  Gold sequences : {len(gold_map):,}")
    print(f"  Pred sequences : {len(pred_map):,}")
    if args.labels:
        print(f"  Filter labels  : {args.labels}")

    results = evaluate(gold_map, pred_map, filter_labels=args.labels)
    print_results(results)


if __name__ == "__main__":
    main()
