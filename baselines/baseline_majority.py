"""
baseline_majority_ud.py — CRF con features de Dependencias Universales para GastroCorp NER 2026.

Extiende el baseline CRF original añadiendo features sintácticas del árbol UD:
  - POS tag universal (NOUN, VERB, ADJ, …)
  - Relación de dependencia del token (nsubj, obj, amod, …)
  - POS y texto de la cabeza sintáctica
  - Relaciones de los hijos directos
  - Señal de verbo culinario en la cabeza (feature dominio-específico)

Uso:
    python baselines/baseline_majority_ud.py \\
        --train menu_train.jsonl \\
        --eval  menu_dev.jsonl \\
        --output predictions/menu_dev_crf_ud.csv

Requisitos adicionales:
    pip install spacy
    python -m spacy download es_core_news_lg
"""

import json
import csv
import argparse
from collections import defaultdict, Counter
from pathlib import Path

import spacy

# ─── Modelo spaCy (cargado una sola vez) ──────────────────────────────────────
# es_core_news_lg ofrece mejor precisión en texto gastronómico que el modelo sm.
# Si el entorno es muy limitado en memoria, sustituir por es_core_news_sm.
try:
    NLP = spacy.load("es_core_news_lg")
except OSError:
    raise OSError(
        "Modelo spaCy no encontrado. Ejecuta:\n"
        "  python -m spacy download es_core_news_lg"
    )

# ─── Utilidades UD ────────────────────────────────────────────────────────────

def tokens_to_spacy_doc(tokens: list[str]) -> spacy.tokens.Doc:
    """
    Convierte una lista de tokens pre-tokenizados en un Doc de spaCy,
    respetando la tokenización original del dataset (sin re-tokenizar).
    """
    doc = spacy.tokens.Doc(NLP.vocab, words=tokens)
    # Aplicar solo tagger y parser (no tokenizer, ya está hecho)
    for pipe_name in ("tagger", "parser"):
        if pipe_name in NLP.pipe_names:
            NLP.get_pipe(pipe_name)(doc)
    return doc


def child_deps(token) -> str:
    """Devuelve las relaciones de los hijos directos como string ordenado."""
    deps = sorted(child.dep_ for child in token.children)
    return "|".join(deps) if deps else "NONE"



# ─── Extracción de features ───────────────────────────────────────────────────

def extract_features(tokens: list[str], domain: str = None) -> list[dict]:
    """
    Construye la lista de feature-dicts para el CRF.

    Mantiene todas las features léxicas del baseline original y añade
    un bloque 'ud.*' con información del árbol de dependencias.
    """
    doc = tokens_to_spacy_doc(tokens)
    features = []

    for i, token in enumerate(doc):
        tok = token.text
        tok_lower = tok.lower()

        prev_tok  = tokens[i-1].lower() if i > 0             else "<START>"
        next_tok  = tokens[i+1].lower() if i < len(tokens)-1 else "<END>"
        prev2_tok = tokens[i-2].lower() if i > 1             else "<START2>"
        next2_tok = tokens[i+2].lower() if i < len(tokens)-2 else "<END2>"

        f = {
            "bias": 1.0,

            # ── Features léxicas originales ───────────────────────────────
            "word.lower": tok_lower,
            "word.isupper": tok.isupper(),
            "word.istitle": tok.istitle(),
            "word.isdigit": tok.isdigit(),

            "has_digit": any(c.isdigit() for c in tok),
            "has_hyphen": "-" in tok,

            "prefix_2": tok_lower[:2],
            "prefix_3": tok_lower[:3],
            "suffix_2": tok_lower[-2:],
            "suffix_3": tok_lower[-3:],

            "prev_word": prev_tok,
            "next_word": next_tok,
            "prev2_word": prev2_tok,
            "next2_word": next2_tok,

            "is_menu": 1 if domain == "menu" else 0,
            "is_recipe": 1 if domain == "recipe" else 0,

            "bigram_prev": prev_tok + "_" + tok_lower,
            "bigram_next": tok_lower + "_" + next_tok,
            "trigram_prev": prev2_tok + "_" + prev_tok + "_" + tok_lower,
            "trigram_next": tok_lower + "_" + next_tok + "_" + next2_tok,

            # ── Nuevas features ortográficas ──────────────────────────────

            # Shape del token
            # Ej:
            # "Licor"    -> Xxxxxx
            # "43"       -> dd
            # "Licor43"  -> Xxxxxdd
            "word.shape": "".join(
                "X" if c.isupper()
                else "x" if c.islower()
                else "d" if c.isdigit()
                else c
                for c in tok
            ),

            # Longitud del token
            "token_length": len(tok),

            # ── Features de Dependencias Universales ──────────────────────

            # POS universal del token actual
            "ud.pos": token.pos_,

            # Relación de dependencia
            "ud.dep": token.dep_,

            # Combinación dep + POS de la cabeza
            "ud.dep_head_pos": token.dep_ + "+" + token.head.pos_,

            # ¿Es raíz?
            "ud.is_root": token.dep_ == "ROOT",

            # Dependencias vecinas
            "ud.prev_dep": doc[i-1].dep_ if i > 0 else "<START>",
            "ud.next_dep": doc[i+1].dep_ if i < len(doc)-1 else "<END>",

            # Distancia al head
            "ud.dist_head": token.i - token.head.i,

            # Número de hijos sintácticos
            "ud.n_children": len(list(token.children)),

            # Posición secuencial
            "is_first_token": i == 0,
            "is_last_token": i == len(tokens)-1,
        }

        # Posición en la secuencia
        if i == 0:
            f["BOS"] = True
        else:
            f["prev_word.isupper"] = tokens[i-1].isupper()

        if i == len(tokens) - 1:
            f["EOS"] = True
        else:
            f["next_word.istitle"] = tokens[i+1].istitle()

        features.append(f)

    return features


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train_majority(train_records: list[dict]) -> dict:
    """Mapa token → etiqueta más frecuente (usado solo para referencia)."""
    token_tag_counts = defaultdict(Counter)
    for rec in train_records:
        tokens   = rec.get("tokens", [])
        ner_tags = rec.get("ner_tags", ["O"] * len(tokens))
        for token, tag in zip(tokens, ner_tags):
            token_tag_counts[token.lower()][tag] += 1
    return {
        token: counts.most_common(1)[0][0]
        for token, counts in token_tag_counts.items()
    }

def oversample_records(records: list[dict]) -> list[dict]:
    """
    Oversampling simple por secuencia para balancear entidades raras.
    """

    balanced = []

    for rec in records:
        tags = rec.get("ner_tags", [])

        multiplier = 1

        # Prioridad a clases raras
        if any("BRAND" in tag for tag in tags):
            multiplier = 4

        elif any("BEVERAGE" in tag for tag in tags):
            multiplier = 2

        elif any("DISH" in tag for tag in tags):
            multiplier = 2

        balanced.extend([rec] * multiplier)

    return balanced

def load_two_datasets(menu_path: str, recipe_path: str) -> list[dict]:
    menu_data   = load_jsonl(menu_path)
    recipe_data = load_jsonl(recipe_path)

    print(f"  Menu sequences   : {len(menu_data):,}")
    print(f"  Recipe sequences : {len(recipe_data):,}")

    combined = menu_data + recipe_data

    print(f"  Original training: {len(combined):,}")

    # ── Oversampling ─────────────────────────
    combined = oversample_records(combined)

    print(f"  Balanced training: {len(combined):,}")

    return combined

def predict_and_save(eval_records: list[dict], crf, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "token_index", "predicted_tag"])

        for rec in eval_records:
            seq_id = rec["id"]
            tokens = rec["tokens"]

            # domain="menu" por defecto en eval; ajustar si hay split de dominio
            features = extract_features(tokens, domain="menu")
            tags = crf.predict([features])[0]

            for idx, tag in enumerate(tags):
                writer.writerow([seq_id, idx, tag])
                total_tokens += 1

    print(f"  Predictions written: {total_tokens:,} tokens")
    print(f"  Output file: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CRF + UD features — GastroCorp NER 2026"
    )
    parser.add_argument("--train",  required=True, help="Training JSONL (menu)")
    parser.add_argument("--eval",   required=True, help="Evaluation JSONL")
    parser.add_argument("--output", required=True, help="Output CSV predictions")
    args = parser.parse_args()

    print(f"\n  GastroCorp NER 2026 — CRF + UD (menu + recipe)")
    print(f"  Train (menu): {args.train}")
    print(f"  Eval        : {args.eval}\n")

    # Cargar datos
    menu_data    = load_jsonl(args.train)
    recipe_data  = load_jsonl("recipe_train.jsonl")
    eval_records = load_jsonl(args.eval)

    print(f"  Menu sequences   : {len(menu_data):,}")
    print(f"  Recipe sequences : {len(recipe_data):,}")
    print(f"  Total training   : {len(menu_data) + len(recipe_data):,}")
    print(f"  Eval sequences   : {len(eval_records):,}\n")

    # Construir matrices de features con UD
    print("  Extrayendo features UD (esto tarda ~1-2 min en datasets grandes)…")
    X_train, y_train = [], []

    for rec in menu_data:
        X_train.append(extract_features(rec["tokens"], domain="menu"))
        y_train.append(rec["ner_tags"])

    for rec in recipe_data:
        X_train.append(extract_features(rec["tokens"], domain="recipe"))
        y_train.append(rec["ner_tags"])

    # Entrenar CRF
    from sklearn_crfsuite import CRF

    print("  Entrenando CRF…")
    crf = CRF(
        algorithm="lbfgs",
        c1=0.01,
        c2=0.1,
        max_iterations=300,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)
    print("  Modelo entrenado (CRF + UD)\n")

    # Predecir y guardar
    predict_and_save(eval_records, crf, args.output)

    print(f"\n  Evaluar con:")
    print(f"  python evaluate.py --gold {args.eval} --pred {args.output}")


if __name__ == "__main__":
    main()