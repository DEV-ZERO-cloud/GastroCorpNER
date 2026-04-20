"""
baseline_transformer.py — Fine-tuning de modelo preentrenado para NER gastronómico.

Entrena un modelo de token classification sobre el corpus GastroCorp y genera
predicciones en el formato requerido por Codabench.

Uso:
    # Entrenar y predecir sobre dev de menús
    python baseline_transformer.py \\
        --model bert-base-multilingual-cased \\
        --train menu_train.jsonl \\
        --eval  menu_dev.jsonl \\
        --output predictions/menu_dev_transformer.csv

    # Con XLM-RoBERTa (mejor rendimiento esperado)
    python baseline_transformer.py \\
        --model xlm-roberta-base \\
        --train menu_train.jsonl \\
        --eval  menu_dev.jsonl \\
        --output predictions/menu_dev_xlmr.csv \\
        --epochs 5 --batch_size 16

Modelos recomendados:
    bert-base-multilingual-cased        mBERT  — buen punto de partida
    dccuchile/bert-base-spanish-wwm-cased  BETO — mejor para textos en español
    xlm-roberta-base                    XLM-R  — estado del arte multilingüe

Requisitos:
    pip install transformers datasets torch seqeval
"""

import json
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import numpy as np


# ── Etiquetas IOB2 ────────────────────────────────────────────────────────────

LABELS = [
    "O",
    "B-DISH",      "I-DISH",
    "B-BEVERAGE",  "I-BEVERAGE",
    "B-INGREDIENT","I-INGREDIENT",
    "B-BRAND",     "I-BRAND",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS)}


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class NERDataset(Dataset):
    """Dataset para token classification con tokenización de subpalabras."""

    def __init__(self, records: list[dict], tokenizer, has_labels: bool = True):
        self.samples   = []
        self.has_labels = has_labels

        for rec in records:
            tokens    = rec["tokens"]
            ner_tags  = rec.get("ner_tags", ["O"] * len(tokens))
            seq_id    = rec["id"]

            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=128,
                padding="max_length",
            )

            # Alinear etiquetas con subpalabras
            word_ids = encoding.word_ids()
            label_ids = []
            prev_word_id = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)        # tokens especiales
                elif wid != prev_word_id:
                    label_ids.append(LABEL2ID.get(ner_tags[wid], 0))
                else:
                    # Subpalabra interior: usar I- o -100 según preferencia
                    tag = ner_tags[wid]
                    if tag.startswith("B-"):
                        label_ids.append(LABEL2ID.get("I-" + tag[2:], 0))
                    else:
                        label_ids.append(LABEL2ID.get(tag, 0))
                prev_word_id = wid

            self.samples.append({
                "seq_id":     seq_id,
                "tokens":     tokens,
                "word_ids":   word_ids,
                "input_ids":  encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels":     label_ids,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids":      torch.tensor(s["input_ids"]),
            "attention_mask": torch.tensor(s["attention_mask"]),
            "labels":         torch.tensor(s["labels"]),
        }


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Calcula F1 span-level usando seqeval."""
    try:
        from seqeval.metrics import f1_score, classification_report
        use_seqeval = True
    except ImportError:
        use_seqeval = False

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    true_labels, pred_labels = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq, pred_seq_out = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_seq.append(ID2LABEL[l])
                pred_seq_out.append(ID2LABEL[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_out)

    if use_seqeval:
        f1 = f1_score(true_labels, pred_labels, average="micro")
        return {"f1_micro": f1}
    else:
        # Cálculo manual simplificado
        tp = fp = fn = 0
        for ts, ps in zip(true_labels, pred_labels):
            for t, p in zip(ts, ps):
                if t != "O" and t == p:
                    tp += 1
                elif t != "O" and t != p:
                    fn += 1
                elif t == "O" and p != "O":
                    fp += 1
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        return {"f1_micro": round(f1, 4)}


# ── Generación de predicciones ────────────────────────────────────────────────

def predict_and_save(model, tokenizer, dataset_obj, output_path: str):
    """Genera predicciones en el formato requerido por Codabench."""
    model.eval()
    device = next(model.parameters()).device

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "token_index", "predicted_tag"])

        for sample in dataset_obj.samples:
            input_ids      = torch.tensor([sample["input_ids"]]).to(device)
            attention_mask = torch.tensor([sample["attention_mask"]]).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()

            # Mapear predicciones de subpalabras a tokens originales
            word_ids = sample["word_ids"]
            token_preds = {}
            for wid, pred_id in zip(word_ids, preds):
                if wid is not None and wid not in token_preds:
                    token_preds[wid] = ID2LABEL[pred_id]

            seq_id = sample["seq_id"]
            for tok_idx in sorted(token_preds.keys()):
                writer.writerow([seq_id, tok_idx, token_preds[tok_idx]])

    print(f"  Predictions saved to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning baseline for GastroCorp NER 2026"
    )
    parser.add_argument("--model",      default="bert-base-multilingual-cased",
                        help="HuggingFace model identifier")
    parser.add_argument("--train",      required=True, help="Training JSONL file")
    parser.add_argument("--eval",       required=True, help="Evaluation JSONL file")
    parser.add_argument("--output",     required=True, help="Output CSV file path")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--save_dir",   default="./model_output",
                        help="Directory to save the fine-tuned model")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  GastroCorp NER 2026 — Transformer Baseline")
    print(f"  Model : {args.model}")
    print(f"  Train : {args.train}")
    print(f"  Eval  : {args.eval}")
    print(f"{'='*60}\n")

    # Cargar tokenizer y modelo
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Cargar datos
    print("Loading data...")
    train_records = load_jsonl(args.train)
    eval_records  = load_jsonl(args.eval)
    print(f"  Train sequences: {len(train_records):,}")
    print(f"  Eval sequences:  {len(eval_records):,}")

    train_dataset = NERDataset(train_records, tokenizer, has_labels=True)
    eval_dataset  = NERDataset(eval_records,  tokenizer, has_labels=True)

    # Entrenamiento
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        logging_steps=50,
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nTraining...")
    trainer.train()

    # Generar predicciones
    print("\nGenerating predictions...")
    predict_and_save(trainer.model, tokenizer, eval_dataset, args.output)

    print("\nDone.")
    print(f"  Model saved to : {args.save_dir}")
    print(f"  Predictions    : {args.output}")
    print(f"\n  Evaluate locally:")
    print(f"  python evaluate.py --gold {args.eval} --pred {args.output}")


if __name__ == "__main__":
    main()
