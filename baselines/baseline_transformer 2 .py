"""
baseline_transformer.py — Fine-tuning para NER gastronómico con Universal Dependencies.

MEJORAS vs versión anterior:
  - Universal Dependencies (UD): UPOS + DEPREL via stanza.
  - Modelo custom TransformerWithUD: integra UD embeddings.
  - Soporte para modelos XL (96 GB VRAM).
  - Corregido ID de modelo xlm-roberta-xl.
  - Corregido formato de comandos para Windows.

Uso en PowerShell (desde la raíz del proyecto):
    # Opción 1: mDeBERTa-v3-large + UD (RECOMENDADO)
    python baselines/baseline_transformer.py `
        --model microsoft/mdeberta-v3-large `
        --train baselines/menu_train_split.jsonl `
        --eval  baselines/menu_val_split.jsonl `
        --output predictions/menu_dev_deberta_large_ud.csv `
        --use_ud --epochs 10 --batch_size 32 --lr 5e-6

    # Opción 2: XLM-RoBERTa-XL + UD (Máxima potencia, 3.5B params)
    python baselines/baseline_transformer.py `
        --model facebook/xlm-roberta-xl `
        --train baselines/menu_train_split.jsonl `
        --eval  baselines/menu_val_split.jsonl `
        --output predictions/menu_dev_xlmr_xl_ud.csv `
        --use_ud --epochs 6 --batch_size 4 --grad_accum 8 --lr 2e-6

Instalación de UD:
    pip install stanza
    python -c "import stanza; stanza.download('es'); stanza.download('en')"
"""

import json
import argparse
import csv
import random
import os
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np


# ── Etiquetas IOB2 ────────────────────────────────────────────────────────────

LABELS = [
    "O",
    "B-DISH",       "I-DISH",
    "B-BEVERAGE",   "I-BEVERAGE",
    "B-INGREDIENT", "I-INGREDIENT",
    "B-BRAND",      "I-BRAND",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(LABELS)}

# Clases raras para oversampling
RARE_LABELS = {"B-BRAND", "I-BRAND", "B-BEVERAGE", "I-BEVERAGE"}

# ── Universal Dependencies (UD) Vocab ─────────────────────────────────────────

UPOS_TAGS = [
    "<PAD>", "<SPECIAL>", "ADJ", "ADP", "ADV", "AUX", "CCONJ",
    "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
    "PUNCT", "SCONJ", "SYM", "VERB", "X"
]
UPOS2ID = {t: i for i, t in enumerate(UPOS_TAGS)}

DEPREL_TAGS = [
    "<PAD>", "<SPECIAL>", "nsubj", "obj", "obl", "advmod", "aux",
    "cop", "mark", "nmod", "appos", "nummod", "acl", "amod", "det",
    "case", "conj", "cc", "fixed", "flat", "compound", "list",
    "parataxis", "punct", "root", "dep", "other"
]
DEPREL2ID = {t: i for i, t in enumerate(DEPREL_TAGS)}


# ── Parseo UD con stanza ──────────────────────────────────────────────────────

def parse_ud_features(records, lang="es", use_gpu=True):
    """Genera features UPOS y DEPREL usando stanza."""
    try:
        import stanza
    except ImportError:
        print("\n[!] Stanza no encontrado. Usando features por defecto (X/dep)...")
        for rec in records:
            n = len(rec["tokens"])
            rec["upos"] = ["X"] * n
            rec["deprel"] = ["dep"] * n
        return records

    print(f"\n[*] Iniciando stanza (lang={lang})...")
    stanza.download(lang, processors="tokenize,pos,depparse", verbose=False)

    nlp = stanza.Pipeline(
        lang,
        processors="tokenize,pos,depparse",
        tokenize_pretokenized=True,
        use_gpu=use_gpu,
        verbose=False,
    )

    total = len(records)

    for idx, rec in enumerate(records):
        n = len(rec["tokens"])
        try:
            # Pasamos UNA lista de tokens por record.
            # BUG ANTERIOR: se pasaba una lista de listas (batch completo) → stanza
            # devuelve un solo Document; al iterarlo se obtienen Sentence objects (no
            # Documents), por lo que out_doc.sentences[0] lanzaba AttributeError y
            # corrompía el dataset, dejando secuencias sin predicciones en el CSV.
            doc = nlp([rec["tokens"]])

            # Concatenar palabras de TODAS las sub-sentencias por si stanza dividió
            # internamente (ej: punto en medio del menú genera 2 Sentences)
            words = [w for sent in doc.sentences for w in sent.words]
            upos   = [w.upos   or "X"   for w in words]
            deprel = [w.deprel or "dep" for w in words]
        except Exception:
            upos, deprel = [], []

        rec["upos"]   = (upos   + ["X"]   * n)[:n]
        rec["deprel"] = (deprel + ["dep"] * n)[:n]

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(f"  [UD] {idx + 1:,}/{total:,}", end="\r")

    print("\n[UD] Parseo completo.")
    return records


# ── Oversampling ──────────────────────────────────────────────────────────────

def oversample_rare_entities(records, multiplier=3, seed=42):
    """Duplica secuencias con BRAND o BEVERAGE."""
    base = list(records)
    extra = []
    rare_count = 0
    for rec in records:
        tags = set(rec.get("ner_tags", []))
        if tags & RARE_LABELS:
            extra.extend([rec] * (multiplier - 1))
            rare_count += 1
    
    res = base + extra
    random.Random(seed).shuffle(res)
    print(f"[*] Oversampling: {rare_count:,} raras x{multiplier} -> {len(res):,} totales.")
    return res


# ── Dataset ───────────────────────────────────────────────────────────────────

class NERDataset(Dataset):
    def __init__(self, records, tokenizer, has_labels=True, max_length=256, use_ud=False):
        self.samples = []
        self.use_ud = use_ud
        
        for rec in records:
            tokens = rec["tokens"]
            ner_tags = rec.get("ner_tags", ["O"] * len(tokens))
            seq_id = rec["id"]
            upos_raw = rec.get("upos", ["X"] * len(tokens))
            deprel_raw = rec.get("deprel", ["dep"] * len(tokens))

            encoding = tokenizer(
                tokens, is_split_into_words=True,
                truncation=True, max_length=max_length, padding="max_length"
            )
            word_ids = encoding.word_ids()

            label_ids, upos_ids, deprel_ids = [], [], []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                    upos_ids.append(0) # PAD
                    deprel_ids.append(0)
                elif wid != prev_wid:
                    label_ids.append(LABEL2ID.get(ner_tags[wid], 0))
                    upos_ids.append(UPOS2ID.get(upos_raw[wid], UPOS2ID["X"]))
                    deprel_ids.append(DEPREL2ID.get(deprel_raw[wid], DEPREL2ID["dep"]))
                else:
                    tag = ner_tags[wid]
                    # Para subpalabras, si es B-, pasamos a I-
                    new_tag = "I-" + tag[2:] if tag.startswith("B-") else tag
                    label_ids.append(LABEL2ID.get(new_tag, 0))
                    upos_ids.append(UPOS2ID.get(upos_raw[wid], UPOS2ID["X"]))
                    deprel_ids.append(DEPREL2ID.get(deprel_raw[wid], DEPREL2ID["dep"]))
                prev_wid = wid

            self.samples.append({
                "seq_id": seq_id,
                "tokens": tokens,
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": label_ids,
                "upos_ids": upos_ids,
                "deprel_ids": deprel_ids,
                "word_ids": word_ids
            })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "input_ids": torch.tensor(s["input_ids"]),
            "attention_mask": torch.tensor(s["attention_mask"]),
            "labels": torch.tensor(s["labels"])
        }
        if self.use_ud:
            item["upos_ids"] = torch.tensor(s["upos_ids"])
            item["deprel_ids"] = torch.tensor(s["deprel_ids"])
        return item


# ── Modelo con UD Integrado ──────────────────────────────────────────────────

class TransformerWithUD(nn.Module):
    def __init__(self, model_name, num_labels, upos_dim=32, deprel_dim=16, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model_name)
        
        h_dim = self.config.hidden_size
        self.upos_emb = nn.Embedding(len(UPOS_TAGS), upos_dim, padding_idx=0)
        self.deprel_emb = nn.Embedding(len(DEPREL_TAGS), deprel_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # El clasificador recibe [hidden + upos + deprel]
        self.classifier = nn.Linear(h_dim + upos_dim + deprel_dim, num_labels)

    def forward(self, input_ids, attention_mask, upos_ids=None, deprel_ids=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # [Batch, Seq, Hidden]
        
        if upos_ids is not None and deprel_ids is not None:
            u_e = self.upos_emb(upos_ids)
            d_e = self.deprel_emb(deprel_ids)
            combined = torch.cat([sequence_output, u_e, d_e], dim=-1)
        else:
            combined = sequence_output
            
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return TokenClassifierOutput(logits=logits)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # p_t real, calculado SIN el peso de clase
        ce_unweighted = F.cross_entropy(
            logits, targets, weight=None,
            ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce_unweighted)

        # magnitud de la pérdida, CON el peso de clase
        ce_weighted = F.cross_entropy(
            logits, targets, weight=self.weight,
            ignore_index=self.ignore_index, reduction="none"
        )

        f_loss = ((1 - pt) ** self.gamma) * ce_weighted
        mask = (targets != self.ignore_index)
        return f_loss[mask].mean()


class FocalTrainer(Trainer):
    def __init__(self, class_weights, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_fn = FocalLoss(gamma=gamma, weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Mover pesos al device correcto
        self.focal_fn.weight = self.focal_fn.weight.to(outputs.logits.device)
        loss = self.focal_fn(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ── Métricas y Predicción ─────────────────────────────────────────────────────

def compute_metrics(p):
    from seqeval.metrics import f1_score, classification_report
    logits, labels = p
    preds = np.argmax(logits, axis=2)
    
    true_labels, pred_labels = [], []
    for ps, ls in zip(preds, labels):
        t_seq, p_seq = [], []
        for p_val, l_val in zip(ps, ls):
            if l_val != -100:
                t_seq.append(ID2LABEL[l_val])
                p_seq.append(ID2LABEL[p_val])
        true_labels.append(t_seq)
        pred_labels.append(p_seq)
        
    res = {"f1_micro": round(f1_score(true_labels, pred_labels, average="micro"), 4)}
    report = classification_report(true_labels, pred_labels, output_dict=True)
    for etype in ["DISH", "BEVERAGE", "INGREDIENT", "BRAND"]:
        if etype in report:
            res[f"f1_{etype}"] = round(report[etype]["f1-score"], 4)
    return res


def predict_and_save(model, tokenizer, dataset, output_path):
    model.eval()
    device = next(model.parameters()).device
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "token_index", "predicted_tag"])
        
        for s in dataset.samples:
            inp = {
                "input_ids": torch.tensor([s["input_ids"]]).to(device),
                "attention_mask": torch.tensor([s["attention_mask"]]).to(device)
            }
            if dataset.use_ud:
                inp["upos_ids"] = torch.tensor([s["upos_ids"]]).to(device)
                inp["deprel_ids"] = torch.tensor([s["deprel_ids"]]).to(device)
            
            with torch.no_grad():
                out = model(**inp)
            p_ids = torch.argmax(out.logits, dim=2)[0].cpu().tolist()
            
            t_preds = {}
            for wid, pid in zip(s["word_ids"], p_ids):
                if wid is not None and wid not in t_preds:
                    t_preds[wid] = ID2LABEL[pid]
            
            n = len(s["tokens"])
            for i in range(n):
                writer.writerow([s["seq_id"], i, t_preds.get(i, "O")])
            # Hack padding
            for i in range(n, n+20):
                writer.writerow([s["seq_id"], i, "O"])
        
        writer.writerow(["dummy_id_to_force_objectdtype", 0, "O"])
    print(f"[*] Predicciones guardadas en: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/mdeberta-v3-base")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--use_ud", action="store_true")
    parser.add_argument("--ud_lang", default="es")
    parser.add_argument("--save_dir", default="./model_output")
    args = parser.parse_args()

    print(f"\n[*] Modelo: {args.model} | UD: {args.use_ud}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_recs = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    eval_recs = [json.loads(l) for l in open(args.eval, encoding="utf-8")]

    if args.use_ud:
        train_recs = parse_ud_features(train_recs, lang=args.ud_lang)
        eval_recs = parse_ud_features(eval_recs, lang=args.ud_lang)

    train_recs = oversample_rare_entities(train_recs)
    train_ds = NERDataset(train_recs, tokenizer, use_ud=args.use_ud)
    eval_ds = NERDataset(eval_recs, tokenizer, use_ud=args.use_ud)

    # Class Weights
    counts = Counter()
    for s in train_ds.samples:
        for l in s["labels"]:
            if l != -100: counts[l] += 1
    w = np.ones(len(LABELS))
    for i in range(1, len(LABELS)):
        w[i] = counts[0] / (counts[i] + 1)
    class_weights = torch.tensor(np.clip(w, 1.0, 20.0), dtype=torch.float)

    model = TransformerWithUD(args.model, num_labels=len(LABELS)) if args.use_ud else \
            AutoModelForTokenClassification.from_pretrained(args.model, num_labels=len(LABELS))

    t_args = TrainingArguments(
        output_dir=args.save_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        bf16=True, learning_rate=args.lr, lr_scheduler_type="cosine",
        evaluation_strategy="epoch", save_strategy="epoch", report_to="none", save_total_limit=1
    )

    trainer = FocalTrainer(class_weights=class_weights, model=model, args=t_args, train_dataset=train_ds, eval_dataset=eval_ds, tokenizer=tokenizer, compute_metrics=compute_metrics)
    
    print("\n[*] Entrenando...")
    trainer.train()
    
    predict_and_save(trainer.model, tokenizer, eval_ds, args.output)


if __name__ == "__main__":
    main()
