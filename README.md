# GastroCorp NER 2026 — Starter Kit

Official starter kit for the **GastroCorp NER 2026** shared task on Codabench.

**Task:** Named Entity Recognition in Spanish–English gastronomic texts  
**Dataset DOI:** [10.5281/zenodo.19183413](https://doi.org/10.5281/zenodo.19183413)  
**Codabench:** [Challenge page](https://www.codabench.org) *(link to be added)*

---

## Repository structure

```
gastrocorp-ner-2026/
├── baselines/
│   ├── baseline_majority.py     # Majority-class baseline (no ML required)
│   └── baseline_transformer.py  # Fine-tuning baseline (mBERT / BETO / XLM-R)
├── evaluate.py                  # Local evaluation script
├── strip_labels.py              # Organizer utility: generate participant files
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
# Download from Zenodo and place files in a data/ directory
# https://doi.org/10.5281/zenodo.19183413
```

### 3. Run the majority-class baseline

No GPU required. Good starting point to verify your setup.

```bash
python baselines/baseline_majority.py \
    --train data/menu_train.jsonl \
    --eval  data/menu_train.jsonl \
    --output predictions/menu_majority.csv

# Evaluate locally (use train as gold since dev has no labels)
python evaluate.py \
    --gold data/menu_train.jsonl \
    --pred predictions/menu_majority.csv
```

### 4. Fine-tune a transformer model

GPU recommended. Expected training time: ~15 min on a single GPU for menus.

```bash
python baselines/baseline_transformer.py \
    --model bert-base-multilingual-cased \
    --train data/menu_train.jsonl \
    --eval  data/menu_train.jsonl \
    --output predictions/menu_mbert.csv \
    --epochs 3
```

### 5. Submit to Codabench

```bash
zip submission.zip predictions/menu_dev_mbert.csv -j
# Upload submission.zip to Codabench
# The CSV inside must be named submission.csv
```

---

## Recommended models

| Model | HuggingFace ID | Notes |
|---|---|---|
| mBERT | `bert-base-multilingual-cased` | Good general baseline |
| BETO | `dccuchile/bert-base-spanish-wwm-cased` | Best for Spanish menus |
| XLM-RoBERTa | `xlm-roberta-base` | Best overall performance expected |

---

## Submission format

The CSV file inside your `submission.zip` must be named `submission.csv`:

```csv
sequence_id,token_index,predicted_tag
menu_000001,0,B-DISH
menu_000001,1,I-DISH
menu_000001,2,O
menu_000001,3,B-INGREDIENT
```

**Valid tags:** `O` · `B-DISH` · `I-DISH` · `B-BEVERAGE` · `I-BEVERAGE` · `B-INGREDIENT` · `I-INGREDIENT` · `B-BRAND` · `I-BRAND`

---

## Evaluation metric

Official metric: **Span-level Micro-F1 (exact boundary match)**  
A prediction is correct only if both the span boundaries and the entity type match exactly.

Use `evaluate.py` to compute this metric locally before submitting.

---

## Citation

```bibtex
@dataset{pena_gnecco2026gastrocorp,
  author    = {Peña Gnecco, Daniel Arturo},
  title     = {{GastroCorp: A Bilingual Gastronomic NER Corpus (Spanish--English)}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19183413},
  url       = {https://doi.org/10.5281/zenodo.19183413}
}
```

---

## Contact

Questions? Use the Codabench forum or contact the organizers through the challenge page.
