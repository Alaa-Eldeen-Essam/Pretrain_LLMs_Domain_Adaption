# WSL Setup

Run these commands inside WSL, not PowerShell.

All phase scripts resume or skip from their latest safe output by default. Add
`--restart` only when you intentionally want to rebuild that phase output.

## 1. Enter The Project

```bash
cd "/mnt/d/Behoos_AI/AI Projects/Training_Custom_LLM"
```

## 2. Create The Conda Environment

This step needs internet access unless all conda and pip packages are already
available in your local caches.

```bash
conda env create -f environment-wsl.yml
conda activate qwen_cpt
```

If the environment already exists:

```bash
conda activate qwen_cpt
```

## 3. Keep Caches On D:

```bash
export PROJECT_ROOT="/mnt/d/Behoos_AI/AI Projects/Training_Custom_LLM"
export HF_HOME="$PROJECT_ROOT/cache/hf_home"
export HF_HUB_CACHE="$PROJECT_ROOT/cache/hf_home/hub"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/cache/transformers"
export HF_DATASETS_CACHE="$PROJECT_ROOT/cache/datasets"
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache/xdg"
export TMPDIR="$PROJECT_ROOT/cache/temp"
```

You can copy these from `.env.example` into your shell profile later if wanted.

## 4. Install OCR System Tools

This step needs internet access unless these apt packages are already available
in your local apt cache or an offline mirror.

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng poppler-utils antiword catdoc
```

Required:

- `tesseract-ocr`
- `tesseract-ocr-ara`
- `tesseract-ocr-eng`
- `poppler-utils`

Optional for old `.doc` files:

- `antiword`
- `catdoc`

## 5. Check Environment Without Downloading Models

```bash
python scripts/00_check_wsl_environment.py
```

Use strict mode only when you want the command to fail if OCR dependencies are
missing:

```bash
python scripts/00_check_wsl_environment.py --strict
```

## 6. Build Data Offline

```bash
python scripts/01_build_inventory.py
python scripts/02_extract_text_ocr.py
python scripts/03_clean_text.py
python scripts/04_build_dataset.py
```

The OCR script is resumable at page level. It writes completed pages
incrementally to `data/ocr_raw/pages.jsonl`, resumes partially completed PDFs
from the next page, shows terminal progress bars, and writes progress to
`data/reports/ocr_progress.json`.

To inspect progress without OCR work:

```bash
python scripts/02_extract_text_ocr.py --status-only
```

To continue after stopping:

```bash
python scripts/02_extract_text_ocr.py
```

If WSL kills the OCR process because a PDF page is too large, reduce DPI:

```bash
python scripts/02_extract_text_ocr.py --dpi 200
```

To intentionally discard previous OCR output and start over:

```bash
python scripts/02_extract_text_ocr.py --restart
```

The same `--restart` pattern applies to inventory, cleaning, dataset, export,
Modelfile, and smoke-test phases.

## 7. Download Model Later

The download phase is intentionally guarded:

```bash
python scripts/05_download_model.py
```

That command writes a skipped report and performs no network access.

When you explicitly want the download:

```bash
python scripts/05_download_model.py --allow-download
```

To intentionally delete and re-download the local model:

```bash
python scripts/05_download_model.py --allow-download --restart
```

## 8. Train Later From Local Files

```bash
python scripts/06_train_unsloth_cpt.py --offline
```

This expects the base model to already exist under `models/base`.

If training is interrupted, rerun the same command. It resumes from the latest
`checkpoint-*` folder automatically.

To intentionally discard checkpoints and start from the base model:

```bash
python scripts/06_train_unsloth_cpt.py --offline --restart
```
