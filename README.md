# Qwen2.5 Arabic Domain Continued-Pretraining Pipeline

This is a WSL-first, phase-based pipeline for preparing an unlabeled Arabic
reference corpus and continued-pretraining `Qwen2.5-3B-Instruct` with Unsloth.

The source corpus is configured as:

`/mnt/d/Behoos_AI/AI Projects/mini_rag/test_mini_rag/مراجع`

All outputs, models, caches, logs, and temporary files stay under this project:

`/mnt/d/Behoos_AI/AI Projects/Training_Custom_LLM`

## Design Rules

- Each numbered script is standalone and can be run independently after its
  previous phase output exists.
- Scripts validate only their direct inputs and write only their own outputs.
- PDFs are OCRed through rendered images and Tesseract. No PyMuPDF embedded text
  extraction is used.
- Conda environment creation and system package installation require internet
  unless the packages are already cached locally.
- The only phase that should use the internet is `05_download_model.py`, and it
  refuses to download unless `--allow-download` is passed.
- The training dataset is plain JSONL with a single `text` field, suitable for
  continued pretraining on unlabeled text.

## Phase Map

1. `scripts/00_check_wsl_environment.py`
   Checks WSL, conda, Tesseract, Poppler, CUDA visibility, project-local cache
   paths, and source folder availability.

2. `scripts/01_build_inventory.py`
   Builds `data/inventory/source_manifest.jsonl` with hashes, file types, and
   relative paths for all source files.

3. `scripts/02_extract_text_ocr.py`
   Extracts text to `data/ocr_raw/pages.jsonl`. PDFs and images are OCRed with
   Tesseract; TXT and DOCX are read directly; legacy DOC uses `antiword` or
   `catdoc` if installed. The script appends each completed page immediately,
   resumes partially completed PDFs from the next page, shows terminal progress
   bars for documents and PDF pages, and writes progress/ETA to
   `data/reports/ocr_progress.json`.

4. `scripts/03_clean_text.py`
   Normalizes Arabic OCR text, removes zero-width marks/tashkeel if configured,
   and removes repeated page headers/footers.

5. `scripts/04_build_dataset.py`
   Converts cleaned pages into train/validation JSONL samples.

6. `scripts/05_download_model.py`
   Downloads the base model into `models/base` only when explicitly allowed.

7. `scripts/06_train_unsloth_cpt.py`
   Runs Unsloth LoRA continued pretraining from local model and dataset files.

8. `scripts/07_export_model.py`
   Exports the adapter to a merged model and optionally GGUF.

9. `scripts/08_create_ollama_modelfile.py`
   Creates an Ollama `Modelfile` for the exported GGUF.

10. `scripts/09_smoke_test_model.py`
    Checks exported artifacts and optionally runs a local Ollama prompt.

## Offline-Safe Order

Before downloads are available, run only:

```bash
cd "/mnt/d/Behoos_AI/AI Projects/Training_Custom_LLM"
conda activate qwen_cpt
python scripts/00_check_wsl_environment.py
python scripts/01_build_inventory.py
```

Run OCR only after Tesseract Arabic and Poppler are installed in WSL:

```bash
python scripts/02_extract_text_ocr.py
python scripts/03_clean_text.py
python scripts/04_build_dataset.py
```

Check OCR progress without doing OCR work:

```bash
python scripts/02_extract_text_ocr.py --status-only
```

Resume OCR after stopping:

```bash
python scripts/02_extract_text_ocr.py
```

This resumes from the next unprocessed page inside a PDF, not just the next
file.

If WSL kills the process on large PDFs, retry with lower DPI:

```bash
python scripts/02_extract_text_ocr.py --dpi 200
```

Restart OCR from zero:

```bash
python scripts/02_extract_text_ocr.py --restart
```

When internet credits are available, download explicitly:

```bash
python scripts/05_download_model.py --allow-download
```

Training and export should be run only after the dataset and local model exist:

```bash
python scripts/06_train_unsloth_cpt.py --offline
python scripts/07_export_model.py --offline
python scripts/08_create_ollama_modelfile.py --force
python scripts/09_smoke_test_model.py
```

## Configuration

- Paths: `configs/paths.yaml`
- OCR, dataset, model, training, and Ollama settings: `configs/training.yaml`
- Project-local cache variables: `.env.example`
