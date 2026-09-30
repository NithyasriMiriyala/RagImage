# Image-Based RAG Question Answering System

Upload an image containing text (scan, screenshot, receipt, notes) and ask questions about it. Answers come **only** from the text found in the image.

**Pipeline:** OpenCV preprocessing → Tesseract OCR → text cleaning → LangChain chunking → HuggingFace embeddings → ChromaDB → Groq LLM (Llama 3)

## Project structure

```
image_rag/
├── rag_pipeline.py   # Core pipeline + CLI
├── app.py            # Streamlit UI
├── requirements.txt
└── README.md
```

## 1. Prerequisites

- Python 3.10+
- **Tesseract OCR engine** (separate from the pip package):
  - Ubuntu/Debian: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki, then set `TESSERACT_CMD` (see below)
- A free **Groq API key**: https://console.groq.com/keys

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the embedding model (~90 MB).

## 3. Configure

Create a `.env` file in the project folder:

```env
GROQ_API_KEY=gsk_your_key_here

# Optional overrides
# GROQ_MODEL=llama-3.1-8b-instant
# TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
# CHROMA_DIR=./chroma_db
# CHUNK_SIZE=500
# CHUNK_OVERLAP=80
# TOP_K=4
```

Or export it directly: `export GROQ_API_KEY=gsk_...` (Windows PowerShell: `$env:GROQ_API_KEY="gsk_..."`).

## 4. Run

**Streamlit UI**
```bash
streamlit run app.py
```

**CLI**
```bash
# Interactive
python rag_pipeline.py --image sample.png --show-text

# Single question
python rag_pipeline.py --image sample.png --question "What is the invoice total?"
```

**As a library**
```python
from rag_pipeline import ImageRAG

rag = ImageRAG()
rag.ingest("sample.png")
print(rag.ask("Summarize this image").text)
```

## How it works

| Step | What happens |
|------|--------------|
| 1. Preprocess | Grayscale → upscale small images → non-local-means denoise → Otsu or adaptive threshold (chosen by lighting) → ensure dark-on-light |
| 2. OCR | `pytesseract` with `--oem 3 --psm 6` |
| 3. Clean + chunk | Fix hyphenation and whitespace; `RecursiveCharacterTextSplitter` (500 chars, 80 overlap) |
| 4. Embed + store | `all-MiniLM-L6-v2` embeddings persisted in ChromaDB (one collection per image hash; re-ingesting the same image never duplicates) |
| 5. RAG chain | Top-k retrieval → strict "context only" prompt → `ChatGroq` (temperature 0) |
| 6. Query | `ImageRAG.ask(question)` returns the answer plus the retrieved chunks |

## Troubleshooting

- **"Tesseract OCR engine not found"** – install Tesseract and/or set `TESSERACT_CMD`.
- **"OCR found no readable text"** – use a higher-resolution or better-lit image; handwriting is poorly supported by Tesseract.
- **Groq auth / model errors** – check `GROQ_API_KEY`; if a model is retired, set `GROQ_MODEL` to a current one from https://console.groq.com/docs/models.
- **Garbled OCR** – adjust the `--psm` value in `extract_text` (`3` for full pages with columns, `11` for sparse text).
- Delete the `chroma_db/` folder to reset stored data.
