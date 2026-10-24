# Free Local RAG Document Q&A (Ollama)

No paid API key or per-request API charge is required. Models run locally.

## 1. Install Ollama
Download and install from https://ollama.com/download, then keep it running.

## 2. Download models
Run in a terminal:
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```
The first download requires internet and disk space. Later inference can run locally. Speed depends on your computer.

## 3. Run the app
Windows:
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload a text-based PDF/TXT, click Process document, then ask questions. The app uses Ollama `/api/embed` for embeddings and `/api/chat` for generation.

## Limitations
- Scanned PDFs need OCR first.
- Chunks/embeddings live in Streamlit session memory; no persistent vector database.
- Small local models can make mistakes; inspect retrieved chunks.
- This app runs locally, not as a public hosted service.

## Resume entry (only after you run and test it)
**Local RAG Document Q&A Application** — Python, Streamlit, Ollama, Embeddings, Cosine Similarity
- Extracted and chunked text from PDF/TXT documents.
- Generated local embeddings and retrieved relevant passages using cosine similarity.
- Integrated a locally hosted LLM to answer questions using retrieved context.
- Displayed source chunks to inspect answer grounding.
