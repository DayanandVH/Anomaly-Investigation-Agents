# Anomaly Investigation Agent

An agent that takes a flagged anomaly from a tabular fraud-detection model,
retrieves relevant business/case context (RAG), and produces a human-readable
explanation and recommended action — using LangGraph for orchestration.

Status: Week 1 — baseline detector done. RAG + agent layer coming next.

## Project structure
```
anomaly-agent/
  src/
    data.py            # loads real or synthetic transaction data
    detector.py        # IsolationForest baseline + per-feature deviation reasons
    knowledge_base.py  # RAG: fraud-pattern/business-rule case notes + retrieval (ChromaDB)
  data/                 # put creditcard.csv here (see below)
  notebooks/            # exploration
  tests/
```

## RAG knowledge base design note
The real dataset's `V1`..`V28` columns are anonymized PCA components with no
real-world meaning — so the knowledge base doesn't pretend to know what "V9"
means. Instead it holds general fraud-analytics patterns and business rules
keyed to the *shape* of an anomaly (how many features deviate, how strongly,
combined with the transaction amount). The agent (Week 3) builds a natural-
language description of a flagged row's shape and retrieves the most relevant
pattern(s) to reason with.

## Setup (Linux)

1. **Python environment (3.11 required)**

   Your system Python (3.7) is too old for `chromadb` and recent LangChain/
   LangGraph versions. Since you're using conda, the cleanest fix is to
   create a fresh environment pinned to Python 3.11:
   ```bash
   cd anomaly-agent
   conda create -n anomaly-agent python=3.11 -y
   conda activate anomaly-agent
   pip install -r requirements.txt
   ```
   (If you don't use conda: install Python 3.11 via your distro — on
   Ubuntu, `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install
   python3.11 python3.11-venv`, then `python3.11 -m venv venv` instead of
   `python3 -m venv venv`.)

   Remember to run `conda activate anomaly-agent` at the start of every
   session before working on this project.

2. **Get the real dataset (optional but recommended)**
   Download the Kaggle "Credit Card Fraud Detection" dataset:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   Place `creditcard.csv` in `data/`. If it's not there, the code
   automatically falls back to a synthetic dataset with the same shape, so
   you can keep developing either way.

3. **Install Ollama** (local LLM backend)
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.1:8b
   ```
   Test it's working: `ollama run llama3.1:8b "hello"`

   **Known issue:** on some older NVIDIA GPU/driver combinations, Ollama's
   backend can crash with `Unsupported device` (segfault) even in CPU-only
   mode. If you hit this and can't resolve it quickly, use the free-tier
   Groq fallback instead (see below) — the rest of the project works
   identically either way, since the LLM backend is swappable.

4. **Fallback: Groq free-tier API** (if local Ollama doesn't work)
   - Get a free API key at https://console.groq.com/keys (no credit card required)
   - `export GROQ_API_KEY="your-key-here"`
   - Run any script with the `--groq` flag instead of the default Ollama path

5. **Run the baseline detector**
   ```bash
   python src/detector.py
   ```
   You should see flagged rows with their top contributing features.

## Roadmap
- [x] Week 1: baseline anomaly detector with per-feature reasons
- [x] Week 2: RAG knowledge base (case notes / business rules) + retrieval
- [x] Week 3: LangGraph agent (retrieve → reason → explain → recommend)
- [ ] Week 4: Streamlit demo UI
- [ ] Week 5: evaluation, polish, deploy
