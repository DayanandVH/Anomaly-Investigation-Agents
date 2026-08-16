# Anomaly Investigation Agent

An agent that takes a flagged anomaly from a tabular fraud-detection model,
retrieves relevant business/case context (RAG), and produces a human-readable
explanation and recommended action — using LangGraph for orchestration.

Status: Week 1 — baseline detector done. RAG + agent layer coming next.

## Project structure
```
anomaly-agent/
  src/
    data.py       # loads real or synthetic transaction data
    detector.py   # IsolationForest baseline + per-feature deviation reasons
  data/            # put creditcard.csv here (see below)
  notebooks/       # exploration
  tests/
```

## Setup (Linux)

1. **Python environment**
   ```bash
   cd anomaly-agent
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Get the real dataset (optional but recommended)**
   Download the Kaggle "Credit Card Fraud Detection" dataset:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   Place `creditcard.csv` in `data/`. If it's not there, the code
   automatically falls back to a synthetic dataset with the same shape, so
   you can keep developing either way.

3. **Install Ollama** (local LLM, used from Week 3 onward)
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.1:8b
   ```
   Test it's working: `ollama run llama3.1:8b "hello"`

   (If your machine is GPU/RAM constrained, `llama3.1:8b` needs ~8GB RAM;
   a lighter option is `qwen2.5:7b` or `phi3:mini`.)

4. **Run the baseline detector**
   ```bash
   python src/detector.py
   ```
   You should see flagged rows with their top contributing features.

## Roadmap
- [x] Week 1: baseline anomaly detector with per-feature reasons
- [ ] Week 2: RAG knowledge base (case notes / business rules) + retrieval
- [ ] Week 3: LangGraph agent (retrieve → reason → explain → recommend)
- [ ] Week 4: Streamlit demo UI
- [ ] Week 5: evaluation, polish, deploy
