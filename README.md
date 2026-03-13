# LearnIQ — CBSE Grade 8 Science AI Tutor
> RAG chatbot · LangChain + ChromaDB + GPT-4o-mini + Streamlit

---

## Quick Start (Local)

### 1. Clone / copy project
```
learniq_chatbot/
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

| OS | Command |
|----|---------|
| macOS / Linux | `export OPENAI_API_KEY="sk-..."` |
| Windows CMD | `setx OPENAI_API_KEY "sk-..."` then restart terminal |
| Windows PowerShell | `$env:OPENAI_API_KEY = "sk-..."` |
| .env file (any OS) | Copy `.env.example` → `.env`, fill in key |

### 4. Place your PDF

Put `NCERT_Class8_Science.pdf` in the project folder **or** update the
path in the Streamlit sidebar to the Google Drive mirror path:

- **Mac** (Google Drive for Desktop — Mirror):
  `/Users/you/Google Drive/My Drive/LearnIQ_Content/NCERT_Class8_Science.pdf`
- **Windows**:
  `G:\My Drive\LearnIQ_Content\NCERT_Class8_Science.pdf`

### 5. Run the app
```bash
streamlit run app.py
```

First run embeds and indexes all 280 pages (~30 seconds, ~$0.03).
Subsequent runs load from disk instantly.

---

## Google Drive Folder Structure
```
Google Drive
└── LearnIQ_Content/
    ├── NCERT_Class8_Science.pdf   ← main source
    ├── Worksheets/
    └── Case Studies/
```
Point the sidebar path input to the local mirror of the PDF above.

---

## Checkpoints (quick reference)

### 🔧 Change subject/grade (1 variable)
In `app.py`, edit the `SUBJECT_CONFIG` dict (~line 30):
```python
SUBJECT_CONFIG = {
    "grade": "9",           # ← change grade here
    "subject": "Math",      # ← change subject here
    "board": "CBSE",
    "pdf_filename": "NCERT_Class9_Math.pdf",
    "chroma_dir": "./chroma_db_class9_math",
}
```

### 🔧 Swap ChromaDB → Pinecone (2 lines in `build_or_load_vectorstore`)
```python
# REMOVE these 2 lines:
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)

# ADD these 2 lines:
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore(index_name="learniq", embedding=embeddings)
```
Also: `pip install langchain-pinecone` and set `PINECONE_API_KEY` env var.

---

## Risk Review

| Risk | Description | Fix |
|------|-------------|-----|
| **Hallucination** | Model may extrapolate beyond retrieved chunks for tricky questions | Add `temperature=0` and a stricter system prompt; log and review low-confidence answers |
| **Cost** | Session stats show *estimated* cost; actual cost depends on chunk length retrieved | Add a hard stop: `if q_count > 300: st.stop()` to cap at ~$0.05 |
| **Deployment** | ChromaDB `persist_directory` writes to the container filesystem — wiped on Streamlit Cloud redeploy | For cloud: use Chroma's HTTP client mode or swap to Pinecone (see above) |

---

## Cost Estimate (entire demo period)
| Item | Cost |
|------|------|
| One-time embedding (280 pages) | ~$0.03 |
| 1,000 student Q&As | ~$0.15 |
| **Total** | **~$0.18** |
Well within the $5 budget.
