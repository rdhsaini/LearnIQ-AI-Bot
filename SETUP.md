# LearnIQ — VS Code + Streamlit Setup Guide

## Your final folder layout
```
learniq_chatbot/
├── LearnIQ_Content/              ← put ALL your PDFs here
│   ├── NCERT_Class8_Science.pdf
│   ├── Worksheets/
│   │   └── worksheet1.pdf
│   └── Case Studies/
│       └── case1.pdf
├── chroma_db/                    ← auto-created on first run (Vector DB)
├── app.py                        ← the full app
├── requirements.txt
├── .env                          ← your API key (never commit to git)
└── .env.example
```

---

## Step-by-step: run in VS Code

### 1. Open the folder in VS Code
File → Open Folder → select `learniq_chatbot/`

### 2. Open the integrated terminal
Terminal → New Terminal  (or Ctrl+`)

### 3. Create a virtual environment (recommended)
```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Set your OpenAI API key

**Option A — .env file (easiest)**
Create a file called `.env` in the project root:
```
OPENAI_API_KEY=sk-your-key-here
```

**Option B — terminal**
```bash
# Mac/Linux
export OPENAI_API_KEY="sk-your-key-here"

# Windows CMD
setx OPENAI_API_KEY "sk-your-key-here"
# then restart the terminal

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-your-key-here"
```

### 6. Add your PDFs
Copy `NCERT_Class8_Science.pdf` into `LearnIQ_Content/`

### 7. Run the app
```bash
streamlit run app.py
```

Browser opens automatically at http://localhost:8501

---

## What happens on first run
1. Reads all PDFs from `LearnIQ_Content/`
2. Splits into ~1,500 chunks
3. Calls OpenAI to embed every chunk (~$0.03, ~30 seconds)
4. Saves vectors to `chroma_db/` on disk

## What happens on every run after that
- Loads `chroma_db/` from disk instantly (no API calls, no cost)
- Ready to answer questions in < 2 seconds

---

## Where LangChain and ChromaDB are used

| Component | What it does | Where in app.py |
|-----------|-------------|-----------------|
| `PyPDFDirectoryLoader` | Reads all PDFs | `load_and_chunk_pdfs()` |
| `RecursiveCharacterTextSplitter` | Cuts into 500-char chunks | `load_and_chunk_pdfs()` |
| `OpenAIEmbeddings` | Turns text → 1536-dim vectors | `get_embeddings()` |
| `Chroma` (Vector DB) | Stores + searches vectors on disk | `build_or_load_vectorstore()` |
| `RetrievalQA` (LangChain) | Retrieves chunks + calls GPT | `build_qa_chain()` |
| `@st.cache_resource` | Runs pipeline once, reuses | `initialise_pipeline()` |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `OPENAI_API_KEY not set` | Check `.env` file or terminal export |
| `No PDFs found` | Make sure PDFs are inside `LearnIQ_Content/` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Re-embed from scratch | Delete the `chroma_db/` folder and re-run |
