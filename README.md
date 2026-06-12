# 🔍 Quora AI Search

A Retrieval-Augmented Generation (RAG) agent that fetches real Quora discussions and synthesizes them into comprehensive, grounded answers using AI.

**🚀 Live App:** [quora-ai-search-4fxpklcfxurkse4nmnitbg.streamlit.app](https://quora-ai-search-4fxpklcfxurkse4nmnitbg.streamlit.app/)

---

## 🧠 What is this?

Instead of relying on a static dataset or an AI's general training knowledge, this agent:

1. **Searches Quora live** for your question using SerpAPI
2. **Stores results** as vector embeddings in ChromaDB
3. **Retrieves** the most semantically relevant chunks
4. **Generates** a detailed, grounded answer using Groq's Llama 3.3

The result: answers backed by real human discussions from Quora, not hallucinated responses.

---

## 🏗️ Architecture

```
User Question
      ↓
SerpAPI → searches "site:quora.com <question>"
      ↓
10 real Quora results fetched
      ↓
sentence-transformers → converts text to embeddings
      ↓
ChromaDB → stores & retrieves most relevant chunks
      ↓
Groq (Llama 3.3-70b) → synthesizes a detailed answer
      ↓
Answer displayed in Streamlit UI ✅
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | `ChromaDB` |
| Live Data | `SerpAPI` (Google search, site:quora.com) |
| LLM | `Groq` (llama-3.3-70b-versatile) |
| UI | `Streamlit` |
| Deployment | `Streamlit Community Cloud` |
| Environment | Python 3.12, Windows, venv |

---

## 📁 Project Structure

```
quora-ai-search/
├── app/
│   ├── streamlit_app.py      # Web UI (main app)
│   ├── rag_agent.py          # Core RAG pipeline (CLI version)
│   ├── create_embeddings.py  # Embedding utility
│   ├── load_data.py          # Data loader utility
│   ├── search.py             # Search utility
│   └── .env                  # API keys (not committed)
├── data/
│   └── quora_sample.json     # Sample Q&A dataset
├── venv/                      # Virtual environment
├── requirements.txt           # Python dependencies (pinned for deployment)
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation (Local)

### 1. Clone the repository
```bash
git clone https://github.com/bhanudupati/quora-ai-search.git
cd quora-ai-search
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file inside the `app/` folder:
```
GROQ_API_KEY=your_groq_api_key_here
SERPAPI_KEY=your_serpapi_key_here
```

Get your free API keys:
- **Groq** (free): https://console.groq.com
- **SerpAPI** (250 free searches/month): https://serpapi.com

---

## 🚀 Running the App

### Streamlit Web UI (recommended)
```bash
cd app
streamlit run streamlit_app.py --server.fileWatcherType none
```
Then open http://localhost:8501 in your browser.

### Command Line
```bash
cd app
python rag_agent.py
```

---

## 💡 How RAG Works (Simply Put)

Think of it like a research assistant:
- **Without RAG**: You ask an AI a question → it answers from memory (may hallucinate)
- **With RAG**: You ask a question → it *looks up* relevant sources first → then answers using those sources

This makes answers more accurate, current, and grounded in real content.

---

## ☁️ Deployment

This app is deployed on **Streamlit Community Cloud**.

Key deployment notes:
- Python version pinned to **3.12** (3.14 causes dependency conflicts)
- `chromadb==0.4.24` and `protobuf==4.25.3` pinned to avoid OpenTelemetry/protobuf conflicts
- `numpy<2.0.0` pinned for compatibility with `chromadb`'s `np.float_` usage
- API keys configured via Streamlit Cloud **Secrets** (not `.env`)

To deploy your own copy:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set main file path to `app/streamlit_app.py`
4. Add `GROQ_API_KEY` and `SERPAPI_KEY` in Advanced Settings → Secrets

---

## 🔑 API Keys & Security

- API keys are stored in `app/.env` (local) or Streamlit Secrets (cloud)
- `.env` is listed in `.gitignore` — **never commit it**
- Both Groq and SerpAPI offer free tiers sufficient for personal use

---

## 🛣️ Roadmap

- [x] Live Quora data retrieval via SerpAPI
- [x] RAG pipeline with ChromaDB + Groq
- [x] Premium Streamlit UI
- [x] Deploy to Streamlit Cloud
- [ ] Persistent ChromaDB storage across sessions
- [ ] Conversation memory for follow-up questions
- [ ] Caching to reduce redundant SerpAPI calls

---

## 📄 License

MIT License — feel free to use, modify, and distribute.