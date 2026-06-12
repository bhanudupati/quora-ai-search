import os
import re
import chromadb
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from serpapi import GoogleSearch

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Quora AI Search",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=DM+Serif+Display:ital@0;1&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f !important;
    color: #e2e2e8;
}

.stApp {
    background-color: #0a0a0f !important;
}

#MainMenu, footer, header,
.stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container {
    max-width: 640px !important;
    padding: 0 2rem 5rem !important;
    margin: 0 auto !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}

.block-container > div {
    width: 100% !important;
}

/* Answer card via st.container(key=...) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(div.st-key-answer_card) > div,
.st-key-answer_card {
    background: #13131a !important;
    border: 1px solid #1e1e2a !important;
    border-left: 3px solid #b06a3a !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.8rem !important;
}

/* ── HERO ── */
.hero {
    padding: 5rem 0 3rem;
    text-align: center;
    width: 100%;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #c97d4e;
    border: 1px solid rgba(201,125,78,0.25);
    border-radius: 999px;
    padding: 5px 14px;
    margin-bottom: 1.6rem;
}

.hero-eyebrow-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #c97d4e;
    display: inline-block;
}

.hero-title {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-weight: 400 !important;
    font-size: 3.2rem;
    line-height: 1.12;
    letter-spacing: -0.02em;
    color: #f0ede8;
    margin-bottom: 1rem;
}

.hero-title .line1 {
    color: #6a6a7a;
    display: block;
}

.hero-title .line2 {
    font-style: italic;
    color: #c97d4e;
    display: block;
}

.hero-sub {
    font-size: 1rem;
    font-weight: 300;
    color: #6a6a7a;
    line-height: 1.65;
    max-width: 400px;
    margin: 0 auto;
    text-align: center;
}

.block-label {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a4a5a;
    margin-bottom: 0.7rem;
    padding-left: 2px;
}

/* Global text contrast for all rendered markdown */
.stMarkdown p,
.stMarkdown li,
.stMarkdown ol,
.stMarkdown ul {
    color: #c8c8d4 !important;
    font-size: 0.96rem !important;
    line-height: 1.75 !important;
}

.stMarkdown strong {
    color: #f5f2ed !important;
    font-weight: 600 !important;
}

.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-weight: 400 !important;
    color: #f0ede8 !important;
    margin: 1.4rem 0 0.6rem !important;
    font-size: 1.3rem !important;
}

.stMarkdown h1:first-child,
.stMarkdown h2:first-child,
.stMarkdown h3:first-child {
    margin-top: 0 !important;
}

/* ── SEARCH SECTION ── */
.search-section {
    margin-top: 2.8rem;
    width: 100%;
}

div[data-testid="stTextInput"],
div[data-testid="stButton"] {
    width: 100% !important;
}

/* Streamlit input override */
.stTextInput > label { display: none !important; }

.stTextInput > div > div {
    background: #13131a !important;
    border: 1px solid #22222e !important;
    border-radius: 12px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.stTextInput > div > div:focus-within {
    border-color: #6b4a30 !important;
    box-shadow: 0 0 0 3px rgba(176,106,58,0.08) !important;
}

.stTextInput > div > div > input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #e2e2e8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    padding: 0.75rem 1rem !important;
    caret-color: #c97d4e !important;
}

.stTextInput > div > div > input::placeholder {
    color: #2e2e3e !important;
}

/* Button */
.stButton { margin-top: 0.6rem; }

.stButton > button {
    background: #b06a3a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    width: 100% !important;
    transition: background 0.18s ease, transform 0.1s ease !important;
}

.stButton > button:hover {
    background: #c97d4e !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── PIPELINE ── */
.pipeline {
    margin: 1.8rem 0 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.step-row {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.82rem;
    color: #38384a;
}

.step-row.active { color: #9a9aaa; }
.step-row.done   { color: #6a6a7a; }

.step-node {
    width: 20px; height: 20px;
    border-radius: 50%;
    border: 1px solid #22222e;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: 0.65rem;
    font-weight: 600;
    color: #38384a;
}

.step-row.done .step-node {
    background: #b06a3a;
    border-color: #b06a3a;
    color: #fff;
}

/* ── ANSWER BLOCK ── */
.answer-block {
    margin-top: 2rem;
    width: 100%;
}

/* ── SOURCES ── */
.sources-block {
    margin-top: 1.4rem;
}

.source-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 0.5rem;
}

.source-pill {
    font-size: 0.73rem;
    color: #5a5a6a;
    background: #13131a;
    border: 1px solid #1e1e2a;
    border-radius: 999px;
    padding: 4px 12px;
    text-decoration: none !important;
    transition: border-color 0.15s, color 0.15s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 280px;
    display: inline-block;
}

.source-pill:hover {
    border-color: #b06a3a;
    color: #c97d4e;
}

/* ── DIVIDER ── */
.section-divider {
    border: none;
    border-top: 1px solid #14141e;
    margin: 2.8rem 0 2rem;
}

/* ── HISTORY ── */
.history-block { }

.history-item {
    padding: 0.9rem 0;
    border-bottom: 1px solid #111118;
}

.history-item:last-child { border-bottom: none; }

.history-q {
    font-size: 0.85rem;
    font-weight: 500;
    color: #b06a3a;
    margin-bottom: 4px;
}

.history-a {
    font-size: 0.8rem;
    color: #38384a;
    line-height: 1.55;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* Spinner */
[data-testid="stSpinner"] > div {
    border-top-color: #b06a3a !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────

load_dotenv(dotenv_path=".env")

@st.cache_resource
def init_clients():
    m = SentenceTransformer("all-MiniLM-L6-v2")
    g = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return m, g

model, groq_client = init_clients()
chroma_client = chromadb.Client()

if "collection" not in st.session_state:
    st.session_state.collection = chroma_client.create_collection("quora_search")
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# RAG FUNCTIONS
# ─────────────────────────────────────────────

def fetch_quora_data(query, num=10):
    search = GoogleSearch({
        "q": f"site:quora.com {query}",
        "num": num, "gl": "us", "hl": "en",
        "api_key": os.getenv("SERPAPI_KEY")
    })
    docs = []
    for item in search.get_dict().get("organic_results", []):
        title   = item.get("title", "")
        snippet = item.get("snippet", "")
        link    = item.get("link", "")
        if snippet:
            docs.append({"content": f"Title: {title}\nAnswer snippet: {snippet}", "title": title, "link": link})
    return docs

def index_docs(docs, qid):
    for i, doc in enumerate(docs):
        emb = model.encode(doc["content"]).tolist()
        st.session_state.collection.add(
            documents=[doc["content"]],
            embeddings=[emb],
            ids=[f"{qid}_{i}"]
        )

def retrieve(query, k=5):
    n = st.session_state.collection.count()
    if n == 0: return []
    res = st.session_state.collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=min(k, n)
    )
    return res["documents"][0]

def generate_answer(query, chunks):
    ctx = "\n\n---\n\n".join(chunks)
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You synthesize Quora discussions into clear, well-structured answers. Use markdown with ## headers and bullet points where helpful. Be detailed but concise."},
            {"role": "user", "content": f"QUORA CONTEXT:\n{ctx}\n\nQUESTION: {query}\n\nProvide a comprehensive, well-structured answer using markdown formatting."}
        ],
        temperature=0.5, max_tokens=1024
    )
    return resp.choices[0].message.content

def run_rag(query):
    qid  = re.sub(r"\W+", "_", query[:20])
    docs = fetch_quora_data(query)
    if not docs:
        return "No Quora results found. Try rephrasing your question.", []
    index_docs(docs, qid)
    chunks = retrieve(query)
    return generate_answer(query, chunks), docs

# ─────────────────────────────────────────────
# UI — HERO
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">
        <span class="hero-eyebrow-dot"></span>
        Powered by real Quora discussions
    </div>
    <h1 class="hero-title"><span class="line1">Ask anything.</span><span class="line2">Get real answers.</span></h1>
    <p class="hero-sub">Searches live Quora discussions and synthesizes them into a single, clear answer using AI.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UI — SEARCH
# ─────────────────────────────────────────────

query = st.text_input(
    "",
    placeholder="e.g. How do I become an AI engineer in 2026?",
    label_visibility="collapsed"
)
search_clicked = st.button("Search Quora →")

# ─────────────────────────────────────────────
# UI — RESULTS
# ─────────────────────────────────────────────

if search_clicked and query.strip():
    with st.spinner(""):
        status = st.empty()

        status.markdown("""
        <div class="pipeline">
            <div class="step-row active"><div class="step-node">1</div>Searching Quora via SerpAPI...</div>
            <div class="step-row"><div class="step-node">2</div>Indexing into ChromaDB</div>
            <div class="step-row"><div class="step-node">3</div>Generating answer with Groq</div>
        </div>""", unsafe_allow_html=True)

        answer, docs = run_rag(query.strip())

        status.markdown("""
        <div class="pipeline">
            <div class="step-row done"><div class="step-node">✓</div>Fetched Quora results</div>
            <div class="step-row done"><div class="step-node">✓</div>Indexed into ChromaDB</div>
            <div class="step-row done"><div class="step-node">✓</div>Generated answer with Groq</div>
        </div>""", unsafe_allow_html=True)

    # Answer label
    st.markdown('<div class="answer-block"><div class="block-label">Answer</div></div>', unsafe_allow_html=True)

    # Answer rendered inside a styled card using a container
    with st.container(key="answer_card"):
        st.markdown(answer)

    # Sources
    if docs:
        pills = "".join([
            f'<a class="source-pill" href="{d["link"]}" target="_blank">↗ {d["title"][:55]}{"…" if len(d["title"])>55 else ""}</a>'
            for d in docs[:6]
        ])
        st.markdown(f"""
        <div class="sources-block">
            <div class="block-label">Sources</div>
            <div class="source-pills">{pills}</div>
        </div>""", unsafe_allow_html=True)

    # Save to history
    st.session_state.history.insert(0, {"q": query.strip(), "a": answer[:260] + "…"})

# ─────────────────────────────────────────────
# UI — HISTORY
# ─────────────────────────────────────────────

if st.session_state.history:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="block-label">Recent searches</div>', unsafe_allow_html=True)
    items = "".join([
        f'<div class="history-item"><div class="history-q">↳ {h["q"]}</div><div class="history-a">{h["a"]}</div></div>'
        for h in st.session_state.history[:5]
    ])
    st.markdown(f'<div class="history-block">{items}</div>', unsafe_allow_html=True)