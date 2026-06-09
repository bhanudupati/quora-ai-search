import os
import re
import streamlit as st
import chromadb
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

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background-color: #0f1117;
        color: #e8e8e8;
    }

    /* Hero title */
    .hero-title {
        font-family: 'Sora', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #b84040, #e05c5c, #f0a500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #1c1f2b !important;
        color: #f0f0f0 !important;
        border: 1px solid #2e3249 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #b84040 !important;
        box-shadow: 0 0 0 2px rgba(184, 64, 64, 0.2) !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #b84040, #e05c5c) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* Answer card */
    .answer-card {
        background-color: #1c1f2b;
        border-left: 3px solid #b84040;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-top: 1.5rem;
        color: #e0e0e0;
        line-height: 1.75;
        font-size: 0.97rem;
    }

    /* Source chips */
    .source-chip {
        display: inline-block;
        background-color: #252836;
        color: #aaa;
        font-size: 0.75rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 3px 3px 0 0;
        border: 1px solid #2e3249;
    }

    .source-chip a {
        color: #e05c5c;
        text-decoration: none;
    }

    /* Step badges */
    .step-badge {
        display: inline-block;
        background: #b84040;
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 20px;
        margin-right: 6px;
        font-family: 'Sora', sans-serif;
        letter-spacing: 0.05em;
    }

    .step-row {
        color: #888;
        font-size: 0.85rem;
        margin: 4px 0;
    }

    /* History item */
    .history-item {
        background: #161820;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border: 1px solid #1e2130;
    }

    .history-q {
        color: #e05c5c;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }

    .history-a {
        color: #aaa;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* Divider */
    hr {
        border-color: #1e2130 !important;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD ENV & INIT CLIENTS (cached)
# ─────────────────────────────────────────────

load_dotenv(dotenv_path=".env")

@st.cache_resource
def init_clients():
    model       = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return model, groq_client

model, groq_client = init_clients()

# Fresh ChromaDB collection per session
chroma_client = chromadb.Client()
if "collection" not in st.session_state:
    st.session_state.collection = chroma_client.create_collection("quora_search")

if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────
# RAG FUNCTIONS
# ─────────────────────────────────────────────

def fetch_quora_data(query, num_results=10):
    search = GoogleSearch({
        "q": f"site:quora.com {query}",
        "num": num_results,
        "gl": "us", "hl": "en",
        "api_key": os.getenv("SERPAPI_KEY")
    })
    organic = search.get_dict().get("organic_results", [])
    docs = []
    for item in organic:
        title   = item.get("title", "")
        snippet = item.get("snippet", "")
        link    = item.get("link", "")
        if snippet:
            docs.append({
                "content": f"Title: {title}\nAnswer snippet: {snippet}",
                "title": title,
                "link": link
            })
    return docs

def index_docs(docs, query_id):
    for i, doc in enumerate(docs):
        embedding = model.encode(doc["content"]).tolist()
        st.session_state.collection.add(
            documents=[doc["content"]],
            embeddings=[embedding],
            ids=[f"{query_id}_{i}"]
        )

def retrieve(query, top_k=5):
    count = st.session_state.collection.count()
    if count == 0:
        return []
    qe = model.encode(query).tolist()
    results = st.session_state.collection.query(
        query_embeddings=[qe],
        n_results=min(top_k, count)
    )
    return results["documents"][0]

def generate_answer(query, chunks):
    context = "\n\n---\n\n".join(chunks)
    prompt  = f"""You are a helpful assistant. Synthesize the Quora discussions below into a clear, detailed answer.
If different people have different opinions, mention that too.

QUORA CONTEXT:
{context}

QUESTION: {query}

Give a comprehensive, well-structured answer."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You synthesize Quora discussions into helpful answers."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.5,
        max_tokens=1024,
    )
    return response.choices[0].message.content

def run_rag(query):
    query_id = re.sub(r"\W+", "_", query[:20])
    docs     = fetch_quora_data(query)
    if not docs:
        return "No Quora results found. Try rephrasing your question.", []
    index_docs(docs, query_id)
    chunks = retrieve(query, top_k=5)
    answer = generate_answer(query, chunks)
    return answer, docs

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.markdown('<div class="hero-title">Quora AI Search</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Ask anything. Get answers synthesized from real Quora discussions.</div>', unsafe_allow_html=True)

query = st.text_input("", placeholder="e.g. How do I become an AI engineer in 2026?", label_visibility="collapsed")
ask   = st.button("Search Quora →")

if ask and query.strip():
    with st.spinner(""):
        # Show live steps
        steps = st.empty()
        steps.markdown("""
        <div class="step-row"><span class="step-badge">1</span> Searching Quora via SerpAPI...</div>
        """, unsafe_allow_html=True)

        answer, docs = run_rag(query.strip())

        steps.markdown("""
        <div class="step-row"><span class="step-badge">1</span> Fetched Quora results &nbsp;✓</div>
        <div class="step-row"><span class="step-badge">2</span> Indexed into ChromaDB &nbsp;✓</div>
        <div class="step-row"><span class="step-badge">3</span> Generated answer with Groq &nbsp;✓</div>
        """, unsafe_allow_html=True)

    # Answer card
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

    # Source chips
    if docs:
        st.markdown("<br>**Sources from Quora:**", unsafe_allow_html=True)
        chips = ""
        for doc in docs[:6]:
            title = doc["title"][:45] + "..." if len(doc["title"]) > 45 else doc["title"]
            chips += f'<span class="source-chip"><a href="{doc["link"]}" target="_blank">↗ {title}</a></span>'
        st.markdown(chips, unsafe_allow_html=True)

    # Save to history
    st.session_state.history.insert(0, {"q": query.strip(), "a": answer[:300] + "..."})

# History
if st.session_state.history:
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("**Recent searches**")
    for item in st.session_state.history[:5]:
        st.markdown(f"""
        <div class="history-item">
            <div class="history-q">Q: {item['q']}</div>
            <div class="history-a">{item['a']}</div>
        </div>
        """, unsafe_allow_html=True)