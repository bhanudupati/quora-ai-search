import os
import re
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq
from serpapi import GoogleSearch

# ─────────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────────

load_dotenv(dotenv_path=".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_KEY  = os.getenv("SERPAPI_KEY")

print("Loading embedding model...")
model       = SentenceTransformer("all-MiniLM-L6-v2")
client      = chromadb.Client()
collection  = client.create_collection(name="quora_questions")
groq_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
# 2. FETCH RICH QUORA DATA VIA SERPAPI
# ─────────────────────────────────────────────

def fetch_quora_data(query: str, num_results: int = 10) -> list[dict]:
    """
    Search Google for site:quora.com and extract ALL available
    text from each result: title, snippet, and rich_snippet.
    """
    print(f"\n🌐 Searching Quora for: '{query}'...")

    search = GoogleSearch({
        "q": f"site:quora.com {query}",
        "num": num_results,
        "gl": "us",       # US results = more English content
        "hl": "en",       # English language
        "api_key": SERPAPI_KEY
    })

    results = search.get_dict()
    organic = results.get("organic_results", [])

    docs = []
    for item in organic:
        title   = item.get("title", "")
        snippet = item.get("snippet", "")
        link    = item.get("link", "")

        # SerpAPI sometimes returns extra answer data here
        rich    = item.get("rich_snippet", {})
        rich_text = ""
        if rich:
            rich_text = str(rich)

        # Combine all available text into one rich document
        full_content = f"Title: {title}\n"
        if snippet:
            full_content += f"Answer snippet: {snippet}\n"
        if rich_text:
            full_content += f"Additional info: {rich_text}\n"
        full_content += f"Source: {link}"

        if snippet:  # Only add if there's actual content
            docs.append({
                "content": full_content,
                "title": title,
                "link": link
            })

    print(f"✅ Found {len(docs)} Quora results.")
    return docs


# ─────────────────────────────────────────────
# 3. INDEX INTO CHROMADB
# ─────────────────────────────────────────────

def index_results(docs: list[dict], query_id: str):
    for i, doc in enumerate(docs):
        embedding = model.encode(doc["content"]).tolist()
        collection.add(
            documents=[doc["content"]],
            embeddings=[embedding],
            ids=[f"{query_id}_{i}"]
        )
        print(f"  📄 [{i+1}] {doc['title'][:60]}...")
    print(f"📦 Indexed {len(docs)} documents.\n")


# ─────────────────────────────────────────────
# 4. RETRIEVE RELEVANT CHUNKS
# ─────────────────────────────────────────────

def retrieve(query: str, top_k: int = 5) -> list[str]:
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )
    return results["documents"][0]


# ─────────────────────────────────────────────
# 5. BUILD PROMPT & GENERATE ANSWER
# ─────────────────────────────────────────────

def build_prompt(query: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""You are a helpful assistant. Below are real answers and discussions from Quora about the user's question.
Synthesize the information from multiple answers to give a comprehensive, well-rounded response.
If different Quora users have different opinions, mention that too.

QUORA CONTEXT:
{context}

USER QUESTION:
{query}

Give a detailed, helpful answer by combining insights from the Quora discussions above."""


def generate_answer(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that synthesizes Quora discussions "
                    "into clear, detailed answers. Always extract maximum insight from "
                    "the context provided, even if snippets are short."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,   # Slightly higher for more natural synthesis
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# 6. FULL RAG PIPELINE
# ─────────────────────────────────────────────

def rag_agent(query: str) -> str:
    # Step 1: Fetch from Quora via SerpAPI
    docs = fetch_quora_data(query, num_results=10)

    if not docs:
        return "Sorry, couldn't find any Quora results for your question."

    # Step 2: Index into ChromaDB
    query_id = re.sub(r"\W+", "_", query[:20])
    index_results(docs, query_id)

    # Step 3: Retrieve top relevant chunks
    chunks = retrieve(query, top_k=5)
    print(f"📚 Retrieved {len(chunks)} most relevant chunk(s).")

    # Step 4: Generate answer
    prompt = build_prompt(query, chunks)
    print("🤖 Generating answer with Groq...\n")
    return generate_answer(prompt)


# ─────────────────────────────────────────────
# 7. INTERACTIVE LOOP
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Quora RAG Agent  |  SerpAPI + ChromaDB + Groq")
    print("=" * 60)
    print("Ask anything! I'll fetch real Quora answers for you.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Your question: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = rag_agent(user_input)

        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60 + "\n")