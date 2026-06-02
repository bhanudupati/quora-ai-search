import json
import chromadb

from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client
client = chromadb.Client()

# Create collection
collection = client.create_collection(name="quora_questions")

# Load data again
with open("data/quora_sample.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Store embeddings again
for item in data:

    text = f"""
    Question: {item['question']}
    Answer: {item['answer']}
    """

    embedding = model.encode(text).tolist()

    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[item["id"]]
    )

# USER QUESTION
query = "How can I learn machine learning?"

# Convert question to embedding
query_embedding = model.encode(query).tolist()

# Search similar documents
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nUser Query:")
print(query)

print("\nMost Relevant Results:\n")

for doc in results["documents"][0]:
    print(doc)
    print("-" * 50)