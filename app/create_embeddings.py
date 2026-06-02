import json
import chromadb

from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load JSON data
with open("data/quora_sample.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Create ChromaDB client
client = chromadb.Client()

# Create collection
collection = client.create_collection(name="quora_questions")

# Process each question-answer pair
for item in data:

    text = f"""
    Question: {item['question']}
    Answer: {item['answer']}
    """

    # Generate embedding
    embedding = model.encode(text).tolist()

    # Store in ChromaDB
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[item["id"]]
    )

print("Embeddings stored successfully!")