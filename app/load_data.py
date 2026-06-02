import json

with open("data/quora_sample.json", "r", encoding="utf-8") as file:
    data = json.load(file)

print(f"Loaded {len(data)} records\n")

for item in data:
    print("Question:", item["question"])
    print("Answer:", item["answer"])
    print("-" * 50)