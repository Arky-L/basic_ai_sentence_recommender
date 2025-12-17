import csv
import random
from embedder import embed
from recommender import recommend

# Load CSV
products = []
with open("products.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        products.append(row)

# Build embeddings from descriptions
descriptions = [p["description"] for p in products]
embeddings = embed(descriptions)

# Pick a random product
query_index = random.randint(0, len(products) - 1)
query_product = products[query_index]

print(f"\nRecommendations for: {query_product['name']}\n")

results = recommend(query_index, embeddings, products)

for r in results:
    print(f"• {r['name']}  → similarity: {r['score']}")
