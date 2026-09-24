from rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag.retrieval.similarity import consine_similarity

embedder = SentenceTransformerEmbedder()

text_a = "Employees receive 21 days of annual leave."

text_b = "Workers are entitled to twenty-one vacation days each year."

text_c = "The company provides health insurance."


embedding_a = embedder.embed_text(text_a)
embedding_b = embedder.embed_text(text_b)
embedding_c = embedder.embed_text(text_c)

similarity_ab = consine_similarity(
    embedding_a,
    embedding_b
)

similarity_ac = consine_similarity(
    embedding_a,
    embedding_c
)

print(f"A: {text_a}")
print(f"B: {text_b}")
print()

print("Annual leave vs vacation days: ", similarity_ab)
print("Annual leave vs helth insurance: ", similarity_ac)



# texts = [
#     "Employees receive 21 days of annual leave.",
#     "Workers are entitled to twenty-one vacation days each year.",
#     "The company provides health insurance.",
# ]

# for text in texts:
#     embedding = embedder.embed_text(text)

#     print("=" * 80)
#     print(text)
#     print(f"Dimensions: {len(embedding)}")
#     print(f"First 10 values: {embedding[:10]}")