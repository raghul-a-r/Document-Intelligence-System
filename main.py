from core.ingestion import extract_text_from_pdf, chunk_text, extract_abstract
from core.embedding import embed_texts
from core.vectorstore import VectorStore
from core.rag import retrieve_and_answer
from core.analytics import cluster_topics, extract_keywords
from core.graph import build_graph, visualize_graph


# ---------------------------
# 1. LOAD DOCUMENT
# ---------------------------
text = extract_text_from_pdf("data/sample.pdf")

# Extract abstract
abstract = extract_abstract(text)

# ---------------------------
# 2. CHUNK + EMBED
# ---------------------------
all_chunks = chunk_text(text)

embeddings = embed_texts(all_chunks)

# ---------------------------
# 3. VECTOR STORE
# ---------------------------
vs = VectorStore(len(embeddings[0]))
vs.add(embeddings, all_chunks)

# ---------------------------
# 4. QUERY (RAG)
# ---------------------------
query = "What is the main idea of the paper?"

answer, retrieved_chunks = retrieve_and_answer(
    query,
    vs,
    abstract=abstract
)

print("\n--- RETRIEVED CHUNKS ---")
for c in retrieved_chunks:
    print("\n", c[:200])

print("\n--- ANSWER ---")
print(answer)


# ---------------------------
# 5. ANALYTICS
# ---------------------------
print("\n--- ANALYTICS ---")

# 🔥 IMPORTANT: use ALL chunks (not retrieved ones)
clusters = cluster_topics(embeddings, all_chunks)

for label, texts in clusters.items():
    print(f"\nCluster {label}:")
    print(texts[0][:200])  # preview

keywords = extract_keywords(all_chunks)

print("\nTop Keywords:")
for word, count in keywords:
    print(word, count)

    
print("\n--- KNOWLEDGE GRAPH ---")

graph = build_graph(all_chunks[:50])  # limit for speed

visualize_graph(graph)