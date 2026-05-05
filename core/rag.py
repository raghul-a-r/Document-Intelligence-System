from core.embedding import embed_texts
from core.compression import compress_context
from core.llm import generate_answer
from core.embedding import embed_texts



def retrieve(query, vectorstore, k=8):
    query_emb = embed_texts([query])[0]
    results = vectorstore.search(query_emb, k)

    priority_chunks = []
    normal_chunks = []

    for chunk in results:
        chunk_lower = chunk.lower()

        # 🔥 PRIORITY: Abstract / Introduction
        if any(word in chunk_lower for word in [
            "abstract", "introduction", "in this paper", "we propose"
        ]):
            priority_chunks.append(chunk)
        else:
            normal_chunks.append(chunk)

    # If we found good chunks → use them
    if priority_chunks:
        selected = priority_chunks
    else:
        selected = normal_chunks

    # Sort by length (longer = more informative)
    selected = sorted(selected, key=lambda x: -len(x))

    return selected[:5]



def retrieve_and_answer(query, vectorstore, k=8, abstract=None):
    # 🔥 PRIORITY: use abstract if available
    if abstract:
        context = abstract
        chunks = [abstract]
    else:
        chunks = retrieve(query, vectorstore, k)
        context = compress_context(chunks)

    answer = generate_answer(query, context)

    return answer, chunks