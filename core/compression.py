def compress_context(chunks):
    # Clean + structured context
    cleaned_chunks = []

    for i, chunk in enumerate(chunks):
        chunk = chunk.replace("\n", " ").strip()
        cleaned_chunks.append(f"[Chunk {i+1}]: {chunk}")

    return "\n\n".join(cleaned_chunks)