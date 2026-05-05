import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import os
from core.ingestion import extract_text_from_pdf, chunk_text, extract_abstract
from core.embedding import embed_texts
from core.vectorstore import VectorStore
from core.rag import retrieve_and_answer
from core.analytics import cluster_topics, extract_keywords
from core.graph import build_graph, visualize_graph

st.set_page_config(page_title="Document Intelligence System", layout="wide")

st.title("📄 Document Intelligence System")
st.sidebar.title("About")
st.sidebar.write("AI-powered Document Intelligence System")

# ---------------------------
# Upload PDF
# ---------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save file
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # ---------------------------
    # Process document
    # ---------------------------
    text = extract_text_from_pdf(file_path)
    abstract = extract_abstract(text)

    all_chunks = chunk_text(text)
    embeddings = embed_texts(all_chunks)

    vs = VectorStore(len(embeddings[0]))
    vs.add(embeddings, all_chunks)

    # ---------------------------
    # Tabs
    # ---------------------------
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "🔗 Graph"])

    # ===========================
    # CHAT TAB
    # ===========================
    with tab1:
        st.subheader("Ask Questions")

        query = st.text_input("Enter your question:")

        if st.button("Ask"):
            if query:
                answer, retrieved_chunks = retrieve_and_answer(
                    query, vs, abstract=abstract
                )

                st.write("### Answer:")
                st.write(answer)

                with st.expander("Retrieved Context"):
                    for c in retrieved_chunks:
                        st.write(c[:300])

    # ===========================
    # ANALYTICS TAB
    # ===========================
    with tab2:
        st.subheader("Document Analytics")

        if st.button("Generate Analytics"):
            clusters = cluster_topics(embeddings, all_chunks)

            st.write("### Topic Clusters")
            for label, texts in clusters.items():
                st.write(f"**Cluster {label}:**")
                st.write(texts[0][:200])

            keywords = extract_keywords(all_chunks)

            st.write("### Top Keywords")
            for word, count in keywords:
                st.write(f"{word} ({count})")

    # ===========================
    # GRAPH TAB
    # ===========================
    with tab3:
        st.subheader("Knowledge Graph")

        if st.button("Generate Graph"):
            graph = build_graph(all_chunks[:50])

            visualize_graph(graph)

            if os.path.exists("graph.png"):
                st.image("graph.png")
