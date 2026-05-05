from sklearn.cluster import KMeans
from collections import Counter
import numpy as np


def cluster_topics(embeddings, chunks, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}

    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(chunks[i])

    return clusters


def extract_keywords(chunks, top_n=10):
    words = []

    for chunk in chunks:
        words.extend(chunk.lower().split())

    # remove small words
    words = [w for w in words if len(w) > 4]

    freq = Counter(words)

    return freq.most_common(top_n)