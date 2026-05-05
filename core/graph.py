import spacy
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib


# matplotlib.use("TkAgg")
nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "PERSON"]:
            entities.append(ent.text)

    # limit entities per chunk
    return list(set(entities))[:5]


def build_graph(chunks):
    G = nx.Graph()

    edge_weights = {}

    for chunk in chunks:
        entities = extract_entities(chunk)

        # only connect meaningful pairs
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                pair = tuple(sorted((entities[i], entities[j])))

                edge_weights[pair] = edge_weights.get(pair, 0) + 1

    # only keep strong relationships
    for (e1, e2), weight in edge_weights.items():
        if weight >= 2:  # threshold
            G.add_edge(e1, e2, weight=weight)

    return G


def visualize_graph(G):
    plt.figure(figsize=(10, 8))

    # keep top nodes only
    nodes = list(G.nodes)[:12]
    G = G.subgraph(nodes)

    pos = nx.spring_layout(G, k=1.2)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2500,
        font_size=9,
        edge_color="gray",
    )

    plt.title("Knowledge Graph (Filtered)")

    plt.savefig("graph.png")
    print("\nGraph saved as graph.png")
