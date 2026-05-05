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
        if ent.label_ in ["ORG", "PRODUCT", "GPE", "PERSON"]:
            entities.append(ent.text)

    return list(set(entities))


def build_graph(chunks):
    G = nx.Graph()

    for chunk in chunks:
        entities = extract_entities(chunk)

        # connect entities in same chunk
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                G.add_edge(entities[i], entities[j])

    return G


def visualize_graph(G):
    plt.figure(figsize=(10, 8))

    # reduce clutter (optional)
    nodes = list(G.nodes)[:15]
    G = G.subgraph(nodes)

    pos = nx.spring_layout(G, k=1.0)    

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=8
    )

    plt.title("Knowledge Graph")

    plt.savefig("graph.png")
    print("\nGraph saved as graph.png")