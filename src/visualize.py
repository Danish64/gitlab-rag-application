import numpy as np
import plotly.express as px
from umap import UMAP

from embeddings import load_vectorstore


def visualize_embeddings(limit=1000):
    """
    Visualize embedding space using UMAP dimensionality reduction.

    Args:
        limit: Max number of chunks to visualize (for performance)
    """
    print("Loading vectorstore...")
    vectorstore = load_vectorstore()
    collection = vectorstore._collection

    print(f"Total chunks in DB: {collection.count()}")

    # Get embeddings and metadata
    data = collection.get(
        limit=limit,
        include=["embeddings", "metadatas", "documents"]
    )

    embeddings = np.array(data["embeddings"])
    print(f"Visualizing {len(embeddings)} chunks...")

    # Extract category from file path (first folder after documents/)
    categories = []
    short_docs = []
    sources = []

    for i, meta in enumerate(data["metadatas"]):
        source = meta.get("source", "unknown")
        sources.append(source)

        # Get category from path: documents/people-group/... -> people-group
        parts = source.split("/")
        category = parts[1] if len(parts) > 1 else "other"
        categories.append(category)

        # Truncate doc for hover
        doc_text = data["documents"][i][:150] + "..." if len(data["documents"][i]) > 150 else data["documents"][i]
        short_docs.append(doc_text)

    # Reduce dimensions with UMAP
    print("Running UMAP dimensionality reduction...")
    umap = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine"
    )
    embeddings_2d = umap.fit_transform(embeddings)

    # Create dataframe for plotly
    import pandas as pd

    df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "category": categories,
        "source": sources,
        "preview": short_docs
    })

    # Create interactive plot
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="category",
        hover_data=["source", "preview"],
        title=f"GitLab RAG Embedding Space ({len(embeddings)} chunks)",
        labels={"category": "Category"},
        width=1200,
        height=800
    )

    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    # Save and show
    output_file = "embedding_visualization.html"
    fig.write_html(output_file)
    print(f"\nSaved to {output_file}")
    print("Opening in browser...")
    fig.show()

    return embeddings_2d, categories


def print_cluster_stats(categories):
    """Print stats about document categories."""
    from collections import Counter
    counts = Counter(categories)

    print("\n--- CATEGORY DISTRIBUTION ---")
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} chunks")


if __name__ == "__main__":
    embeddings_2d, categories = visualize_embeddings(limit=2000)
    print_cluster_stats(categories)
