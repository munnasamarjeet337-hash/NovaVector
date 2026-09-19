# NovaVector

A lightweight, in-memory vector database and local RAG pipeline written from scratch in C++17. 

I built this to understand how vector similarity engines like Pinecone and Chroma handle indexing under the hood, specifically comparing spatial trees against multi-layer proximity graphs when dealing with the curse of dimensionality.

## What it Does

- **3 Search Algorithms:** Flat (Brute Force), KD-Tree, and HNSW (Hierarchical Navigable Small World).
- **Distance Metrics:** Cosine Similarity, Euclidean ($L_2$), and Manhattan ($L_1$).
- **Local RAG Pipeline:** Interfaces directly with local Ollama (`nomic-embed-text` for 768D embeddings and `llama3.2` for context injection). Zero external API calls.
- **Microsecond Benchmarks:** Live side-by-side latency comparisons across all indices.
- **PCA Visualizer:** Web interface mapping high-dimensional clusters onto a 2D plane via power-iteration PCA.

## Why HNSW?

In low dimensions ($\le 20D$), KD-Trees partition space cleanly and prune subtrees quickly. But at 768 dimensions (standard transformer output), almost every point sits near the boundary of the bounding box. Pruning breaks down, and KD-Tree search times collapse into brute force. 

HNSW fixes this by organizing points into a skip-list-style graph. Searches hop across coarse upper layers to find the neighborhood fast, then zoom in at layer 0—keeping lookup times at $O(\log N)$ even in high-dimensional spaces.

## Project Structure

```text
NovaVector/
├── include/
│   ├── types.hpp          # Vector & document struct definitions
│   ├── metrics.hpp        # Distance formulas (Cosine, L2, L1)
│   ├── brute_force.hpp    # Exact O(N*d) scan
│   ├── kd_tree.hpp        # Spatial partitioning tree
│   └── hnsw.hpp           # Multi-layer graph index
├── src/
│   └── main.cpp           # REST API routes, RAG orchestrator, and server
├── httplib.h              # Header-only HTTP server
├── index.html             # Visualization UI & benchmark dashboard
└── README.md
