# Idea Bubbles — Roadmap

## Semantic Cluster Labels (priority)

The current cluster labeling extracts shared keywords or picks the most central idea's text. This produces labels like "coffee" for a group containing "coffee" and "tea", when the ideal label would be something like "hot beverages" or "caffeinated drinks".

Generating truly semantic labels requires an LLM that can abstract over the group members. Options explored:

1. **External LLM API (recommended)** — A lightweight API call per cluster (e.g., Claude, OpenAI). Most reliable for producing abstract labels. Would require an API key field in the UI, stored in localStorage. Minimal token usage since we're just sending a handful of short idea texts per cluster and asking for a 2-3 word label.

2. **Local small LLM via Transformers.js** — Models like Phi-3-mini can run in-browser but are 1-2GB+ downloads. Too heavy for a casual web app.

3. **Embedding-based label matching** — Precompute embeddings for a vocabulary of abstract category terms and find the nearest match to each cluster's centroid. Limited by the predefined vocabulary — can't generate novel labels.

**Decision:** Deferred. Revisit when adding other API-dependent features to amortize the cost of requiring an API key.

## Other Future Directions

- **Collaborative / multi-user** — Would require a backend (e.g., FastAPI + WebSockets)
- **Export to other formats** — JSON, Markdown outline grouped by cluster
- **Import ideas** — Paste a list, upload a text file
- **Undo/redo** — Track idea additions and deletions
- **Keyboard navigation** — Arrow keys to move between bubbles, delete key to remove
