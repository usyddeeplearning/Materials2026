# Week 6 Tutorial Outline — Graph Neural Networks

## Overview
Single notebook (`Week10_Graph_Convolutional_Networks.ipynb`), following the
same pattern as previous weeks: custom PyTorch layer for the forward pass,
autograd for backward.

Dataset: **Zachary's Karate Club** (34-node social network, classic GCN demo).

---

## §0 — Why Graphs?
- Euclidean data (images, sequences) vs non-Euclidean / relational data
- Real-world graph examples: social networks, molecules, knowledge graphs
- Why CNNs fail on graphs: variable neighbourhood size, no spatial ordering
  (lecture P2–P5)

## §1 — Graph Fundamentals
1. **Nodes & Edges** — graph definition, edge list
2. **Adjacency matrix A** — binary, symmetric for undirected graphs
3. **Self-loops**: Â = A + I — why they matter for GCN
4. **Degree matrix D** — diagonal matrix counting connections per node
5. **Weighted & directed graphs** — brief mention
6. **Visualise with NetworkX**: load Karate Club, draw graph, print A and D
   (lecture P6–P17)

## §2 — GCN: Spatial Perspective
- Intuition: a node's new representation aggregates its own features + neighbours'
- GCN propagation rule:
  `H^(l+1) = σ( D̂^{-1/2} Â D̂^{-1/2}  H^(l)  W^(l) )`
- Why symmetric normalisation: prevent large-degree nodes from dominating
- Hand-traced example on a small 4-node graph (just matrix arithmetic in markdown)
  (lecture P18–P26)

## §3 — GCNLayer (Custom PyTorch Implementation)
- Helper `build_A_hat_norm(A)`: add self-loops → compute D̂^{-1/2} Â D̂^{-1/2}
- `GCNLayer(nn.Module)`:
  - weight matrix W stored as `nn.Parameter`
  - `forward(H, A_hat_norm)` computes `σ(A_hat_norm @ H @ W)` by hand
  - backward via autograd (same pattern as LinearLayer / ConvLayer)
- Single-layer sanity check on Karate Club: inspect output shape and values

## §4 — Node Classification: Karate Club
- Semi-supervised setup: 34 nodes, 2 labeled (one per community)
- 2-layer GCN model with the custom `GCNLayer`
- Training loop: cross-entropy on labeled nodes, full-graph forward pass
- Evaluation: accuracy on all 34 nodes
- Visualise: scatter plot of 2-D hidden-layer embeddings (colour = community)
  (lecture P27 — node classification)

## §5 — Other Prediction Tasks (Brief Demo)
- **Graph classification**: mean-pool node embeddings → MLP classifier;
  demo on a small synthetic dataset (two classes of random graphs)
- **Link prediction**: dot product between node embeddings as edge score;
  demo on held-out edges from Karate Club
  (lecture P27 — graph & link prediction)

---

## Notes
- Spectral GCN / Laplacian eigen-decomposition: **skipped**
  (covered in lecture P28–P39 for reference; too advanced for tutorial).
- NetworkX is used only for graph construction and visualisation;
  all trainable computations use PyTorch tensors.
- Reuse the `nn.Module` / `nn.Parameter` / autograd pattern from
  ConvLayer and LinearLayer in Week 5.
