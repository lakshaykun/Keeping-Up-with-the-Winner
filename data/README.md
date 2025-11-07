# Data Guide

Place your graphs in this folder if you prefer filesystem loading (the Streamlit app also supports uploads).

## Formats supported
- **Edgelist**: whitespace-separated `u v` pairs per line.
- **Adjacency CSV**: square CSV of 0/1 entries.

## SNAP Datasets
To reproduce the exact paper experiments, download these datasets and convert to edgelists if needed:
- Social circles — Facebook (4039 nodes)
- Facebook pages of public figures (~11,565 nodes)

Then load them via the Streamlit UI or adapt `experiments/reproduce_fig4_7.py`.
