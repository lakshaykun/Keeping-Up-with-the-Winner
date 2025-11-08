import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from utils import load_graph_from_edgelist, load_graph_from_adj_csv, graph_to_adjacency, ensure_symmetric_binary
from critical_params import compute_critical_mu_u
from algorithm1 import algorithm1_local_search
from bisis import simulate_fixed_point
from plots import plot_market_share_vs_eps, fig_to_png_bytes
from baselines import baseline_degree, baseline_eigenvector, baseline_netshield

st.set_page_config(page_title="bi-SIS Community Targeting", layout="wide")
st.title("Keeping Up with the Winner — Survival Threshold Dashboard")

st.markdown(
    """
        **Workflow**
        1. **Upload/Select Graph** →
        2. **Configure Parameters** →
        3. **Compute Critical (μ_c, u_c)** →
        4. **Run Algorithm 1 (ε-sweep)** →
        5. **Compare Baselines** →
        6. **Export Results**
    """
)

# -----------------------
# Sidebar: Data Loading
# -----------------------
with st.sidebar:
    st.header("1) Graph Input")
    src_choice = st.radio("Choose graph source", ["Sample", "Upload (edgelist)", "Upload (adjacency CSV)"])

    if src_choice == "Sample":
        sample = st.selectbox("Sample graph", ["Zachary Karate (34)", "Erdos-Renyi n=100, p=0.04"])
    else:
        uploaded = st.file_uploader("Upload file", type=["edgelist", "csv", "txt"])    

    st.header("2) Model Parameters")
    tau1 = st.number_input("tau1 (Product 1)", min_value=0.001, max_value=2.0, value=0.8, step=0.01)
    tau2 = st.number_input("tau2 (Product 2)", min_value=0.001, max_value=2.0, value=0.05, step=0.01)

    st.header("3) Sweep Configuration")
    delta_C = st.number_input("Budget increment ΔC", min_value=1e-5, max_value=10.0, value=0.1, step=0.01)
    num_steps = st.slider("Number of budget increments", min_value=5, max_value=30, value=15)
    eps_min = st.number_input("epsilon min", value=1e-8, format="%.1e")
    eps_max = st.number_input("epsilon max", value=1e-1, format="%.1e")
    num_eps = st.slider("# epsilon points (log-spaced)", min_value=5, max_value=30, value=15)
    iters = st.slider("Local-search iterations", min_value=1, max_value=10, value=1)

    st.header("4) Cost Scheme")
    cost_scheme = st.selectbox("Costs w_i", ["Homogeneous (all 1)", "Degree", "Eigenvector"])

    st.header("5) Baselines")
    enable_deg = st.checkbox("Compare Degree baseline", value=True)
    enable_evc = st.checkbox("Compare Eigenvector baseline", value=True)
    enable_ns = st.checkbox("Compare NetShield baseline", value=False)
    ns_k = st.number_input("NetShield k (if enabled)", min_value=1, value=50)

# -----------------------
# Main: Load Graph
# -----------------------
@st.cache_data(show_spinner=False)
def build_graph(src_choice, sample, uploaded_bytes):
    import networkx as nx
    if src_choice == "Sample":
        if sample.startswith("Zachary"):
            G = nx.karate_club_graph()
        else:
            G = nx.erdos_renyi_graph(100, 0.04, seed=7)
        if not nx.is_connected(G):
            cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(cc).copy()
        return G
    else:
        if uploaded_bytes is None:
            return None
        name = uploaded_bytes.name.lower()
        data = uploaded_bytes.read()
        if name.endswith(".csv"):
            return load_graph_from_adj_csv(data)
        else:
            return load_graph_from_edgelist(data)

G = build_graph(src_choice, sample if src_choice=="Sample" else None, uploaded if src_choice!="Sample" else None)

if G is None:
    st.info("Upload a graph to proceed.")
    st.stop()

A = graph_to_adjacency(G)
A = ensure_symmetric_binary(A)
N = A.shape[0]

st.success(f"Graph loaded with N={N} nodes, E={int(A.sum()/2)} edges.")

# -----------------------
# Costs
# -----------------------
import networkx as nx
if cost_scheme == "Homogeneous (all 1)":
    costs = np.ones(N)
elif cost_scheme == "Degree":
    Gnx = nx.from_numpy_array(A)
    deg = np.array([d for _, d in Gnx.degree()], dtype=float)
    costs = np.maximum(deg, 1.0)
else:
    Gnx = nx.from_numpy_array(A)
    ev = nx.eigenvector_centrality_numpy(Gnx)
    v = np.array([ev[i] for i in range(N)], dtype=float)
    costs = np.maximum(v, 1e-3)

# -----------------------
# Critical (mu_c, u_c)
# -----------------------
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Compute critical (μ_c, u_c)"):
        with st.spinner("Computing x*, Perron vector, and critical pair..."):
            mu_c, u_c, x_star = compute_critical_mu_u(A, tau1, tau2)
        st.session_state["mu_c"] = mu_c
        st.session_state["u_c"] = u_c
        st.session_state["x_star"] = x_star
        st.success(f"μ_c = {mu_c:.6g}")
        st.write("u_c stats:", pd.Series(u_c).describe())

with col2:
    if "mu_c" in st.session_state:
        st.metric("μ_c (critical)", f"{st.session_state['mu_c']:.6g}")
        st.caption("Derived via Lemma 3.2: u_c ∝ Sx^{-1} v, μ_c = (1/τ₂ - λ(SxA)) / (u_cᵀ Sx u_c)")

if "mu_c" not in st.session_state:
    st.stop()

mu_c = st.session_state["mu_c"]
u_c = st.session_state["u_c"]
x_star = st.session_state["x_star"]

# -----------------------
# Epsilon sweep
# -----------------------
log_eps = np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)

if st.button("Run Auto Budget Sweep"):
    w = costs
    C_min = np.sqrt(mu_c) * np.dot(w, u_c)
    st.metric("Critical Budget C_min", f"{C_min:.4f}")

    budgets = [C_min + k * delta_C for k in range(num_steps)]

    rows = []
    for C in budgets:
        scale0 = C / (np.sqrt(mu_c) * np.dot(costs, u_c))
        u_scaled = np.clip(u_c * scale0, 0, 1)

        u_opt = algorithm1_local_search(
            A, tau1, tau2, mu_c, u_scaled, x_star, eps_min,
            iters=iters, costs=costs, budget_C=C
        )


        x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_opt)

        # More robust effective community definition
        eff_size = float(np.sum(u_opt) * len(u_opt))

        rows.append({
            "Budget": float(C),
            "AvgX": float(x.mean()),
            "AvgY": float(y.mean()),
            "EffSize": eff_size
        })

    df = pd.DataFrame(rows)
    st.session_state["budget_results_df"] = df

    st.subheader("Results: Survival Threshold Experiment")
    st.dataframe(df)

    # --- Plot AvgY vs Budget ---
    fig1 = plt.figure()
    plt.plot(df["Budget"], df["AvgY"], marker="o")
    plt.axvline(x=C_min, color='r', linestyle='--', label='C_min (Critical)')
    plt.xlabel("Budget C")
    plt.ylabel("AvgY (Product 2)")
    plt.title("Product 2 Survival vs Budget")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    st.pyplot(fig1)

    # --- Plot Effective Community Size ---
    fig2 = plt.figure()
    plt.plot(df["Budget"], df["EffSize"], marker="s", color="purple")
    plt.axvline(x=C_min, color='r', linestyle='--', label='C_min (Critical)')
    plt.xlabel("Budget C")
    plt.ylabel("Expected Community Size (Σu_i)")
    plt.title("Community Expansion vs Budget")
    plt.legend()
    plt.grid(True, which="both", ls=":")
    st.pyplot(fig2)

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Sweep Results CSV", csv, file_name="budget_sweep_results.csv")

if st.button("Run Algorithm 1 (ε-sweep)"):
    rows = []
    figs = []

    # Our approach
    avgX_list, avgY_list = [], []
    for eps in log_eps:
        u = algorithm1_local_search(A, tau1, tau2, mu_c, u_c, x_star, eps, iters=iters, costs=costs)
        x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u)
        avgX_list.append(x.mean())
        avgY_list.append(y.mean())
        rows.append({"method":"Our Approach","epsilon":float(eps),"AvgX":float(x.mean()),"AvgY":float(y.mean())})

    # Baselines
    if enable_deg:
        u_deg = baseline_degree(A)
        x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_deg)
        for eps in log_eps:
            rows.append({"method":"Degree","epsilon":float(eps),"AvgX":float(x.mean()),"AvgY":float(y.mean())})

    if enable_evc:
        u_evc = baseline_eigenvector(A)
        x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_evc)
        for eps in log_eps:
            rows.append({"method":"EVC","epsilon":float(eps),"AvgX":float(x.mean()),"AvgY":float(y.mean())})

    if enable_ns:
        u_ns = baseline_netshield(A, k=int(ns_k))
        # normalize to [0,1] prob-style (so scale like u)
        if u_ns.sum() > 0:
            u_ns = u_ns / u_ns.sum()
        x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_ns)
        for eps in log_eps:
            rows.append({"method":"NetShield","epsilon":float(eps),"AvgX":float(x.mean()),"AvgY":float(y.mean())})

    df = pd.DataFrame(rows)
    st.session_state["results_df"] = df

    st.subheader("Results Table")
    st.dataframe(df)

    st.subheader("Plots (Our Approach)")
    res_ours = list(zip(log_eps, avgX_list, avgY_list))
    f1, f2 = plot_market_share_vs_eps(res_ours, title="Our Approach")
    st.pyplot(f1)
    st.pyplot(f2)

    png1 = fig_to_png_bytes(f1)
    png2 = fig_to_png_bytes(f2)
    st.download_button("Download Plot (Product 2)", png1, file_name="plot_product2.png")
    st.download_button("Download Plot (Product 1)", png2, file_name="plot_product1.png")

# -----------------------
# Compare methods: AvgY vs epsilon
# -----------------------
if "results_df" in st.session_state:
    st.subheader("Comparison: AvgY vs epsilon (all methods)")
    df = st.session_state["results_df"]
    methods = df["method"].unique()

    fig = plt.figure()
    for m in methods:
        sub = df[df["method"] == m]
        sub = sub.sort_values("epsilon")
        plt.xscale("log")
        plt.plot(sub["epsilon"], sub["AvgY"], marker="o", label=m)
    plt.xlabel("epsilon")
    plt.ylabel("AvgY (Product 2)")
    plt.title("AvgY vs epsilon")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    st.pyplot(fig)

    png = fig_to_png_bytes(fig)
    st.download_button("Download Comparison Plot (AvgY)", png, file_name="comparison_AvgY.png")

    st.subheader("Export CSV")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results CSV", csv, file_name="results.csv")
