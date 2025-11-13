import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from datetime import datetime

from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from utils import load_graph_from_edgelist, load_graph_from_adj_csv, graph_to_adjacency, ensure_symmetric_binary
from critical_params import compute_critical_mu_u
from algorithm1 import algorithm1_local_search
from bisis import simulate_fixed_point
from plots import plot_market_share_vs_eps, fig_to_png_bytes
from baselines import baseline_degree, baseline_eigenvector, baseline_netshield
from pdf_report import create_pdf_report

st.set_page_config(page_title="bi-SIS Community Targeting", layout="wide")

# Header with description
st.title("🎯 Keeping Up with the Winner")
st.markdown("### Community Targeting Dashboard for Competing Product Diffusion")

st.info("""
**What does this do?** This dashboard helps you find the optimal community targeting strategy 
for a new product competing against an established dominant product in a social network.
""")

# Workflow overview
with st.expander("📋 How to Use This Dashboard", expanded=False):
    st.markdown("""
    **Step-by-Step Workflow:**
    
    1. **📊 Load Your Network** - Choose a sample graph or upload your own network data
    2. **⚙️ Set Model Parameters** - Configure spreading rates (tau1, tau2) for both products
    3. **🎯 Compute Critical Point** - Calculate the critical intervention parameters (μ_c, u_c)
    4. **🚀 Run Analysis** - Choose between:
       - **Budget Sweep**: See how market share changes with different budgets
       - **Epsilon Sweep**: Optimize targeting strategy across different precision levels
    5. **📈 Compare Results** - Evaluate against baseline strategies (Degree, Eigenvector, NetShield)
    6. **💾 Export Data** - Download results as CSV and plots as PNG
    
    **Quick Start:** Use the default settings and click "Compute Critical Parameters" to begin!
    """)

# -----------------------
# Sidebar: Configuration
# -----------------------
with st.sidebar:
    st.header("📊 1. Graph Input")
    st.markdown("Choose your network data source:")
    
    src_choice = st.radio(
        "Data Source",
        ["Sample", "Upload (edgelist)", "Upload (adjacency CSV)"],
        help="Sample graphs are pre-loaded. Upload your own for custom analysis."
    )

    if src_choice == "Sample":
        sample = st.selectbox(
            "Select Sample Graph",
            ["Zachary Karate (34)", "Erdos-Renyi n=100, p=0.04"],
            help="Karate Club: Classic social network. Erdos-Renyi: Random graph for testing."
        )
    else:
        uploaded = st.file_uploader(
            "Upload file",
            type=["edgelist", "csv", "txt"],
            help="Edgelist: node pairs per line. CSV: adjacency matrix."
        )    

    st.divider()
    st.header("⚙️ 2. Model Parameters")
    st.markdown("**Product Spreading Rates:**")
    
    tau1 = st.number_input(
        "τ₁ (Product 1 - Dominant)",
        min_value=0.001,
        max_value=2.0,
        value=0.8,
        step=0.01,
        help="Higher values = Product 1 spreads faster (typically 0.5-1.0)"
    )
    tau2 = st.number_input(
        "τ₂ (Product 2 - New/Weaker)",
        min_value=0.001,
        max_value=2.0,
        value=0.05,
        step=0.01,
        help="Lower values = Product 2 needs stronger intervention (typically 0.01-0.1)"
    )

    st.divider()
    st.header("💰 3. Cost & Budget")
    
    cost_scheme = st.selectbox(
        "Node Cost Function",
        ["Homogeneous (all 1)", "Degree", "Eigenvector"],
        help="Homogeneous: equal cost. Degree: cost ∝ connections. Eigenvector: cost ∝ influence."
    )
    
    st.markdown("**Budget Sweep Settings:**")
    delta_C = st.number_input(
        "Budget Increment (ΔC)",
        min_value=1e-5,
        max_value=10.0,
        value=0.1,
        step=0.01,
        help="Step size for budget analysis"
    )
    num_steps = st.slider(
        "Number of Budget Steps",
        min_value=5,
        max_value=30,
        value=15,
        help="More steps = finer analysis but slower"
    )

    st.divider()
    st.header("🔬 4. Algorithm Settings")
    st.markdown("**Epsilon Sweep Configuration:**")
    
    col_eps1, col_eps2 = st.columns(2)
    with col_eps1:
        eps_min = st.number_input(
            "ε min",
            value=1e-8,
            format="%.1e",
            help="Smallest perturbation"
        )
    with col_eps2:
        eps_max = st.number_input(
            "ε max",
            value=1e-1,
            format="%.1e",
            help="Largest perturbation"
        )
    
    num_eps = st.slider(
        "Epsilon Points (log-spaced)",
        min_value=5,
        max_value=30,
        value=15,
        help="Number of epsilon values to test"
    )
    iters = st.slider(
        "Local-Search Iterations",
        min_value=1,
        max_value=10,
        value=1,
        help="More iterations = better optimization but slower"
    )

    st.divider()
    st.header("📊 5. Baseline Comparisons")
    st.markdown("Compare against other targeting strategies:")
    
    enable_deg = st.checkbox("Degree Centrality", value=True, help="Target high-degree nodes")
    enable_evc = st.checkbox("Eigenvector Centrality", value=True, help="Target influential nodes")
    enable_ns = st.checkbox("NetShield", value=False, help="Network shield approach")
    
    if enable_ns:
        ns_k = st.number_input(
            "NetShield k (nodes)",
            min_value=1,
            value=50,
            help="Number of nodes to select"
        )

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

# Display graph statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📍 Nodes", N)
with col2:
    st.metric("🔗 Edges", int(A.sum()/2))
with col3:
    avg_degree = A.sum(axis=1).mean()
    st.metric("📊 Avg Degree", f"{avg_degree:.2f}")

st.success("✅ Graph loaded successfully!")

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
st.divider()
st.header("🎯 Step 1: Compute Critical Parameters")

st.markdown("""
The critical parameters (μ_c, u_c) define the minimum intervention needed for Product 2 to survive.
These are computed using **Lemma 3.2** from the research paper.
""")

col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    if st.button("🔄 Compute Critical Parameters", type="primary", use_container_width=True):
        with st.spinner("Computing fixed point x*, Perron eigenvector, and critical parameters..."):
            mu_c, u_c, x_star = compute_critical_mu_u(A, tau1, tau2)
        st.session_state["mu_c"] = mu_c
        st.session_state["u_c"] = u_c
        st.session_state["x_star"] = x_star
        st.success(f"✅ Computed: μ_c = {mu_c:.6g}")

with col3:
    if "mu_c" in st.session_state:
        st.metric("μ_c (Critical Intervention)", f"{st.session_state['mu_c']:.6g}")
        with st.expander("View u_c Statistics"):
            st.write(pd.Series(st.session_state['u_c']).describe())

if "mu_c" not in st.session_state:
    st.info("👆 Please compute critical parameters first to proceed with analysis.")
    st.stop()

mu_c = st.session_state["mu_c"]
u_c = st.session_state["u_c"]
x_star = st.session_state["x_star"]

# -----------------------
# Generate Full Report Button
# -----------------------
st.divider()
st.header("📄 Generate Comprehensive Report")

st.markdown("""
Generate a complete PDF report that includes:
- **All analyses** with current settings (Budget Sweep + Epsilon Sweep + Baselines)
- **Critical parameters** and graph statistics
- **Visualizations** of all results
- **Comparative analysis** across methods
- **Professional formatting** ready for presentation or publication
""")

if st.button("📊 Generate Full PDF Report", type="primary", use_container_width=True, key="full_report_btn"):
    with st.spinner("🔄 Running complete analysis and generating report..."):
        try:
            # Collect graph information
            if src_choice == "Sample":
                graph_name = sample
            else:
                graph_name = uploaded.name if uploaded else "Custom Graph"
            
            graph_stats = {
                'N': N,
                'E': int(A.sum()/2),
                'avg_degree': A.sum(axis=1).mean()
            }
            
            # Collect parameters
            params = {
                'tau1': tau1,
                'tau2': tau2,
                'cost_scheme': cost_scheme,
                'eps_min': eps_min,
                'eps_max': eps_max,
                'num_eps': num_eps,
                'iters': iters,
                'delta_C': delta_C,
                'num_steps': num_steps
            }
            
            # Collect critical parameters
            u_c_series = pd.Series(u_c)
            critical_params_data = {
                'mu_c': mu_c,
                'u_c_stats': {
                    'mean': u_c_series.mean(),
                    'std': u_c_series.std(),
                    'min': u_c_series.min(),
                    'max': u_c_series.max(),
                    'sum': u_c_series.sum()
                }
            }
            
            st.write("✅ Step 1/4: Running Budget Sweep Analysis...")
            
            # Run Budget Sweep
            w = costs
            C_min = np.sqrt(mu_c) * np.dot(w, u_c)
            budgets = [C_min + k * delta_C for k in range(num_steps)]
            
            budget_rows = []
            for C in budgets:
                u_opt = algorithm1_local_search(
                    A, tau1, tau2, mu_c, u_c.copy(), x_star, eps_min,
                    iters=iters, costs=costs, budget_C=C
                )
                x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_opt)
                budget_rows.append({
                    "Budget": float(C),
                    "AvgX": float(x.mean()),
                    "AvgY": float(y.mean()),
                    "ExpectedCommunitySize": float(np.sum(u_opt))
                })
            
            budget_df = pd.DataFrame(budget_rows)
            
            st.write("✅ Step 2/4: Running Epsilon Sweep (Algorithm 1)...")
            
            # Run Epsilon Sweep
            log_eps = np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)
            epsilon_rows = []
            
            # Our approach - parallel
            def process_eps(eps):
                u = algorithm1_local_search(A, tau1, tau2, mu_c, u_c, x_star, eps, iters=iters, costs=costs)
                x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u)
                return {
                    "method": "Our Approach",
                    "epsilon": float(eps),
                    "AvgX": float(x.mean()),
                    "AvgY": float(y.mean())
                }
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_eps, eps) for eps in log_eps]
                for future in as_completed(futures):
                    epsilon_rows.append(future.result())
            
            st.write("✅ Step 3/4: Running Baseline Comparisons...")
            
            # Baselines
            if enable_deg:
                u_deg = baseline_degree(A)
                x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_deg)
                for eps in log_eps:
                    epsilon_rows.append({
                        "method": "Degree",
                        "epsilon": float(eps),
                        "AvgX": float(x.mean()),
                        "AvgY": float(y.mean())
                    })
            
            if enable_evc:
                u_evc = baseline_eigenvector(A)
                x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_evc)
                for eps in log_eps:
                    epsilon_rows.append({
                        "method": "EVC",
                        "epsilon": float(eps),
                        "AvgX": float(x.mean()),
                        "AvgY": float(y.mean())
                    })
            
            if enable_ns:
                u_ns = baseline_netshield(A, k=int(ns_k))
                if u_ns.sum() > 0:
                    u_ns = u_ns / u_ns.sum()
                x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_ns)
                for eps in log_eps:
                    epsilon_rows.append({
                        "method": "NetShield",
                        "epsilon": float(eps),
                        "AvgX": float(x.mean()),
                        "AvgY": float(y.mean())
                    })
            
            epsilon_df = pd.DataFrame(epsilon_rows)
            
            st.write("✅ Step 4/4: Generating visualizations and PDF...")
            
            # Create figures
            figures = {}
            
            # Budget sweep plots
            fig1 = plt.figure(figsize=(8, 5))
            plt.plot(budget_df["Budget"], budget_df["AvgY"], marker="o", linewidth=2, color='#1f77b4')
            plt.axvline(x=C_min, color='red', linestyle='--', linewidth=2, label='C_min')
            plt.xlabel("Budget C")
            plt.ylabel("Average Market Share (Product 2)")
            plt.title("Product 2 Survival vs Budget")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            figures["Budget Sweep: Product 2 Market Share"] = fig1
            
            fig2 = plt.figure(figsize=(8, 5))
            plt.plot(budget_df["Budget"], budget_df["ExpectedCommunitySize"], marker="s", linewidth=2, color='purple')
            plt.axvline(x=C_min, color='red', linestyle='--', linewidth=2, label='C_min')
            plt.xlabel("Budget C")
            plt.ylabel("Expected Community Size")
            plt.title("Community Expansion vs Budget")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            figures["Budget Sweep: Community Size"] = fig2
            
            # Epsilon sweep comparison
            fig3 = plt.figure(figsize=(10, 6))
            methods = epsilon_df['method'].unique()
            colors_map = {'Our Approach': '#1f77b4', 'Degree': '#ff7f0e', 'EVC': '#2ca02c', 'NetShield': '#d62728'}
            for m in methods:
                sub = epsilon_df[epsilon_df['method'] == m].sort_values('epsilon')
                plt.semilogx(sub['epsilon'], sub['AvgY'], marker='o', label=m, 
                           linewidth=2, color=colors_map.get(m))
            plt.xlabel("Epsilon (ε)")
            plt.ylabel("Average Market Share (Product 2)")
            plt.title("Method Comparison: Product 2 Market Share")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            figures["Epsilon Sweep: Method Comparison"] = fig3
            
            # Generate PDF
            pdf_buffer = create_pdf_report(
                graph_name=graph_name,
                graph_stats=graph_stats,
                params=params,
                critical_params=critical_params_data,
                budget_results=budget_df,
                epsilon_results=epsilon_df,
                figures=figures,
                filename="bi_sis_analysis_report.pdf"
            )
            
            # Store results in session state
            st.session_state['full_report_budget'] = budget_df
            st.session_state['full_report_epsilon'] = epsilon_df
            st.session_state['full_report_pdf'] = pdf_buffer
            
            st.success("✅ Full report generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Show download button if report exists
if 'full_report_pdf' in st.session_state:
    st.download_button(
        label="📥 Download PDF Report",
        data=st.session_state['full_report_pdf'],
        file_name=f"bi_sis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    # Show preview of results
    with st.expander("📊 Preview Report Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Budget Sweep Results")
            st.dataframe(st.session_state['full_report_budget'])
        with col2:
            st.subheader("Epsilon Sweep Results")
            st.dataframe(st.session_state['full_report_epsilon'])

# -----------------------
# Analysis Options
# -----------------------
st.divider()
st.header("🚀 Step 2: Run Individual Analyses")
st.markdown("*Or run individual analyses below if you prefer step-by-step exploration:*")

# Create tabs for different analyses
tab1, tab2 = st.tabs(["💰 Budget Sweep Analysis", "🔬 Epsilon Sweep (Algorithm 1)"])

# -----------------------
# Tab 1: Budget sweep
# -----------------------
with tab1:
    st.markdown("""
    **Budget Sweep Analysis** shows how Product 2's market share changes as you increase the advertising budget.
    This helps you understand the relationship between investment and market penetration.
    """)
    
    st.markdown("---")
    
    log_eps = np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)

    if st.button("🚀 Run Budget Sweep", type="primary", use_container_width=True):
        w = costs
        C_min = np.sqrt(mu_c) * np.dot(w, u_c)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Critical Budget (C_min)", f"{C_min:.4f}")
        with col2:
            st.metric("📈 Max Budget", f"{C_min + (num_steps-1)*delta_C:.4f}")

        budgets = [C_min + k * delta_C for k in range(num_steps)]

        # Process budgets sequentially (simpler and often faster)
        rows = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, C in enumerate(budgets):
            status_text.text(f"Processing budget {idx+1}/{len(budgets)}: C = {C:.4f}")
            
            u_opt = algorithm1_local_search(
                A, tau1, tau2, mu_c, u_c.copy(), x_star, eps_min,
                iters=iters, costs=costs, budget_C=C
            )
            x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_opt)
            expected_community_size = float(np.sum(u_opt))
            
            rows.append({
                "Budget": float(C),
                "AvgX": float(x.mean()),
                "AvgY": float(y.mean()),
                "ExpectedCommunitySize": expected_community_size
            })
            
            progress_bar.progress((idx + 1) / len(budgets))
        
        progress_bar.empty()
        status_text.empty()
        
        df = pd.DataFrame(rows)
        st.session_state["budget_results_df"] = df

        st.success("✅ Budget sweep completed!")
        
        st.subheader("📊 Results Summary")
        st.dataframe(df, use_container_width=True)

        # --- Plot AvgY vs Budget ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Product 2 Market Share vs Budget")
            fig1 = plt.figure(figsize=(8, 5))
            plt.plot(df["Budget"], df["AvgY"], marker="o", linewidth=2, markersize=6, color='#1f77b4')
            plt.axvline(x=C_min, color='red', linestyle='--', linewidth=2, label='C_min (Critical)')
            plt.xlabel("Budget C", fontsize=11)
            plt.ylabel("Average Market Share (Product 2)", fontsize=11)
            plt.title("Product 2 Survival vs Budget", fontsize=12, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, which="both", ls=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig1)

        with col2:
            st.subheader("Community Size vs Budget")
            fig2 = plt.figure(figsize=(8, 5))
            plt.plot(df["Budget"], df["ExpectedCommunitySize"], marker="s", linewidth=2, markersize=6, color="purple")
            plt.axvline(x=C_min, color='red', linestyle='--', linewidth=2, label='C_min (Critical)')
            plt.xlabel("Budget C", fontsize=11)
            plt.ylabel("Expected Community Size (Σu_i)", fontsize=11)
            plt.title("Expected Number of Nodes in Community vs Budget", fontsize=12, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, which="both", ls=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig2)

        # Download CSV
        st.subheader("💾 Export Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, file_name="budget_sweep_results.csv", mime="text/csv")
        with col2:
            png1 = fig_to_png_bytes(fig1)
            st.download_button("📥 Download Plot 1", png1, file_name="budget_vs_market_share.png", mime="image/png")
        with col3:
            png2 = fig_to_png_bytes(fig2)
            st.download_button("📥 Download Plot 2", png2, file_name="budget_vs_community_size.png", mime="image/png")

# -----------------------
# Tab 2: Epsilon sweep
# -----------------------
with tab2:
    st.markdown("""
    **Epsilon Sweep (Algorithm 1)** optimizes the community targeting strategy using local perturbation search.
    Different epsilon values control the granularity of optimization.
    """)
    
    st.markdown("---")
    st.markdown("---")
    
    log_eps = np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)
    
    if st.button("🔬 Run Epsilon Sweep", type="primary", use_container_width=True):
        rows = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Function to process epsilon for our approach
        def process_epsilon_ours(eps, A, tau1, tau2, mu_c, u_c, x_star, iters, costs):
            u = algorithm1_local_search(A, tau1, tau2, mu_c, u_c, x_star, eps, iters=iters, costs=costs)
            x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u)
            return {
                "method": "Our Approach",
                "epsilon": float(eps),
                "AvgX": float(x.mean()),
                "AvgY": float(y.mean())
            }
        
        # Our approach - USE PARALLEL PROCESSING (this is the major computational task)
        status_text.text("Running Algorithm 1 (Our Approach) - Parallel Processing...")
        avgX_list, avgY_list = [], []
        
        max_workers = min(4, len(log_eps))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(
                process_epsilon_ours,
                A=A, tau1=tau1, tau2=tau2, mu_c=mu_c, u_c=u_c,
                x_star=x_star, iters=iters, costs=costs
            )
            
            futures = {executor.submit(process_func, eps): eps for eps in log_eps}
            
            completed = 0
            eps_results = []
            total_tasks = len(log_eps) + enable_deg + enable_evc + enable_ns
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    eps_results.append(result)
                    completed += 1
                    progress_bar.progress(completed / total_tasks)
                except Exception as exc:
                    st.error(f"Epsilon processing failed: {exc}")
            
            # Sort by epsilon and extract results
            eps_results = sorted(eps_results, key=lambda x: x["epsilon"])
            rows.extend(eps_results)
            avgX_list = [r["AvgX"] for r in eps_results]
            avgY_list = [r["AvgY"] for r in eps_results]

        current_step = len(log_eps)

        # Baselines - SEQUENTIAL (fast, only 1 simulation each)
        if enable_deg:
            status_text.text("Running Degree Centrality Baseline...")
            u_deg = baseline_degree(A)
            x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_deg)
            for eps in log_eps:
                rows.append({
                    "method": "Degree",
                    "epsilon": float(eps),
                    "AvgX": float(x.mean()),
                    "AvgY": float(y.mean())
                })
            current_step += 1
            progress_bar.progress(current_step / total_tasks)
        
        if enable_evc:
            status_text.text("Running Eigenvector Centrality Baseline...")
            u_evc = baseline_eigenvector(A)
            x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_evc)
            for eps in log_eps:
                rows.append({
                    "method": "EVC",
                    "epsilon": float(eps),
                    "AvgX": float(x.mean()),
                    "AvgY": float(y.mean())
                })
            current_step += 1
            progress_bar.progress(current_step / total_tasks)
        
        if enable_ns:
            status_text.text("Running NetShield Baseline...")
            u_ns = baseline_netshield(A, k=int(ns_k))
            if u_ns.sum() > 0:
                u_ns = u_ns / u_ns.sum()
            x, y = simulate_fixed_point(A, tau1, tau2, mu_c, u_ns)
            for eps in log_eps:
                rows.append({
                    "method": "NetShield",
                    "epsilon": float(eps),
                    "AvgX": float(x.mean()),
                    "AvgY": float(y.mean())
                })
            current_step += 1
            progress_bar.progress(current_step / total_tasks)

        progress_bar.empty()
        status_text.empty()
        
        df = pd.DataFrame(rows)
        st.session_state["results_df"] = df

        st.success("✅ Epsilon sweep completed!")

        st.subheader("📊 Results Summary")
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 Visualization: Our Approach")
        res_ours = list(zip(log_eps, avgX_list, avgY_list))
        f1, f2 = plot_market_share_vs_eps(res_ours, title="Our Approach")
        
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(f1)
        with col2:
            st.pyplot(f2)

        st.subheader("💾 Export Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            png1 = fig_to_png_bytes(f1)
            st.download_button("📥 Download Plot (Product 2)", png1, file_name="epsilon_product2.png", mime="image/png")
        with col2:
            png2 = fig_to_png_bytes(f2)
            st.download_button("📥 Download Plot (Product 1)", png2, file_name="epsilon_product1.png", mime="image/png")
        with col3:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, file_name="epsilon_sweep_results.csv", mime="text/csv")

# -----------------------
# Compare methods: AvgY vs epsilon
# -----------------------
if "results_df" in st.session_state:
    st.divider()
    st.header("📊 Step 3: Method Comparison")
    st.markdown("Compare the performance of our approach against baseline strategies.")
    
    df = st.session_state["results_df"]
    methods = df["method"].unique()

    fig = plt.figure(figsize=(10, 6))
    colors = {'Our Approach': '#1f77b4', 'Degree': '#ff7f0e', 'EVC': '#2ca02c', 'NetShield': '#d62728'}
    
    for m in methods:
        sub = df[df["method"] == m]
        sub = sub.sort_values("epsilon")
        plt.xscale("log")
        color = colors.get(m, None)
        plt.plot(sub["epsilon"], sub["AvgY"], marker="o", label=m, linewidth=2, markersize=6, color=color)
    
    plt.xlabel("Epsilon (ε)", fontsize=12)
    plt.ylabel("Average Market Share (Product 2)", fontsize=12)
    plt.title("Market Share Comparison Across Different Strategies", fontsize=13, fontweight='bold')
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("💾 Export Comparison")
    col1, col2 = st.columns(2)
    with col1:
        png = fig_to_png_bytes(fig)
        st.download_button("📥 Download Comparison Plot", png, file_name="comparison_AvgY.png", mime="image/png")
    with col2:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Full Results CSV", csv, file_name="full_results.csv", mime="text/csv")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Keeping Up with the Winner</strong> - bi-SIS Community Targeting Dashboard</p>
    <p>Based on the research paper: <em>"Keeping Up with the Winner! Targeted Advertisement to Communities in Social Networks"</em></p>
    <p><a href="https://arxiv.org/abs/2403.19903" target="_blank">arXiv:2403.19903</a></p>
</div>
""", unsafe_allow_html=True)
