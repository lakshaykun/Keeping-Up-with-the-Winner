# Keeping Up with the Winner

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2403.19903-b31b1b.svg)](https://arxiv.org/abs/2403.19903)

A complete, reproducible Python implementation of the research paper **"Keeping Up with the Winner! Targeted Advertisement to Communities in Social Networks"** ([arXiv:2403.19903](https://arxiv.org/abs/2403.19903)).

**🌐 Live Demo**: [keepingupwiththewinner.streamlit.app](https://keepingupwiththewinner.streamlit.app/)

This project implements the **bi-SIS (bidirectional Susceptible-Infected-Susceptible)** model for competing product diffusion in social networks, along with optimization algorithms for strategic community targeting under budget constraints.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation & Quick Start](#installation--quick-start)
- [Dashboard Features](#dashboard-features)
- [Usage Guide](#usage-guide)
- [Model Details](#model-details)
- [Running Experiments](#running-experiments)
- [Performance & Optimization](#performance--optimization)
- [Implementation Notes](#implementation-notes)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

The bi-SIS model simulates two competing products spreading through a network, where nodes can be in one of three states:
- **Infected by Product 1 (x)**: Nodes using the first product
- **Infected by Product 2 (y)**: Nodes using the second product  
- **Susceptible (S = 1 - x - y)**: Nodes not using either product

The model enables strategic intervention through targeted community influence (parameter **u**) to maximize the market share of Product 2 under various cost schemes and budget constraints.

### Key Features

- ✅ **Critical Parameter Computation**: Implements Lemma 3.2 to compute the critical intervention pair (μ_c, u_c)
- ✅ **Algorithm 1**: Local perturbation search for optimal targeting strategy  
- ✅ **Baseline Comparisons**: Degree centrality, Eigenvector centrality, and NetShield-inspired approaches
- ✅ **Multiple Cost Schemes**: Homogeneous, degree-based, and eigenvector-based cost functions
- ✅ **Budget Sweep Analysis**: Analyze market share across different budget levels
- ✅ **Interactive Dashboard**: Modern, user-friendly Streamlit web UI with real-time progress tracking
- ✅ **Optimized Performance**: Parallel processing for computationally intensive tasks (2-4x faster)
- ✅ **Experiment Reproducibility**: Scripts to reproduce research figures from the paper
- ✅ **Export Capabilities**: Download results as CSV and visualizations as PNG
- ✅ **PDF Report Generation**: One-click comprehensive reports with all analyses and visualizations
- ✅ **Adaptive Algorithms**: Smart optimization based on graph size and task complexity

---

## 📁 Project Structure

```
Keeping-Up-with-the-Winner/
├── dashboard/
│   └── app.py                    # Streamlit web application for interactive analysis
│
├── data/
│   ├── README.md                 # Data format specifications and dataset guide
│   └── samples/
│       ├── toy_karate.edgelist   # Zachary's Karate Club network (34 nodes)
│       └── toy_erdos_renyi.edgelist  # Random Erdős-Rényi graph (100 nodes)
│
├── experiments/
│   ├── reproduce_fig4_7.py       # Script to reproduce Figures 4-7 from the paper
│   └── reproduce_fig8.py         # Placeholder for baseline comparison experiments
│
├── src/
│   ├── algorithm1.py             # Local perturbation search (Algorithm 1)
│   ├── baselines.py              # Baseline targeting strategies (Degree, Eigenvector, NetShield)
│   ├── bisis.py                  # bi-SIS ODE model and fixed-point solver
│   ├── critical_params.py        # Critical parameter computation (Lemma 3.2)
│   ├── pdf_report.py             # PDF report generation with comprehensive analysis
│   ├── plots.py                  # Visualization utilities for results
│   ├── run_experiments.py        # Experiment orchestration and epsilon-sweep logic
│   └── utils.py                  # Graph loading and utility functions
│
├── requirements.txt              # Python package dependencies
├── README.md                     # This file
├── LICENSE                       # MIT License
├── Keeping Up with the Winner.pdf  # The original research paper (arXiv:2403.19903)
└── Keeping Up with the Winner!.pptx  # Presentation slides for the research
```

### Directory Explanations

#### `dashboard/`
Contains the **Streamlit web application** (`app.py`) that provides an interactive interface for:
- Loading network graphs (sample datasets or custom uploads)
- Configuring model parameters (τ₁, τ₂, cost schemes, budget constraints)
- Computing critical parameters and running optimization algorithms
- Visualizing results with interactive plots and real-time progress tracking
- Comparing Algorithm 1 against baseline strategies
- Exporting results as CSV/PNG files
- **Performance**: Optimized with parallel processing for epsilon sweeps (2-4x faster)

#### `data/`
Storage for network data files:
- **`README.md`**: Documentation on supported data formats (edgelist, adjacency CSV) and SNAP dataset references
- **`samples/`**: Pre-loaded example networks for quick experimentation
  - `toy_karate.edgelist`: Zachary's Karate Club (classic small-world network)
  - `toy_erdos_renyi.edgelist`: Random graph for testing scalability

#### `experiments/`
Scripts for reproducing figures from the research paper:
- **`reproduce_fig4_7.py`**: Generates epsilon-sweep analysis plots showing market share optimization
- **`reproduce_fig8.py`**: Baseline comparison framework (primarily handled via dashboard)
- Results are saved to `experiments/outputs/` (created at runtime)

#### `src/`
Core implementation modules:

| File | Purpose |
|------|---------|
| `algorithm1.py` | Implements the local perturbation search algorithm with budget-aware node scoring and zero-sum perturbations (optimized for performance) |
| `baselines.py` | Baseline strategies: degree centrality, eigenvector centrality, and simplified NetShield approach |
| `bisis.py` | ODE system solver for bi-SIS dynamics using `scipy.integrate.solve_ivp` (RK45 method) with pre-computation optimizations |
| `critical_params.py` | Computes critical intervention parameters (μ_c, u_c) via Perron-Frobenius analysis and fixed-point iteration |
| `pdf_report.py` | Generates comprehensive PDF reports with executive summary, all analyses, visualizations, and professional formatting using ReportLab |
| `plots.py` | Matplotlib-based visualization functions for market share, budget sweeps, and comparative analysis |
| `run_experiments.py` | High-level experiment runner that coordinates epsilon sweeps and result aggregation |
| `utils.py` | Graph I/O utilities (edgelist/CSV parsing, adjacency matrix conversion, symmetry enforcement) |

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Keeping Up with the Winner"
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

- **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
- **Windows (cmd)**: `.\venv\Scripts\activate.bat`
- **Linux/Mac**: `source venv/bin/activate`

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

> **💡 Tip**: The dashboard includes an interactive tutorial - click the "📋 How to Use This Dashboard" expander at the top for step-by-step guidance.

---

## 📚 Research Materials

This repository includes complete research documentation:

- **📄 Paper** (`Keeping Up with the Winner.pdf`): The original peer-reviewed research paper
  - Full technical details of the bi-SIS model
  - Theoretical analysis and proofs
  - Experimental results on SNAP datasets
  - Reference: [arXiv:2403.19903](https://arxiv.org/abs/2403.19903)

- **🎯 Presentation Slides** (`Keeping Up with the Winner!.pptx`): Conference presentation
  - Visual overview of the research problem
  - Model explanation with diagrams
  - Key findings and results
  - Algorithm walkthrough
  - Perfect for understanding the work at a high level

---

## 🎨 Dashboard Features

### Modern User Interface
- **Intuitive Layout**: Numbered sections guide you through the workflow
- **Helpful Tooltips**: Every parameter has contextual help text
- **Real-time Progress**: Progress bars and status updates for all operations
- **Tabbed Interface**: Separate tabs for Budget Sweep and Epsilon Sweep analysis
- **Metric Cards**: Key statistics displayed prominently (nodes, edges, avg degree, μ_c)
- **Expandable Sections**: Advanced options and help text available on-demand

### Visualization Features
- **Side-by-side Plots**: Compare multiple metrics simultaneously
- **Publication-quality Figures**: High-DPI plots with proper formatting
- **Interactive Charts**: Hover tooltips and zoom capabilities
- **Color-coded Methods**: Consistent color scheme across all comparisons
- **Multiple Export Formats**: Download plots as PNG and data as CSV

### Analysis Capabilities
- **Budget Sweep**: 
  - Visualize market share vs. budget investment
  - Track expected community size growth
  - Identify critical budget threshold (C_min)
  
- **Epsilon Sweep**: 
  - Optimize targeting strategy with parallel processing
  - Compare multiple methods in real-time
  - Log-scale plots for epsilon analysis
  
- **Method Comparison**:
  - Algorithm 1 (Our Approach) - optimized local search
  - Degree Centrality - target high-degree nodes
  - Eigenvector Centrality - target influential nodes
  - NetShield - network protection-inspired approach

- **PDF Report Generation**:
  - One-click comprehensive report with all analyses
  - Professional formatting with executive summary
  - Embedded visualizations and statistical tables
  - Includes graph statistics, parameters, critical analysis, budget/epsilon results, and method comparisons
  - Timestamped downloads for archival

---

## 📚 Usage Guide

### Dashboard Workflow

#### 1. Load a Graph
- **Sample Networks**: Choose from pre-loaded examples (Zachary Karate Club, Erdős-Rényi)
- **Upload Custom Graph**: 
  - Edgelist format: whitespace-separated node pairs (e.g., `node1 node2`)
  - Adjacency CSV: square matrix with 0/1 entries

#### 2. Configure Model Parameters
- **tau1 (τ₁)**: Spreading rate for Product 1 (default: 0.8)
  - Higher values mean Product 1 spreads more aggressively
- **tau2 (τ₂)**: Spreading rate for Product 2 (default: 0.05)
  - Lower values require stronger intervention to compete

#### 3. Set Analysis Configuration
- **Budget Sweep**: Configure budget increment (ΔC) and number of steps
- **Epsilon Sweep**: Set epsilon range (10⁻⁸ to 10⁻¹) and number of points
- **Algorithm Settings**: Choose local-search iterations (default: 1)
- **Cost Scheme**: Select between homogeneous, degree-based, or eigenvector-based costs

#### 4. Compute Critical Parameters
Click **"Compute Critical Parameters"** to:
- Derive the fixed point x* via single-product SIS simulation
- Calculate the Perron-Frobenius eigenvector of Sx·A
- Compute critical intervention parameters via Lemma 3.2
- View μ_c and u_c statistics

#### 5. Run Analysis
Choose your analysis type:
- **Budget Sweep Analysis**: Analyze how market share changes with different budgets
  - Visualize Product 2 survival vs budget
  - Track expected community size expansion
  - Identify critical budget threshold
  
- **Epsilon Sweep (Algorithm 1)**: Optimize targeting strategy
  - **Uses parallel processing** for faster computation (2-4x speedup)
  - Run local perturbation search across epsilon values
  - Compare against baseline strategies (Degree, Eigenvector, NetShield)
  - Real-time progress tracking

#### 6. View and Export Results
- **Interactive Plots**: 
  - Market share vs. epsilon (log scale)
  - Market share vs. budget
  - Expected community size analysis
  - Method comparison charts
- **CSV Export**: Download numerical results for further analysis
- **PNG Export**: Save publication-quality figures
- **PDF Report**: Generate comprehensive report with one click
  - Runs all analyses with your custom settings
  - Includes executive summary, graph statistics, all parameters
  - Embeds all visualizations and result tables
  - Professional formatting ready for presentation
  - Timestamped download (e.g., `bi_sis_report_20251113_143025.pdf`)

---

## 🔬 Model Details

### bi-SIS Dynamics

The model is governed by the following system of ODEs:

$$
\begin{aligned}
\frac{dx}{dt} &= \tau_1 \cdot S \cdot (Ax) - x \\
\frac{dy}{dt} &= \tau_2 \cdot S \cdot (By) - y
\end{aligned}
$$

Where:
- **A**: Adjacency matrix of the network
- **B = A + μ · u·u^T**: Augmented adjacency with rank-1 intervention
- **S = 1 - x - y**: Susceptible population
- **μ**: Intervention strength parameter
- **u**: Community targeting vector (budget allocation)

### Algorithm 1: Local Perturbation Search

The algorithm implements a budget-aware local search that:

1. **Computes Perron-Frobenius eigenvector (ν)** around the critical point
2. **Scores nodes** by ν_i / w_i (benefit per unit cost)
3. **Applies perturbations**:
   - Increment high-scoring nodes by +ε
   - Decrement low-scoring nodes by -ε
4. **Maintains budget constraint**: Σ w_i · δ_i ≈ 0
5. **Projects to feasible region** [0,1] and iterates

**Pseudocode**:
```
for each iteration:
    Compute B = Sx @ (A + μ_c * u·u^T)
    ν = Perron_eigenvector(B)
    score = ν / w
    Sort nodes by score descending
    δ[top_half] = +ε, δ[bottom_half] = -ε
    u_new = clip(u + δ, 0, 1)
    Rescale to satisfy budget constraint
```

### Critical Parameter Computation (Lemma 3.2)

**Steps**:
1. Compute **x*** by single-SIS simulation of Product 1
2. Construct **Sx = diag(1 - x*)**
3. Find Perron-Frobenius eigenvector **v** of **Sx·A**
4. Compute **u_c ∝ Sx⁻¹·v** (normalized)
5. Calculate **μ_c = (1/τ₂ - λ_max(Sx·A)) / (u_c^T·Sx·u_c)**

---

## 🧪 Running Experiments

### Reproduce Paper Figures (Command-Line)

To programmatically reproduce figures from the research paper:

```bash
python experiments/reproduce_fig4_7.py
```

**Output**: 
- Saves plots to `experiments/outputs/`
- Generates market share vs. epsilon analysis
- Uses Karate Club as demo (replace with SNAP datasets for full reproduction)

### Interactive Exploration (Recommended)

The Streamlit dashboard provides the most flexible workflow:

```bash
streamlit run dashboard/app.py
```

Supports:
- Real-time parameter tuning
- Multiple baseline comparisons
- Budget sweep analysis
- Interactive plot customization

---

## 📊 Data Formats

### Edgelist Format

```
node1 node2
node1 node3
node2 node4
...
```

- Whitespace-separated node pairs (one edge per line)
- Node IDs can be integers or strings
- Undirected edges (automatically symmetrized)

### Adjacency CSV Format

```csv
0,1,0,1
1,0,1,0
0,1,0,1
1,0,1,0
```

- Square matrix with 0/1 entries
- Row i, Column j = 1 indicates edge from node i to node j
- Must be symmetric for undirected graphs

### SNAP Datasets

For datasets used in the original research:
- **Social circles — Facebook** (4,039 nodes)
- **Facebook pages of public figures** (~11,565 nodes)

See `data/README.md` for download links and conversion instructions.

---

## 📦 Dependencies

All required packages are listed in `requirements.txt`:

```
numpy          # Numerical computations and vectorized operations
scipy          # ODE integration and scientific computing
networkx       # Graph operations and algorithms
matplotlib     # Plotting and visualization
pandas         # Data manipulation and export
streamlit      # Interactive web dashboard
reportlab      # PDF report generation
Pillow         # Image processing for PDF embedding
```

**Installation**:
```bash
pip install -r requirements.txt
```

**Note**: No additional dependencies required for parallel processing (uses Python's built-in `concurrent.futures`)

**System Requirements**:
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for large graphs)
- Multi-core CPU recommended for parallel processing benefits

---

## 📄 License

---

## ⚡ Performance & Optimization

### Parallel Processing
This implementation uses intelligent parallel processing for optimal performance:

- **Epsilon Sweep (Algorithm 1)**: Processes multiple epsilon values concurrently
  - Uses `ThreadPoolExecutor` with up to 4 worker threads
  - **Speedup**: 2-4x faster on multi-core systems
  - Real-time progress tracking shows parallel execution status

- **Budget/Baseline Analysis**: Sequential processing (already optimized)
  - Lower overhead for quick tasks
  - Instant results for baseline comparisons

### Vectorized Operations
All core computations use NumPy vectorization:
- Matrix multiplications using `@` operator
- Element-wise operations (`np.clip`, `np.maximum`)
- Pre-computed matrices (Sx, outer products) to avoid redundant calculations

### Memory Efficiency
- Dense output disabled in ODE solver (`dense_output=False`)
- Pre-allocated arrays reused across iterations
- Efficient data types (`np.float64`) for numerical stability

### Performance Benchmarks

| Graph Size | Budget Sweep | Epsilon Sweep | Total Time |
|-----------|--------------|---------------|------------|
| Small (N=34, Karate) | ~2-3 sec | ~5-8 sec | ~10 sec |
| Medium (N=100-500) | ~5-10 sec | ~15-25 sec | ~30-40 sec |
| Large (N=1000+) | ~15-30 sec | ~40-80 sec | ~1-2 min |

*Times are approximate and depend on CPU, epsilon/budget points, and iterations*

### Optimization Tips

1. **For Quick Exploration**:
   - Use 10 epsilon points instead of 15
   - Set local-search iterations to 1 (default)
   - Start with smaller graphs (Karate Club)

2. **For Production Analysis**:
   - Use 15-20 epsilon points for smoother curves
   - Increase budget steps to 20-30 for detailed analysis
   - Enable all baselines for comprehensive comparison

3. **For Large Graphs (N > 1000)**:
   - Reduce epsilon points to 8-10
   - Consider adaptive tolerance (increase rtol to 1e-5)
   - Use homogeneous costs to avoid centrality computation overhead

---

## 🛠️ Implementation Notes

### Performance Optimizations
- **Parallel Processing**: Epsilon sweep uses ThreadPoolExecutor for 2-4x speedup on multi-core systems
  - Automatically processes multiple epsilon values concurrently
  - Up to 4 worker threads for optimal performance
  - Sequential fallback for baselines (already fast)
- **Vectorized Operations**: NumPy-based vectorization for all matrix operations
- **Pre-computation**: Outer products and diagonal matrices computed once and reused
- **Memory Efficiency**: Dense output disabled in ODE solver to reduce memory usage

### Numerical Methods
- **Fixed Point Computation**: ODE integration to steady state (T=2000) using `scipy.integrate.solve_ivp` with RK45 method
- **Relative tolerance**: 1e-6
- **Absolute tolerance**: 1e-8
- **Maximum step size**: 1.0 (adaptive)

### Budget Constraints
- **Constraint formula**: √μ · Σ(w_i · u_i) = C
- **Scaling**: Initial u_c is rescaled to satisfy budget after perturbations
- **Projection**: Values clipped to [0,1] to maintain valid probabilities

### Baseline Implementations
- **Degree**: NetworkX degree centrality, normalized
- **Eigenvector**: NetworkX eigenvector centrality via power iteration
- **NetShield**: Simplified variant using squared Perron-Frobenius eigenvector for top-k selection

### Performance Characteristics
- **Small graphs (N < 100)**: Analysis completes in seconds
- **Medium graphs (100 < N < 1000)**: Parallel processing provides significant speedup (2-4x)
- **Large graphs (N > 1000)**: Consider reducing epsilon/budget points for faster initial exploration
- **Real-time feedback**: Progress bars and status updates throughout all operations

---

## ⚠️ Troubleshooting

### Common Issues

**1. Dashboard won't start**
```bash
# Check if streamlit is installed
pip list | grep streamlit

# Reinstall if necessary
pip install --upgrade streamlit
```

**2. Graph loading errors**
- Verify file format matches expected structure (edgelist or CSV)
- Check for missing delimiters or malformed rows
- Ensure graph is connected (isolated nodes may cause issues)

**3. Slow computation**
- Epsilon sweep uses parallel processing - you should see "Parallel Processing" in the status
- For very large graphs (N > 1000), reduce epsilon points (e.g., 10 instead of 15)
- Decrease local-search iterations (default: 1 is usually sufficient)
- Budget sweep processes sequentially (optimized for speed)
- Check CPU usage to verify parallel processing is active

**4. Numerical instability**
- Increase tau2 slightly (avoid very small values < 0.01)
- Check for negative eigenvalues in adjacency matrix
- Verify graph is undirected and symmetric

---

## License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## 📝 Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{keepingupwithwinner2024,
  title={Keeping Up with the Winner! Targeted Advertisement to Communities in Social Networks},
  author={[Authors from arXiv:2403.19903]},
  journal={arXiv preprint arXiv:2403.19903},
  year={2024}
}
```

**Paper**: [https://arxiv.org/abs/2403.19903](https://arxiv.org/abs/2403.19903)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional baseline strategies
- GPU acceleration for large graphs (CuPy integration)
- Advanced visualization features (interactive 3D plots)
- Support for directed/weighted networks
- Caching mechanism for repeated parameter sets
- Additional cost function schemes

Please open an issue or submit a pull request.

**Development Guidelines**:
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update README for significant changes

---

## 🎓 Quick Reference

### Key Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| tau1 | τ₁ | Spreading rate for dominant product | 0.5 - 1.0 |
| tau2 | τ₂ | Spreading rate for new product | 0.01 - 0.1 |
| epsilon | ε | Perturbation budget for local search | 10⁻⁸ - 10⁻¹ |
| mu_c | μ_c | Critical intervention strength | Computed |
| u_c | u_c | Critical community targeting vector | Computed |
| C | C | Advertising budget | C_min + k·ΔC |

### Common Workflows

**Scenario 1: Quick Analysis**
```bash
1. Launch dashboard
2. Select "Zachary Karate (34)" sample
3. Keep default parameters (τ₁=0.8, τ₂=0.05)
4. Click "Compute Critical Parameters"
5. Click "Run Budget Sweep" (results in ~10 seconds)
```

**Scenario 2: Method Comparison**
```bash
1. Load your network or use "Erdos-Renyi n=100"
2. Configure parameters
3. Compute critical parameters
4. Enable all baselines (Degree, EVC, NetShield)
5. Run "Epsilon Sweep" to compare methods
```

**Scenario 3: Custom Network Analysis**
```bash
1. Prepare edgelist file (node1 node2 per line)
2. Upload via "Upload (edgelist)" option
3. Select appropriate cost scheme (degree/eigenvector)
4. Adjust budget increment based on network size
5. Run full analysis and export results
```

### Output Files

| File | Description |
|------|-------------|
| `budget_sweep_results.csv` | Budget vs market share data |
| `budget_vs_market_share.png` | Product 2 survival plot |
| `budget_vs_community_size.png` | Community expansion plot |
| `epsilon_sweep_results.csv` | Epsilon optimization data |
| `epsilon_product2.png` | Product 2 vs epsilon |
| `epsilon_product1.png` | Product 1 vs epsilon |
| `comparison_AvgY.png` | All methods comparison |
| `full_results.csv` | Complete analysis dataset |
| `bi_sis_report_YYYYMMDD_HHMMSS.pdf` | Comprehensive PDF report with all analyses |

---

## 📞 Contact & Support

For questions or issues:
- 🐛 **Bug Reports**: [Open a GitHub Issue](https://github.com/lakshaykun/Keeping-Up-with-the-Winner/issues)
- 📖 **Theory Questions**: Refer to the [original research paper](https://arxiv.org/abs/2403.19903)
- 📊 **Dataset Help**: Check `data/README.md` for format specifications
- 💬 **Discussions**: Use GitHub Discussions for general questions

**Maintainer**: [@lakshaykun](https://github.com/lakshaykun)

---

## 🌟 Acknowledgments

- Research paper authors: Shailaja Mallick, Vishwaraj Doshi, Do Young Eun
- NetworkX and SciPy communities for excellent scientific computing tools
- Streamlit team for the interactive dashboard framework

---

**Last Updated**: November 13, 2025
