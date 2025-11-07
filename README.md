# Keeping Up with the Winner

This repository implements the research paper (`[**Keeping Up with the Winner! Targeted Advertisement to Communities in Social Networks**](https://arxiv.org/abs/2403.19903)`). The project includes algorithms for computing critical parameters, running local perturbation search (Algorithm 1), comparing baseline strategies, and provides an interactive Streamlit dashboard for exploration and analysis.

## Overview

The bi-SIS model simulates two competing products spreading through a network, where nodes can be in one of three states: infected by Product 1 (x), infected by Product 2 (y), or susceptible (S = 1 - x - y). The model allows for strategic intervention through targeted community influence (parameter u) to maximize the market share of Product 2.

### Key Features

- **Critical Parameter Computation**: Implements Lemma 3.2 to compute the critical intervention pair (μ_c, u_c)
- **Algorithm 1**: Local perturbation search for optimal targeting strategy under budget constraints
- **Baseline Comparisons**: Degree centrality, Eigenvector centrality, and NetShield-inspired approaches
- **Interactive Dashboard**: Streamlit-based UI for parameter exploration and visualization
- **Cost Schemes**: Supports homogeneous, degree-based, and eigenvector-based cost functions
- **Experiment Reproducibility**: Scripts to reproduce research figures

## Project Structure

```
.
├── dashboard/
│   └── app.py              # Streamlit web application
├── data/
│   ├── README.md           # Data format guide
│   └── samples/            # Example networks (Karate Club, Erdős-Rényi)
├── experiments/
│   ├── reproduce_fig4_7.py # Script for reproducing Figures 4-7
│   └── reproduce_fig8.py   # Script for baseline comparison (Figure 8)
├── src/
│   ├── algorithm1.py       # Local perturbation search implementation
│   ├── baselines.py        # Baseline targeting strategies
│   ├── bisis.py            # bi-SIS ODE model and fixed-point solver
│   ├── critical_params.py  # Critical parameter computation (Lemma 3.2)
│   ├── plots.py            # Visualization utilities
│   ├── run_experiments.py  # Experiment orchestration
│   └── utils.py            # Graph loading and utility functions
├── requirements.txt        # Python dependencies
├── Keeping Up with the Winner.pdf        # The Research Paper
└── README.md              # This file
```

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Keeping Up with the Winner"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows (PowerShell): `.\venv\Scripts\Activate.ps1`
   - Windows (cmd): `.\venv\Scripts\activate.bat`
   - Linux/Mac: `source venv/bin/activate`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Usage Guide

### Dashboard Workflow

1. **Load a Graph**:
   - Select from sample networks (Zachary Karate Club, Erdős-Rényi)
   - Upload your own network (edgelist or adjacency CSV format)

2. **Configure Model Parameters**:
   - `tau1` (τ₁): Spreading rate for Product 1 (default: 0.8)
   - `tau2` (τ₂): Spreading rate for Product 2 (default: 0.05)

3. **Set Algorithm Parameters**:
   - Epsilon range: Perturbation budget (default: 10⁻⁸ to 10⁻¹)
   - Number of epsilon points: Log-spaced values to test (default: 15)
   - Local-search iterations: Refinement steps per epsilon (default: 1)

4. **Choose Cost Scheme**:
   - **Homogeneous**: All nodes have equal cost (w_i = 1)
   - **Degree**: Cost proportional to node degree
   - **Eigenvector**: Cost based on eigenvector centrality

5. **Compute Critical Parameters**:
   - Click "Compute critical (μ_c, u_c)" to calculate the critical intervention pair
   - This derives the fixed point x* and Perron eigenvector

6. **Run Algorithm 1**:
   - Execute epsilon-sweep with local perturbation search
   - Compare against selected baselines (Degree, Eigenvector, NetShield)

7. **View and Export Results**:
   - Interactive plots showing market share vs. epsilon
   - Comparison plots across all methods
   - Download results as CSV or PNG

### Running Experiments

To reproduce research figures programmatically:

```bash
python experiments/reproduce_fig4_7.py
```

Results will be saved to `experiments/outputs/`.

## Model Details

### bi-SIS Dynamics

The model is governed by the following ODEs:

- **dx/dt** = τ₁ · S · (Ax) - x
- **dy/dt** = τ₂ · S · (Bx) - y

Where:
- A is the adjacency matrix
- B = A + μ · u·u^T (augmented with intervention)
- S = 1 - x - y (susceptible population)

### Algorithm 1: Local Perturbation Search

The algorithm implements a one-step perturbation-based local search that:
1. Computes the Perron-Frobenius eigenvector (ν) around the critical point
2. Scores nodes by ν_i / w_i (benefit per cost)
3. Pairs high-scoring nodes (+ε) with low-scoring nodes (-ε)
4. Maintains budget constraint: Σ w_i · δ_i ≈ 0
5. Projects to feasible region [0,1] and iterates

## Data Formats

### Edgelist Format
```
node1 node2
node1 node3
node2 node4
...
```
Whitespace-separated pairs (one edge per line).

### Adjacency CSV Format
Square matrix with 0/1 entries representing edges.

For datasets used in research (SNAP Facebook networks), see `data/README.md`.

## Dependencies

- **numpy**: Numerical computations
- **scipy**: ODE integration and scientific computing
- **networkx**: Graph operations and algorithms
- **matplotlib**: Plotting and visualization
- **pandas**: Data manipulation
- **streamlit**: Interactive web dashboard

## Implementation Notes

- **Fixed Point Computation**: The ODE fixed point is obtained via time-evolution to steady state (T=2000) using `scipy.integrate.solve_ivp` with RK45 method
- **NetShield Baseline**: A simplified variant using Perron-Frobenius eigenvector scoring as a practical proxy for the full NetShield algorithm
- **Numerical Stability**: The algorithm includes projection steps and zero-sum budget enforcement to maintain feasibility

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite the corresponding paper on bi-SIS competing contagion models and community targeting strategies.
