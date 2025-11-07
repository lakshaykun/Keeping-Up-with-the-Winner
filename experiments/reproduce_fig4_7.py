import numpy as np
import networkx as nx
from pathlib import Path

from src.run_experiments import run_epsilon_sweep
from src.plots import plot_market_share_vs_eps


def main():
    outdir = Path(__file__).resolve().parent / 'outputs'
    outdir.mkdir(parents=True, exist_ok=True)

    # Example: use Karate club as small stand-in for quick check
    # For full replication, load SNAP Facebook graph (4039 nodes) into A
    G = nx.karate_club_graph()
    A = nx.to_numpy_array(G, dtype=float)

    tau1, tau2 = 0.8, 0.05  # as in the paper for 4k case
    eps_list = np.logspace(-8, -1, 15)

    mu_c, u_c, x_star, results = run_epsilon_sweep(A, tau1, tau2, eps_list)
    fig1, fig2 = plot_market_share_vs_eps(results, title="Our Approach (Demo)")
    fig1.savefig(outdir / 'fig_product2.png', dpi=200)
    fig2.savefig(outdir / 'fig_product1.png', dpi=200)


if __name__ == '__main__':
    main()
