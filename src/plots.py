import io
import numpy as np
import matplotlib.pyplot as plt


def plot_market_share_vs_eps(results, title="Market Share vs epsilon"):
    eps = np.array([r[0] for r in results])
    avgX = np.array([r[1] for r in results])
    avgY = np.array([r[2] for r in results])

    fig1 = plt.figure()
    plt.xscale("log")
    plt.plot(eps, avgY, marker="o")
    plt.xlabel("epsilon")
    plt.ylabel("AvgY (Product 2)")
    plt.title(title + " — Product 2")
    plt.grid(True, which="both", ls=":")

    fig2 = plt.figure()
    plt.xscale("log")
    plt.plot(eps, avgX, marker="o")
    plt.xlabel("epsilon")
    plt.ylabel("AvgX (Product 1)")
    plt.title(title + " — Product 1")
    plt.grid(True, which="both", ls=":")

    return fig1, fig2


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()
