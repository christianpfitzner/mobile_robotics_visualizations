"""
Least Squares Fitting Visualizations (1-D, 2-D, 3-D)

Demonstrates ordinary least-squares regression for:
  • 1-D: polynomial curve fitting to noisy scalar data
  • 2-D: plane fitting to noisy 2-D surface data
  • 3-D: hyperplane fitting to noisy 3-D data (visualised as a 3-D scatter
        with the fitted plane)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3-D projection


# ─────────────────────────────────────────────────────────────────────────────
# 1-D Polynomial Least Squares
# ─────────────────────────────────────────────────────────────────────────────

def least_squares_1d(x, y, degree=3):
    """
    Fit a polynomial of the given *degree* to (x, y) using the normal
    equations: θ = (XᵀX)⁻¹ Xᵀy.

    Returns the coefficient vector (highest degree first) and the design
    matrix evaluated over a fine grid for plotting.
    """
    # Vandermonde design matrix
    X = np.vander(x, N=degree + 1, increasing=False)
    # Normal equations
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs


def plot_1d(ax, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-2, 2, 40)
    y_true = 0.5 * x ** 3 - x + 1.0
    y_noisy = y_true + rng.normal(0, 0.4, len(x))

    for degree, color in [(1, "tomato"), (3, "steelblue"), (5, "green")]:
        coeffs = least_squares_1d(x, y_noisy, degree=degree)
        y_fit = np.polyval(coeffs, x)
        ax.plot(x, y_fit, color=color, linewidth=1.8, label=f"Degree {degree}")

    ax.scatter(x, y_noisy, c="gray", s=20, alpha=0.6, label="Noisy data", zorder=3)
    ax.plot(x, y_true, "k--", linewidth=1, alpha=0.5, label="True function")
    ax.set_title("1-D Polynomial Least Squares")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# 2-D Plane / Bilinear Least Squares
# ─────────────────────────────────────────────────────────────────────────────

def least_squares_2d(x, y, z):
    """
    Fit z = a·x + b·y + c to scattered (x, y, z) data.

    Returns (a, b, c).
    """
    A = np.column_stack([x, y, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs  # [a, b, c]


def plot_2d(ax, seed=1):
    rng = np.random.default_rng(seed)
    n = 120
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)
    z_true = 1.5 * x - 0.8 * y + 2.0
    z_noisy = z_true + rng.normal(0, 0.5, n)

    a, b, c = least_squares_2d(x, y, z_noisy)

    # Plot scattered points and the fitted plane
    sc = ax.scatter(x, y, c=z_noisy, cmap="viridis", s=20, alpha=0.7, label="Noisy data")
    plt.colorbar(sc, ax=ax, pad=0.02, label="z value")

    xx, yy = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    zz = a * xx + b * yy + c
    ax.contourf(xx, yy, zz, levels=15, cmap="coolwarm", alpha=0.35)
    ax.contour(xx, yy, zz, levels=10, colors="k", linewidths=0.5, alpha=0.4)

    ax.set_title("2-D Plane Least Squares\n"
                 f"Fitted: z = {a:.2f}x + {b:.2f}y + {c:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# 3-D Hyperplane Least Squares
# ─────────────────────────────────────────────────────────────────────────────

def least_squares_3d(x, y, z, w):
    """
    Fit w = a·x + b·y + c·z + d to 3-D data.

    Returns (a, b, c, d).
    """
    A = np.column_stack([x, y, z, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, w, rcond=None)
    return coeffs


def plot_3d(ax, seed=2):
    rng = np.random.default_rng(seed)
    n = 100
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)
    z = rng.uniform(-2, 2, n)
    w_true = 1.0 * x - 0.5 * y + 0.8 * z + 1.5
    w_noisy = w_true + rng.normal(0, 0.4, n)

    a, b, c, d = least_squares_3d(x, y, z, w_noisy)
    w_pred = a * x + b * y + c * z + d
    residuals = w_noisy - w_pred

    sc = ax.scatter(x, y, z, c=residuals, cmap="RdBu", s=25, alpha=0.8)
    plt.colorbar(sc, ax=ax, pad=0.1, label="Residual")

    ax.set_title("3-D Hyperplane Least Squares\n(colour = residual)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    rmse = np.sqrt(np.mean(residuals ** 2))
    ax.text2D(0.02, 0.95, f"RMSE = {rmse:.4f}", transform=ax.transAxes, fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("Least Squares Fitting (1-D, 2-D, 3-D)", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection="3d")

    plot_1d(ax1)
    plot_2d(ax2)
    plot_3d(ax3)

    plt.tight_layout()
    plt.savefig("least_squares_result.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Least squares visualizations generated.")


if __name__ == "__main__":
    main()
