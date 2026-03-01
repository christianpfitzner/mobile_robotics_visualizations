"""
Random Sample Consensus (RANSAC) Visualization

RANSAC is a robust estimation method that fits a model to data containing
a significant number of outliers.  Here it is used for 2D line fitting.
"""

import numpy as np
import matplotlib.pyplot as plt


def fit_line(p1, p2):
    """Return (a, b, c) for the line ax + by + c = 0 through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    norm = np.sqrt(a ** 2 + b ** 2)
    return a / norm, b / norm, c / norm


def point_to_line_dist(points, a, b, c):
    """Return the perpendicular distance from each point to ax+by+c=0."""
    return np.abs(a * points[:, 0] + b * points[:, 1] + c)


def ransac_line(points, n_iterations=200, threshold=0.1, min_inliers=10, seed=0):
    """
    Fit a line to *points* using RANSAC.

    Returns
    -------
    best_model   : (a, b, c) of the best line found
    best_inliers : boolean mask of inlier points
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    best_model = None

    for _ in range(n_iterations):
        idx = rng.choice(n, 2, replace=False)
        a, b, c = fit_line(points[idx[0]], points[idx[1]])
        dists = point_to_line_dist(points, a, b, c)
        inliers = dists < threshold

        if inliers.sum() > best_count and inliers.sum() >= min_inliers:
            best_count = inliers.sum()
            best_inliers = inliers
            best_model = (a, b, c)

    # Refit using all inliers for a better estimate
    if best_model is not None and best_inliers.sum() >= 2:
        inlier_pts = points[best_inliers]
        x = inlier_pts[:, 0]
        y = inlier_pts[:, 1]
        # Ordinary least-squares refit (vertical residuals)
        coeffs = np.polyfit(x, y, 1)
        # Convert y = mx + q  →  -mx + y - q = 0  (normalised)
        m, q = coeffs
        a, b, c = fit_line(
            np.array([x.min(), m * x.min() + q]),
            np.array([x.max(), m * x.max() + q]),
        )
        dists = point_to_line_dist(points, a, b, c)
        best_inliers = dists < threshold
        best_model = (a, b, c)

    return best_model, best_inliers


def generate_data(n_inliers=80, n_outliers=40, noise=0.05, seed=7):
    """Generate data: a noisy line plus random outliers."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n_inliers)
    y = 0.6 * x + 1.0 + rng.normal(0, noise, n_inliers)
    inlier_pts = np.column_stack([x, y])

    outlier_x = rng.uniform(-2.5, 2.5, n_outliers)
    outlier_y = rng.uniform(-1.5, 3.5, n_outliers)
    outlier_pts = np.column_stack([outlier_x, outlier_y])

    points = np.vstack([inlier_pts, outlier_pts])
    true_labels = np.array([True] * n_inliers + [False] * n_outliers)
    return points, true_labels


def main():
    points, true_labels = generate_data()

    model, inliers = ransac_line(points, n_iterations=300, threshold=0.15)

    # ── Build a plotting x-range and draw the fitted line ─────────────────
    x_plot = np.linspace(points[:, 0].min() - 0.1, points[:, 0].max() + 0.1, 200)
    a, b, c = model  # ax + by + c = 0  →  y = -(ax+c)/b
    if abs(b) > 1e-10:
        y_plot = -(a * x_plot + c) / b
    else:
        y_plot = np.zeros_like(x_plot)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Random Sample Consensus (RANSAC) – Line Fitting", fontsize=14, fontweight="bold")

    for ax, title, mask_inlier in [
        (axes[0], "Raw data (true inliers / outliers)", true_labels),
        (axes[1], "RANSAC result (detected inliers / outliers)", inliers),
    ]:
        ax.set_title(title)
        ax.scatter(
            points[mask_inlier, 0], points[mask_inlier, 1],
            c="steelblue", s=30, label="Inliers", zorder=3,
        )
        ax.scatter(
            points[~mask_inlier, 0], points[~mask_inlier, 1],
            c="tomato", s=30, marker="x", label="Outliers", zorder=3,
        )
        ax.plot(x_plot, y_plot, "k--", linewidth=1.5, label="Fitted line")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Summary stats
    n_correct = np.sum(inliers == true_labels)
    accuracy = 100 * n_correct / len(points)
    fig.text(
        0.5, 0.01,
        f"Detected inliers: {inliers.sum()}  |  "
        f"Classification accuracy: {accuracy:.1f}%",
        ha="center", fontsize=10,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("ransac_result.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"RANSAC: {inliers.sum()} inliers detected. Classification accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
