"""
Iterative Closest Point (ICP) Algorithm Visualization

ICP is used to align two point clouds by iteratively finding the best
rotation and translation that minimizes the distance between corresponding
points.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def rotation_matrix(angle):
    """Return a 2D rotation matrix for the given angle (in radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def generate_point_cloud(n_points=30, noise=0.05, seed=42):
    """Generate a simple 2D point cloud shaped like an arc."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, np.pi, n_points)
    x = np.cos(angles) + rng.normal(0, noise, n_points)
    y = np.sin(angles) + rng.normal(0, noise, n_points)
    return np.column_stack([x, y])


def find_nearest_neighbors(source, target):
    """Find the nearest neighbor in target for each point in source."""
    indices = np.array([
        np.argmin(np.linalg.norm(target - s, axis=1))
        for s in source
    ])
    return indices


def compute_transformation(source, target_matched):
    """Compute the optimal rotation and translation using SVD."""
    src_mean = source.mean(axis=0)
    tgt_mean = target_matched.mean(axis=0)

    src_centered = source - src_mean
    tgt_centered = target_matched - tgt_mean

    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_mean - R @ src_mean
    return R, t


def icp(source, target, max_iterations=30, tolerance=1e-5):
    """
    Run the ICP algorithm.

    Returns a list of (transformed_source, error) tuples for each iteration.
    """
    current = source.copy()
    history = [(current.copy(), float("inf"))]

    for _ in range(max_iterations):
        indices = find_nearest_neighbors(current, target)
        matched_target = target[indices]

        R, t = compute_transformation(current, matched_target)
        current = (R @ current.T).T + t

        error = np.mean(np.linalg.norm(current - matched_target, axis=1))
        history.append((current.copy(), error))

        if len(history) > 2 and abs(history[-2][1] - error) < tolerance:
            break

    return history


def main():
    np.random.seed(0)

    # Generate the target point cloud
    target = generate_point_cloud(n_points=40, noise=0.03, seed=1)

    # Generate source as a transformed version of target
    angle = np.radians(35)
    translation = np.array([0.8, -0.4])
    R_true = rotation_matrix(angle)
    source = (R_true @ target.T).T + translation + np.random.normal(0, 0.02, target.shape)

    # Run ICP
    history = icp(source, target, max_iterations=50)
    errors = [h[1] for h in history if h[1] != float("inf")]

    # ── Static summary figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Iterative Closest Point (ICP)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.set_title("Alignment: Initial → Final")
    ax.scatter(target[:, 0], target[:, 1], c="steelblue", s=40, label="Target", zorder=3)
    ax.scatter(source[:, 0], source[:, 1], c="tomato", s=40, marker="^", label="Source (initial)", zorder=3)
    final_src = history[-1][0]
    ax.scatter(final_src[:, 0], final_src[:, 1], c="green", s=40, marker="s", label="Source (aligned)", zorder=3)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.set_title("Convergence")
    ax2.plot(range(1, len(errors) + 1), errors, marker="o", color="steelblue", linewidth=1.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Mean distance error")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("icp_result.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"ICP converged in {len(history) - 1} iterations. Final error: {errors[-1]:.6f}")


if __name__ == "__main__":
    main()
