"""
Iterative Closest Point (ICP) Algorithm Visualization

ICP is used to align two point clouds by iteratively finding the best
rotation and translation that minimizes the distance between corresponding
points.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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

    Returns a list of dicts, one per iteration, each containing:
        source_before:  source positions entering this iteration
        matched_target: nearest-neighbour target points (the correspondences)
        source_after:   source positions after the rigid transform
        error:          mean point-to-point distance after transform
    """
    current = source.copy()
    history = []

    for _ in range(max_iterations):
        source_before = current.copy()

        indices = find_nearest_neighbors(current, target)
        matched_target = target[indices]

        R, t = compute_transformation(current, matched_target)
        current = (R @ current.T).T + t

        error = np.mean(np.linalg.norm(current - matched_target, axis=1))
        history.append({
            'source_before':  source_before,
            'matched_target': matched_target,
            'source_after':   current.copy(),
            'error':          error,
        })

        if len(history) > 1 and abs(history[-2]['error'] - error) < tolerance:
            break

    return history


def build_animation(source, target, history, interval=1000):
    """
    Build a FuncAnimation that walks through ICP iteration by iteration.

    Two frames per iteration:
      frame 0          — initial state, source at original position
      frame 2k-1 (k≥1) — Find correspondences: source_before + dashed lines to matched_target
      frame 2k   (k≥1) — Apply transform: source_after, lines cleared, error plot updated
    """
    errors = [h['error'] for h in history]
    all_iters = list(range(1, len(history) + 1))
    total_frames = 1 + 2 * len(history)

    fig, (ax_align, ax_conv) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ICP — Step-by-Step Animation", fontsize=14, fontweight="bold")

    # ── alignment panel ───────────────────────────────────────────────────
    ax_align.set_title("Point Cloud Alignment")
    ax_align.set_aspect("equal")
    ax_align.set_xlabel("x")
    ax_align.set_ylabel("y")
    ax_align.grid(True, alpha=0.3)

    ax_align.scatter(target[:, 0], target[:, 1],
                     c="steelblue", s=40, label="Target", zorder=3)

    (sc_source,) = ax_align.plot(
        source[:, 0], source[:, 1],
        "^", color="tomato", markersize=6, label="Source", zorder=4,
    )

    # correspondence lines (one per source point, initially invisible)
    corr_lines = [
        ax_align.plot([], [], "--", color="grey", linewidth=0.8, alpha=0.6, zorder=2)[0]
        for _ in range(len(source))
    ]

    ax_align.legend(fontsize=8, loc="upper right")

    # status text box
    status_text = ax_align.text(
        0.02, 0.97, "",
        transform=ax_align.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.9),
    )

    # ── convergence panel ─────────────────────────────────────────────────
    ax_conv.set_title("Convergence")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Mean distance error")
    ax_conv.set_yscale("log")
    ax_conv.grid(True, alpha=0.3)
    ax_conv.set_xlim(0.5, len(history) + 0.5)
    ax_conv.set_ylim(min(errors) * 0.5, max(errors) * 2)

    # faint full curve as guide
    ax_conv.plot(all_iters, errors, color="steelblue", linewidth=1,
                 alpha=0.2, zorder=1)

    (live_line,) = ax_conv.plot([], [], marker="o", color="steelblue",
                                linewidth=1.5, zorder=2)

    plt.tight_layout()

    # ── update function ───────────────────────────────────────────────────
    def update(frame):
        # clear correspondence lines
        for ln in corr_lines:
            ln.set_data([], [])

        if frame == 0:
            # initial state
            sc_source.set_data(source[:, 0], source[:, 1])
            live_line.set_data([], [])
            status_text.set_text("Initial state\nerror: —")
            return

        k = (frame + 1) // 2          # iteration index (1-based)
        phase = frame % 2              # 1 = correspondences, 0 = transform applied

        entry = history[k - 1]

        if phase == 1:
            # frame 2k-1: show correspondences
            pts = entry['source_before']
            sc_source.set_data(pts[:, 0], pts[:, 1])
            mpts = entry['matched_target']
            for i, ln in enumerate(corr_lines):
                ln.set_data([pts[i, 0], mpts[i, 0]], [pts[i, 1], mpts[i, 1]])
            status_text.set_text(
                f"Iter {k} — find correspondences\nerror: {entry['error']:.5f}"
            )
            # convergence plot shows previous iterations
            live_line.set_data(all_iters[:k - 1], errors[:k - 1])
        else:
            # frame 2k: apply transform
            pts = entry['source_after']
            sc_source.set_data(pts[:, 0], pts[:, 1])
            status_text.set_text(
                f"Iter {k} — apply transform\nerror: {entry['error']:.5f}"
            )
            live_line.set_data(all_iters[:k], errors[:k])

    anim = animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=interval,
        blit=False,
        repeat=True,
        repeat_delay=2000,
    )
    return anim


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
    errors = [h['error'] for h in history]

    # ── Static summary figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Iterative Closest Point (ICP)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.set_title("Alignment: Initial → Final")
    ax.scatter(target[:, 0], target[:, 1], c="steelblue", s=40, label="Target", zorder=3)
    ax.scatter(source[:, 0], source[:, 1], c="tomato", s=40, marker="^", label="Source (initial)", zorder=3)
    final_src = history[-1]['source_after']
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
    plt.savefig(os.path.join(SCRIPT_DIR, "icp_result.png"), dpi=120, bbox_inches="tight")

    # ── Animation figure ──────────────────────────────────────────────────
    anim = build_animation(source, target, history, interval=1000)

    gif_path = os.path.join(SCRIPT_DIR, "icp_animation.gif")
    try:
        anim.save(gif_path, writer="pillow", fps=1)
        print(f"Animation saved to {gif_path}")
    except Exception as exc:
        print(f"Could not save GIF ({exc}).")
        print("Install pillow with: pip install pillow")

    plt.show()
    print(f"ICP converged in {len(history)} iterations. Final error: {errors[-1]:.6f}")


if __name__ == "__main__":
    main()
