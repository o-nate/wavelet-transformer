"""Generate phase difference key for XWT power spectrum plots"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_phase_difference_key(export: bool = False) -> None:
    """Generate plot"""
    # Set up the figure with equal aspect ratio to avoid distortions
    plt.figure(figsize=(10, 10))
    plt.axis("equal")  # Ensures the unit circle maintains its shape

    # Create a unit circle
    theta = np.linspace(0, 2 * np.pi, 100)  # Generate angles for a full circle
    x = np.cos(theta)  # x-coordinates
    y = np.sin(theta)  # y-coordinates
    plt.plot(x, y, "k")  # Plot the unit circle

    # Draw x-axis and y-axis with dashed lines
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")  # x-axis
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")  # y-axis

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)

    # * Label phase in radians
    plt.text(1.1, 0, "0", fontsize=18, color="k")
    plt.text(0, 1.1, r"$\pi/2$", fontsize=18, color="k")
    plt.text(-1.1, 0, r"$\pi$", fontsize=18, color="k")
    plt.text(0, -1.1, r"$-\pi/2$", fontsize=18, color="k")

    # * Label regions of lead/lag, phase/anti-phase relationships
    plt.text(0.5, 0.5, r"$x$ leads $y$", fontsize=14, color="k")
    plt.text(-0.75, 0.5, r"$y$ leads $x$ (anti-phase)", fontsize=14, color="k")
    plt.text(-0.75, -0.5, r"$x$ leads $y$ (anti-phase)", fontsize=14, color="k")
    plt.text(0.5, -0.5, r"$y$ leads $x$", fontsize=14, color="k")

    # * Add example of phase difference arrow
    angle = np.pi / 3
    x_arrow = np.cos(angle)
    y_arrow = np.sin(angle)

    plt.arrow(
        0,
        0,
        x_arrow - 0.07,
        y_arrow - 0.07,
        head_width=0.1,
        head_length=0.1,
        fc="k",
        ec="k",
        linestyle="-",
        linewidth=1,
        label=r"$\phi_{{xy}}$",
    )

    # * Strip axis labels
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # * Remove frame
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)

    # * Include legend
    plt.legend()

    # * Export as image
    if export:
        # * Get results directory
        parent_dir = Path(__file__).parents[1]
        export_file = parent_dir / "results" / "phase_diff_key.png"
        plt.savefig(export_file)

    # Show the plot
    plt.show()


def main() -> None:
    """Run script"""
    plot_phase_difference_key(export=True)


if __name__ == "__main__":
    main()
