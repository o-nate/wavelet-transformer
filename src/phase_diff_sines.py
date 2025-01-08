"""Generate plot to demonstrate phase difference with sine waves"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)


def plot_phase_diff(export: bool = False) -> None:
    """Generate plot"""

    # * Create t-values for plotting the sine waves
    t = np.linspace(0, 2 * np.pi, 100)  # t-values from 0 to 2π (full sine wave period)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # * First subplot, waves in-phase
    axs[0].plot(t, np.sin(t), label="sin(t)", color="blue")
    axs[0].plot(t, np.sin(t) + 1, label="sin(t)+1", color="red")
    axs[0].set_title("In-Phase")
    axs[0].legend()

    # * Second subplot with waves in anti-phase (out of phase by π radians)
    axs[1].plot(t, np.sin(t), label="sin(t)", color="blue")
    axs[1].plot(t, np.sin(t + np.pi), label="sin(t + π)", color="red")
    axs[1].set_title("Anti-Phase")
    axs[1].legend()

    # * Third subplot, waves out of phase by π/2 radians
    axs[2].plot(t, np.sin(t), label="sin(t)", color="blue")
    axs[2].plot(t, np.sin(t + np.pi / 2), label="sin(t + π/2)", color="red")
    axs[2].set_title("Out of Phase by π/2")
    axs[2].legend()

    # Adjust spacing between subplots for better visibility
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Keep some space for the supertitle

    # * Export as image
    if export:
        # * Get results directory
        parent_dir = Path(__file__).parents[1]
        export_file = parent_dir / "results" / "phase_diff_sines.png"
        plt.savefig(export_file)

    # Show the plot
    plt.show()


def main() -> None:
    """Run script"""
    plot_phase_diff(export=True)


if __name__ == "__main__":
    main()
