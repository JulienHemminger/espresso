import os
from datetime import datetime


def save_plot_with_timestamp(fig, base_directory="/home/main/"):
    """
    Saves the provided figure as a PNG with a timestamped filename.
    Ensures the directory exists before saving.
    """
    # Create the timestamp string
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"lerp2d_{timestamp}.png"

    # Ensure the directory exists (optional, but good practice)
    if not os.path.exists(base_directory):
        print(f"Directory {base_directory} not found. Saving to current directory.")
        full_path = filename
    else:
        full_path = os.path.join(base_directory, filename)

    # Save the figure
    # bbox_inches='tight' is recommended to prevent clipping of labels/legends
    fig.savefig(full_path, bbox_inches="tight", dpi=300)
    print(f"Figure successfully saved to: {full_path}")
