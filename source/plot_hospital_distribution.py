import os
import matplotlib.pyplot as plt
import numpy as np


def plot_train_distribution(dist_dict, save_dir="../data/plots"):
    """
    Plot class distribution for hospital train splits.

    dist_dict format:
    {
        "Hospital A": {0: proportion, 1: proportion},
        "Hospital B": {0: proportion, 1: proportion},
        "Hospital C": {0: proportion, 1: proportion}
    }
    """

    os.makedirs(save_dir, exist_ok=True)

    hospitals = list(dist_dict.keys())
    class_1 = [dist_dict[h].get(1, 0.0) for h in hospitals]
    class_0 = [dist_dict[h].get(0, 0.0) for h in hospitals]

    x = np.arange(len(hospitals))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, class_1, width, label="Class 1 (CKD)")
    plt.bar(x + width / 2, class_0, width, label="Class 0 (Non-CKD)")

    plt.xticks(x, hospitals)
    plt.ylabel("Proportion")
    plt.ylim(0, 1.0)
    plt.title("Train Data Class Distribution Across Hospitals")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    save_path = os.path.join(
        save_dir, "hospital_simulation_train_distribution.png"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[INFO] Distribution plot saved at: {save_path}")