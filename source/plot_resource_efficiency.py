import os
import matplotlib.pyplot as plt

def plot_resource_efficiency(resource_metrics, save_dir="../data/plots"):
    """
    resource_metrics format:
    {
        'Hospital A': {'time': float, 'memory': float},
        'Hospital B': {'time': float, 'memory': float},
        'Hospital C': {'time': float, 'memory': float}
    }
    """

    os.makedirs(save_dir, exist_ok=True)

    hospitals = list(resource_metrics.keys())
    training_time = [resource_metrics[h]['time'] for h in hospitals]
    memory_usage = [resource_metrics[h]['memory'] for h in hospitals]

    # --- Academic-style plot config ---
    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("white")

    # -------- (a) Training Time --------
    axes[0].bar(hospitals, training_time)
    axes[0].set_title("(a) Training Time per Hospital")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_ylim(0, max(training_time) * 1.25)

    for i, v in enumerate(training_time):
        axes[0].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    # -------- (b) Memory Usage --------
    axes[1].bar(hospitals, memory_usage)
    axes[1].set_title("(b) Memory Usage per Hospital")
    axes[1].set_ylabel("Memory (MB)")
    axes[1].set_ylim(0, max(memory_usage) * 1.25)

    for i, v in enumerate(memory_usage):
        axes[1].text(i, v + 5, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "local_gnn_resource_efficiency.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Resource efficiency figure saved at: {save_path}")