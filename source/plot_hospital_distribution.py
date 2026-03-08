import os
import matplotlib.pyplot as plt
import numpy as np


def get_distribution(df):
    dist = df["classification"].value_counts(normalize=True).to_dict()
    return dist.get(0, 0), dist.get(1, 0)


def plot_smote_comparison(train_before, train_after, save_dir="../data/plots"):
    os.makedirs(save_dir, exist_ok=True)
    hospitals = ["Hospital A", "Hospital B", "Hospital C"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    width = 0.35
    x = np.arange(len(hospitals))

    datasets = [
        ("Train Before SMOTE", train_before),
        ("Train After SMOTE", train_after)
    ]

    for i, (title, data_list) in enumerate(datasets):
        class0 = []
        class1 = []
        for df in data_list:
            c0, c1 = get_distribution(df)
            class0.append(c0)
            class1.append(c1)
        ax = axes[i]
        ax.bar(x - width/2, class0, width, label="Class 0 (Non-CKD)")
        ax.bar(x + width/2, class1, width, label="Class 1 (CKD)")
        ax.set_xticks(x)
        ax.set_xticklabels(hospitals)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    save_path = os.path.join(save_dir, "smote_train_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Plot saved at: {save_path}")