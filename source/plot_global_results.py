import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(path="../results/global_test_results.json"):
    with open(path, "r") as f:
        return json.load(f)


def plot_global_model_performance():
    data = load_results()

    hospitals = [h["hospital"].replace("_", " ").title() for h in data["hospitals"]]
    accuracy = [h["accuracy"] for h in data["hospitals"]]

    plt.figure(figsize=(7, 5))
    plt.bar(hospitals, accuracy)
    plt.ylabel("Accuracy")
    plt.title("Global Model Performance on Test Data")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("../data/plots/fig4_global_model_accuracy.png", dpi=300)
    plt.close()


def plot_federated_fairness_summary():
    data = load_results()

    labels = [
        "Macro Accuracy\n(Fairness)",
        "Micro Accuracy\n(Deployment)"
    ]
    values = [data["macro_accuracy"], data["micro_accuracy"]]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values)
    plt.ylabel("Accuracy")
    plt.title("Federated Model Performance Summary")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("../data/plots/fig5_federated_fairness_summary.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_global_model_performance()
    plot_federated_fairness_summary()