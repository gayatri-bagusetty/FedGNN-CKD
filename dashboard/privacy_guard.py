def sanitize_metrics(metrics):
    return {
        "noised_accuracy": metrics["noised_accuracy"],
        "epsilon": metrics["epsilon"],
        "time_update": metrics["time_update"],
        "time_analysis": metrics["time_analysis"]
    }