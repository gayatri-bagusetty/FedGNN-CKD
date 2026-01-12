import time
from status_tracker import update_step

def run_pipeline():
    start_time = time.time()

    update_step(1, "Running")
    time.sleep(1)
    update_step(1, "Completed")

    update_step(2, "Running")
    time.sleep(1)
    update_step(2, "Completed")

    update_step(3, "Protected")
    time.sleep(1)

    update_step(4, "Protected")
    time.sleep(1)

    update_step(5, "Running")
    time.sleep(1)
    update_step(5, "Completed")

    update_step(6, "Completed")

    return {
        "noised_accuracy": 98.7,
        "epsilon": 2.0,
        "time_update": "2 mins ago",
        "time_analysis": f"{int(time.time() - start_time)} secs",
        "diagnosis": "High Risk of CKD"
    }