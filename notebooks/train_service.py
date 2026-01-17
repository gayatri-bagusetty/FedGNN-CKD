import os
import papermill as pm
import time

BASE_DIR = os.path.dirname(__file__)
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")

os.makedirs(RUNTIME_DIR, exist_ok=True)

NOTEBOOKS = {
    "arff": "convert_uci_arff_to_csv.ipynb",
    "preprocessing": "preprocessing.ipynb",
    "hospital": "hospital_simulation.ipynb",
    "graph": "graph_construction.ipynb",
    "local_train": "local_gnn_training.ipynb",
    "ldp": "local_ldp.ipynb",
    "fedavg": "fedavg_server.ipynb",
}


def run_notebook(nb, out):
    pm.execute_notebook(
        os.path.join(BASE_DIR, nb),
        os.path.join(RUNTIME_DIR, out),
        log_output=True
    )


def run_full_training(uploaded_file, update_status):

    # ---------------- STEP 1 ----------------
    update_status(1, "processing", 5)

    if uploaded_file.name.endswith(".arff"):
        run_notebook(NOTEBOOKS["arff"], "01_arff.ipynb")

    run_notebook(NOTEBOOKS["preprocessing"], "02_preprocessing.ipynb")
    update_status(1, "done", 20)
    time.sleep(0.5)

    # ---------------- STEP 2 ----------------
    update_status(2, "processing", 30)
    run_notebook(NOTEBOOKS["hospital"], "03_hospital.ipynb")
    run_notebook(NOTEBOOKS["graph"], "04_graph.ipynb")
    update_status(2, "done", 45)
    time.sleep(0.5)

    # ---------------- STEP 3 ----------------
    update_status(3, "processing", 60)
    run_notebook(NOTEBOOKS["local_train"], "05_local_train.ipynb")
    update_status(3, "done", 70)
    time.sleep(0.5)

    # ---------------- STEP 4 ----------------
    update_status(4, "processing", 80)
    run_notebook(NOTEBOOKS["ldp"], "06_ldp.ipynb")
    update_status(4, "done", 90)
    time.sleep(0.5)

    # ---------------- STEP 5 ----------------
    update_status(5, "processing", 95)
    run_notebook(NOTEBOOKS["fedavg"], "07_fedavg.ipynb")
    update_status(5, "sent", 100)