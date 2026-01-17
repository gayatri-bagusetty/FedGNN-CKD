import subprocess
import sys

def run(script):
    print(f"\n>>> Running {script}")
    subprocess.run([sys.executable, script], check=True)

def run_pipeline():
    print("\n===== PIPELINE STARTED =====\n")

    # STEP 1: Preprocessing (CSV already given)
    run("preprocessing.py")

    # STEP 2: Graph Construction
    run("graph_construction.py")

    # STEP 3: Local GNN Training
    run("local_gnn_training.py")

    # STEP 4: Apply Local Differential Privacy
    run("local_ldp.py")

    # STEP 5: Federated Averaging
    run("fedavg_server.py")

    # STEP 6: Global Model Testing
    run("test_global_model.py")

    print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====\n")

if __name__ == "__main__":
    run_pipeline()