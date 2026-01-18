import subprocess
import sys

def run(script):
    print(f"\n>>> Running {script}")
    subprocess.run([sys.executable, script], check=True)

def run_pipeline():
    print("\n===== PIPELINE STARTED =====\n")

    # STEP 1: Preprocessing 
    run("preprocessing.py")

    # STEP 2: Hospital Simulation (MISSING STEP - ADD THIS)
    # This creates the ../data/processed/hospital_A, B, and C folders
    run("hospital_simulation.py")

    # STEP 3: Graph Construction
    run("graph_construction.py")

    # STEP 4: Local GNN Training
    run("local_gnn_training.py")

    # STEP 5: Apply Local Differential Privacy
    run("local_ldp.py")

    # STEP 6: Federated Averaging
    run("fedavg_server.py")

    # STEP 7: Global Model Testing
    run("test_global_model.py")

    # STEP 8: Explainable AI & Inference Test
    run("inference_xai.py")
    
    print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====\n")

if __name__ == "__main__":
    run_pipeline()