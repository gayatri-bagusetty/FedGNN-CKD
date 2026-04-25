import subprocess
import sys

def run(script):
    print(f"\n>>> Running {script}")
    subprocess.run([sys.executable, script], check=True)

def run_pipeline():
    print("\n===== SERVER PIPELINE STARTED =====\n")

    # STEP 1: Federated Averaging
    run("fedavg_server.py")

    # STEP 2: Global Model Testing
    run("test_global_model.py")
    
    print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====\n")

if __name__ == "__main__":
    run_pipeline()