import subprocess
import sys

def run_script(script_name):
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

if __name__ == "__main__":
    print("Running data exploration...")
    run_script('data_exploration.py')
    
    print("Running preprocessing...")
    run_script('preprocessing.py')
    
    print("Running model training...")
    run_script('model_training.py')
    
    print("Running prediction example...")
    run_script('predict.py')
    
    print("Project completed!")