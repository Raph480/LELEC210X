import json
import glob
from collections import defaultdict
import numpy as np

def load_logs(filepaths):
    logs = []
    for path in filepaths:
        with open(path, 'r') as f:
            entries = [json.loads(line) for line in f if line.strip()]
            logs.append(entries)
    return logs

def merge_logs(logs):
    merged = defaultdict(lambda: {'kfold_acc': [], 'test_accuracy': []})

    for log in logs:
        for entry in log:
            # Create a unique key based on hyperparameters
            hyper_key = json.dumps(entry['hyperparameters'], sort_keys=True)
            merged[hyper_key]['kfold_acc'].append(entry['results']['kfold_acc'])
            merged[hyper_key]['test_accuracy'].append(entry['results']['test_accuracy'])
    
    # Now compute mean and variance
    output = []
    for hyper_key, results in merged.items():
        hyperparameters = json.loads(hyper_key)
        kfold_acc = np.array(results['kfold_acc'])
        test_accuracy = np.array(results['test_accuracy'])

        entry = {
            "hyperparameters": hyperparameters,
            "results": {
                "kfold_acc_mean": round(np.mean(kfold_acc), 5),
                "kfold_acc_variance": round(np.var(kfold_acc), 5),
                "test_accuracy_mean": round(np.mean(test_accuracy), 5),
                "test_accuracy_variance": round(np.var(test_accuracy), 5)
            }
        }
        output.append(entry)
    
    # Sort by kfold_acc_mean descending
    output.sort(key=lambda x: x['results']['kfold_acc_mean'], reverse=True)
    return output

def save_merged_logs(output_path, merged_logs):
    with open(output_path, 'w') as f:
        for entry in merged_logs:
            f.write(json.dumps(entry) + '\n')

# Example usage
if __name__ == "__main__":
    # List of files to process
    filepaths = ['grid_search_raw_20250425_110409.log','grid_search_raw_20250427_002035.log'] 
    logs = load_logs(filepaths)
    merged_logs = merge_logs(logs)
    save_merged_logs('merged_logs.log', merged_logs)

    print(f"Merged, sorted, and rounded results saved to 'merged_logs.txt'.")