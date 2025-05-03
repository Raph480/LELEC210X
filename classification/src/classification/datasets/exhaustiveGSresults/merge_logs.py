import json
import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

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


def analyze_hyperparameter(logfile, hyperparameter_name):
    # Load all entries
    with open(logfile, 'r') as f:
        entries = [json.loads(line) for line in f if line.strip()]
    
    # Organize results by hyperparameter value
    grouped_results = defaultdict(lambda: {'kfold_acc': [], 'test_accuracy': []})

    for entry in entries:
        hyper_value = entry['hyperparameters'].get(hyperparameter_name)
        if hyper_value is None:
            continue  # Skip if this hyperparameter doesn't exist
        grouped_results[hyper_value]['kfold_acc'].append(entry['results']['kfold_acc'])
        grouped_results[hyper_value]['test_accuracy'].append(entry['results']['test_accuracy'])
    
    # Compute mean and std
    hyper_values = []
    kfold_means = []
    kfold_stds = []
    testacc_means = []
    testacc_stds = []

    for value, results in grouped_results.items():
        hyper_values.append(str(value))  # Make sure labels are strings
        kfold_acc = np.array(results['kfold_acc'])
        test_acc = np.array(results['test_accuracy'])

        kfold_means.append(np.mean(kfold_acc))
        kfold_stds.append(np.std(kfold_acc))

        testacc_means.append(np.mean(test_acc))
        testacc_stds.append(np.std(test_acc))
    
    # Sort by hyperparameter values (if possible)
    try:
        hyper_values_numeric = np.array([float(hv) for hv in hyper_values])
        sorted_indices = np.argsort(hyper_values_numeric)
        hyper_values = np.array(hyper_values)[sorted_indices]
        kfold_means = np.array(kfold_means)[sorted_indices]
        kfold_stds = np.array(kfold_stds)[sorted_indices]
        testacc_means = np.array(testacc_means)[sorted_indices]
        testacc_stds = np.array(testacc_stds)[sorted_indices]
    except:
        pass  # If conversion fails, just leave as is
    
    # Plotting
    x = np.arange(len(hyper_values))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot kfold_acc with error bars
    ax.errorbar(x, kfold_means, yerr=kfold_stds, label='kfold_accuracy (train)', fmt='-o', capsize=5, color='blue')

    # Plot test_accuracy with error bars
    ax.errorbar(x, testacc_means, yerr=testacc_stds, label='test_accuracy', fmt='-s', capsize=5, color='green')

    # Annotate values next to the points
    max = np.max(kfold_means)
    shift = max/8
    for xi, yi in zip(x, kfold_means):
        ax.annotate(f'{yi:.3f}', (xi-shift, yi), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='black')
    for xi, yi in zip(x, testacc_means):
        ax.annotate(f'{yi:.3f}', (xi-shift, yi), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8, color='black')

    ax.set_ylabel('Scores')
    ax.set_xlabel(hyperparameter_name)
    ax.set_title(f'Impact of {hyperparameter_name} on Results')
    ax.set_xticks(x)
    ax.set_xticklabels(hyper_values, rotation=45)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    #plt.savefig(f"individual_{hyperparameter_name}.svg")

    plt.tight_layout()
    plt.show()



# Example usage
if __name__ == "__main__":
    # List of files to process
    filepaths = ['grid_search_raw_20250428_183649.log','grid_search_sorted_20250427_170058.log'] 
    logs = load_logs(filepaths)
    merged_logs = merge_logs(logs)
    save_merged_logs('merged_refined_logs.log', merged_logs)

    #print(f"Merged, sorted, and rounded results saved to 'merged_logs.txt'.")
    #log_path= "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/exhaustiveGSresults/grid_search_sorted_20250427_002035.log"
    #log_path  = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/exhaustiveGSresults/grid_search_sorted_20250425_110409.log"
    
    #Refined grid search
    #log_path = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/exhaustiveGSresults/grid_search_sorted_20250427_170058.log"
    #analyze_hyperparameter(log_path, 'window_type')
