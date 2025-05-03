import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import json
import numpy as np

# Chemins des fichiers
accuracies_logs_path = r"all_logs_with_aug.log"
power_logs_path = r"interpolated_powers.json"
output_path = r"complete_merged_data.json"

def merge_logs(accuracies_logs_path, power_logs_path, output_path):
    # Charger les données des fichiers
    with open(accuracies_logs_path, 'r') as merged_file:
        merged_logs = [json.loads(line.strip()) for line in merged_file]

    with open(power_logs_path, 'r') as power_file:
        power_logs = [json.loads(line.strip()) for line in power_file]

    # Fusionner les données
    for power_entry in power_logs:
        power_hyperparams = power_entry["hyperparameters"]
        power_result = power_entry["results"]

        for merged_entry in merged_logs:
            merged_hyperparams = merged_entry["hyperparameters"]

            # Vérifier si les hyperparamètres correspondent
            if all(merged_hyperparams.get(key) == value for key, value in power_hyperparams.items()):
                # Ajouter les nouvelles informations dans les résultats
                merged_entry["results"].update(power_result)

    # Sauvegarder les données fusionnées dans un nouveau fichier
    with open(output_path, 'w') as output_file:
        for entry in merged_logs:
            output_file.write(json.dumps(entry) + '\n')

    print(f"Fichier fusionné créé : {output_path}")

# Uncomment the following line to run the merge_logs function
merge_logs(accuracies_logs_path, power_logs_path, output_path)

# Chemin vers le fichier log
file_path = rf"{output_path}"

# Lire le fichier et extraire les données
data = []
with open(file_path, 'r') as file:
    for line in file:
        # Charger chaque ligne comme un objet JSON
        entry = json.loads(line.strip())
        # Combiner les hyperparamètres et les résultats en une seule liste plate
        combined = list(entry["hyperparameters"].values()) + list(entry["results"].values())
        data.append(combined)

# Convertir les données en tableau NumPy
data_array = np.array(data)



# Afficher la forme du tableau pour vérification
print("Shape of data array:", data_array.shape)
print("First 5 rows of data array:\n", data_array[:5])



# Convert data to a pandas DataFrame for easier handling
df = pd.DataFrame(data, columns=["bit_sensitivity", "melvec_height", "n_melvec", "window_type", "kfold_acc_mean", "test_accuracy_mean", "power"])
#df = pd.DataFrame(data, columns=["bit_sensitivity", "melvec_height", "n_melvec", "window_type", "kfold_acc_mean", "kfold_acc_variance", "test_accuracy_mean", "test_accuracy_variance", "power"])
df["power"] = 1 / df["power"]

# Create a scatter plot
fig = px.scatter(
    df,
    x="power",
    y="test_accuracy_mean",
    hover_data=["melvec_height", "n_melvec","bit_sensitivity","window_type"],  # Display char1 and char2 on hover
    title="Pareto Diagram",
    labels={"power": "Power efficiency [1/mW]", "kfold_acc_mean": "Accuracy (kfold)"}
)


# Show the plot
fig.show()