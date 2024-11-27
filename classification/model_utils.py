import matplotlib.pyplot as plt
import numpy as np
import os
"Machine learning tools"
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time

import pickle
from classification.datasets import Dataset
from classification.utils.audio_student import AudioUtil, Feature_vector_DS
from classification.utils.plots import (
    plot_decision_boundaries,
    plot_specgram,
    show_confusion_matrix,
)
from classification.utils.utils import accuracy
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold



### 0. UTILS FUNCTIONS
###--------------------------------------------

#Dataset preparation
def get_dataset_matrix(normalization=False, verbose=False):

    dataset = Dataset()
    classnames = dataset.list_classes()

    fm_dir = "data/feature_matrices/"  # where to save the features matrices
    model_dir = "data/models/"  # where to save the models
    os.makedirs(fm_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Hyperparameters to tune in 2nd part of project?
    if normalization:
        myds = Feature_vector_DS(dataset, Nft=512, nmel=20, duration=950, shift_pct=0.2, normalize=True)
    else:
        myds = Feature_vector_DS(dataset, Nft=512, nmel=20, duration=950, shift_pct=0.2)
    
    "Some attributes..."
    myds.nmel
    myds.duration
    myds.shift_pct
    myds.sr
    myds.data_aug
    myds.ncol

    idx = 0
    #myds.display(["birds", idx])

    #Matrixed dataset
    "Random split of 70:30 between training and test"
    train_pct = 0.7

    featveclen = len(myds["fire", 0])  # number of items in a feature vector
    nitems = len(myds)  # number of sounds in the dataset
    naudio = dataset.naudio  # number of audio files in each class
    nclass = dataset.nclass  # number of classes
    nlearn = round(naudio * train_pct)  # number of sounds among naudio for training

    data_aug_factor = 1
    class_ids_aug = np.repeat(classnames, naudio * data_aug_factor)
    

    "Compute the matrixed dataset, this takes some seconds, but you can then reload it by commenting this loop and decommenting the np.load below"
    X = np.zeros((data_aug_factor * nclass * naudio, featveclen))
    for s in range(data_aug_factor):
        for class_idx, classname in enumerate(classnames):
            for idx in range(naudio):
                featvec = myds[classname, idx]
                X[s * nclass * naudio + class_idx * naudio + idx, :] = featvec
    #np.save(fm_dir + "feature_matrix_2D.npy", X)
    
    #X = np.load(fm_dir+"feature_matrix_2D.npy")


    "Labels"
    y = class_ids_aug.copy()
    if verbose:
        print(f"Shape of the feature matrix : {X.shape}")
        print(f"Number of labels : {len(y)}")
    
    return X, y, classnames 

#K Fold 
def perform_Kfold(X_train, y_train, model, kf, normalization, reduction, PCA_components=0, verbose=True, dB_mismatch=0, FV_normalization=False):
    """
    #Write 
    """
    if (verbose):
        print(f"\nNormalization: {normalization}")
        print(f"Reduction: {reduction}")

        if reduction:
            print(f"PCA components: {PCA_components}")

    # K-Fold cross-validation to asses the accuracy of the model
    accuracy_train = []
    accuracy_val = []

    for train_index, test_index in kf.split(X_train):
        X_train_kf, X_val = X_train[train_index], X_train[test_index]
        y_train_kf, y_val = y_train[train_index], y_train[test_index]

        if dB_mismatch:
            X_val = X_val * 10 ** (-dB_mismatch / 20)

        if FV_normalization:
            X_train_kf = X_train_kf / np.linalg.norm(X_train_kf, axis=1, keepdims=True)
            X_val = X_val / np.linalg.norm(X_val, axis=1, keepdims=True)
        
        if normalization:
            scaler = StandardScaler()
            X_train_kf = scaler.fit_transform(X_train_kf)
            X_val = scaler.transform(X_val)

        if reduction:
            pca = PCA(n_components=PCA_components)
            X_train_kf = pca.fit_transform(X_train_kf)
            X_val = pca.transform(X_val)

        model.fit(X_train_kf, y_train_kf)
        accuracy_train.append(accuracy(y_train_kf, model.predict(X_train_kf)))
        accuracy_val.append(accuracy(y_val, model.predict(X_val)))

    if (verbose):
        print("\nK-Fold results:\n---------------------")
        if (dB_mismatch):
            print("DB mismatch: ", dB_mismatch)
        print(f"K-Fold Mean accuracy on train set: {np.mean(accuracy_train):.5f}")
        print(f"K-Fold Mean accuracy on validation set: {np.mean(accuracy_val):.5f}")

        #Standard deviation
        print(f"K-Fold Standard deviation on train set: {np.std(accuracy_train):.5f}")
        print(f"K-Fold Standard deviation on validation set: {np.std(accuracy_val):.5f}")

    return np.mean(accuracy_train), np.std(accuracy_train), np.mean(accuracy_val), np.std(accuracy_val)

def analyse_nb_splits_Kfold(X_train, y_train, model, normalization, reduction, PCA_components, verbose):
    
    #From previous best hyperparameters
    params = {'bootstrap': True, 'max_depth': 4, 'min_samples_leaf': 2,
            'min_samples_split': 4, 'n_estimators': 150}

    model_rf = RandomForestClassifier(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print("Kfold with 5 splits")
    accuracy_train_tab = []
    accuracy_val_tab = []
    for i in range(10):
        accuracy_train, _, accuracy_val, _ = perform_Kfold(X_train, y_train, model_rf, kf, normalization, reduction, PCA_components=0,verbose=False)
        accuracy_train_tab.append(accuracy_train)
        accuracy_val_tab.append(accuracy_val)

    print(f"Mean accuracy on train set for 5 splits: {np.mean(accuracy_train_tab):.5f}")
    print(f"Mean accuracy on validation set for 5 splits: {np.mean(accuracy_val_tab):.5f}")
    print("accucary std on train set for 5 splits: ", np.std(accuracy_train_tab))
    print("accucary std on validation set for 5 splits: ", np.std(accuracy_val_tab))

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    print("Kfold with 10 splits")
    accuracy_train_tab = []
    accuracy_val_tab = []

    for i in range(10):
        accuracy_train, _, accuracy_val, _ = perform_Kfold(X_train, y_train, model_rf, kf, normalization, reduction, PCA_components=0,verbose=False)
        accuracy_train_tab.append(accuracy_train)
        accuracy_val_tab.append(accuracy_val)

    print(f"Mean accuracy on train set for 10 splits: {np.mean(accuracy_train_tab):.5f}")
    print(f"Mean accuracy on validation set for 10 splits: {np.mean(accuracy_val_tab):.5f}")
    print("accucary std on train set for 10 splits: ", np.std(accuracy_train_tab))
    print("accucary std on validation set for 10 splits: ", np.std(accuracy_val_tab))

def boxplot_models(mean1, mean2, mean3, std1, std2, std3, name1, name2, name3,title):
    # Calculate boxplot statistics for each model
    def calculate_box(mean, std):
        q1 = mean - 0.675 * std  # Approximate 25th percentile
        q3 = mean + 0.675 * std  # Approximate 75th percentile
        whisker_low = mean - 1.5 * std  # Lower whisker
        whisker_high = mean + 1.5 * std  # Upper whisker
        return {
            'whislo': whisker_low,  # Bottom whisker
            'q1': q1,               # 25th percentile
            'med': mean,            # Median
            'q3': q3,               # 75th percentile
            'whishi': whisker_high  # Top whisker
        }
    
    # Box plot data
    boxes = [
        calculate_box(mean1, std1),
        calculate_box(mean2, std2),
        calculate_box(mean3, std3)
    ]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Adjust positions of boxplots with spacings
    positions = [1, 3, 5]  # Manually spacing the box plots
    ax.bxp(boxes, positions=positions, showfliers=False, patch_artist=True)
    
    # Annotate mean and std
    for pos, mean, std, name in zip(positions, [mean1, mean2, mean3], [std1, std2, std3], [name1, name2, name3]):
        ax.text(pos, mean + 1.5 * std, f"Mean: {mean:.2f}\nStd: {std:.2f}", 
                ha='center', va='center', fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.8))

    # Customize the plot
    ax.set_title("Boxplots of Models with Annotations")
    ax.set_xticks(positions)
    ax.set_xticklabels([name1, name2, name3])
    ax.set_ylabel("Accuracy")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(title)
    
    # Show plot
    plt.tight_layout()
    plt.show()


def perform_KFold_eval_n_PC(X_train,y_train,model,kf, PC_start, PC_end, mean_witout_PCA,verbose):
    #Store values to plot them
    accuracy_mean = []
    accuracy_std = []

    for i in range(PC_start, PC_end):
        print(f"PCA components: {i}")
        _,_, mean, std = perform_Kfold(X_train, y_train, model, kf, False, True, PCA_components=i, verbose=False, dB_mismatch=0, FV_normalization=True)

        accuracy_mean.append(mean)
        accuracy_std.append(std)
        

    plot_PC(accuracy_mean, accuracy_std, PC_start, PC_end, mean_witout_PCA)

import matplotlib.pyplot as plt

def plot_PC(accuracy_mean, accuracy_std, PC_start, PC_end, mean_witout_PCA):
    # Ensure accuracy_mean and accuracy_std are lists of the correct length
    assert len(accuracy_mean) == len(accuracy_std), "accuracy_mean and accuracy_std must have the same length"

    # Plot the mean accuracy
    plt.plot(range(PC_start, PC_end), accuracy_mean, label="Mean Accuracy", color="blue")
    
    # Fill the area between mean - std and mean + std (shading the std range)
    plt.fill_between(range(PC_start, PC_end), 
                     [m - s/2 for m, s in zip(accuracy_mean, accuracy_std)], 
                     [m + s/2 for m, s in zip(accuracy_mean, accuracy_std)], 
                     color='blue', alpha=0.3, label="Standard Deviation")

    #Plot a horizontal line for the mean accuracy without PCA
    plt.axhline(y=mean_witout_PCA, color='r', linestyle='--', label=f"Mean accuracy without PCA: {mean_witout_PCA:.3f}")

    # Add labels and title
    plt.xlabel("Number of PCA components")
    plt.ylabel("Mean accuracy")
    plt.title("Mean KFold accuracy for different number of PCA components")
    plt.legend()


    
    # Show the plot
    plt.show()


# General grid search to find hyper parameters
def perform_grid_search(model, param_grid, kf, X, y, model_name):
    start_time = time.time()
    print("\n-----------------------------------------------")
    print("Start grid search for", model_name, "model...\n")
    
    score = make_scorer(lambda y_true, y_false: np.abs(accuracy(y_true, y_false)), greater_is_better=True)


    grid = GridSearchCV(model, param_grid, cv=kf, scoring=score, n_jobs=-1, verbose=1, return_train_score=True)
    grid.fit(X, np.ravel(y))

    stop_time = time.time()
    print(f'Finished : execution time for {model_name} model: {stop_time-start_time:.2f} seconds')
    
    return grid

def print_result(model, model_name, accuracy_train, accuracy_test,grid=True):
    print(f"Best parameters for {model_name} model:")
    if (grid):
        for key, value in model.best_params_.items():
            print(f"\t{key}: {value}")
    print(f"Results for {model_name} model:")
    print(f"\Accuracy on train set: {accuracy_train:.5f}")
    print(f"\Accuracy on test set: {accuracy_test:.5f}")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
def plot_grid_search_heatmap(X_train, y_train, param_grid):
    """
    Perform grid search and plot the results as a heatmap.
    """
    # Extract hyperparameter values from the param_grid
    n_estimators_values = param_grid['min_samples_leaf']
    max_depth_values = param_grid['min_samples_split']
    
    # Create a matrix to store the mean accuracy scores
    mean_accuracies = np.zeros((len(max_depth_values), len(n_estimators_values)))

    # Loop through all combinations of n_estimators and max_depth
    for i, n_estimators in enumerate(n_estimators_values):
        for j, max_depth in enumerate(max_depth_values):
            # Create a RandomForestClassifier with the current combination of parameters
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            
            # Perform cross-validation and compute the mean accuracy
            accuracies = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            mean_accuracies[j, i] = accuracies.mean()  # Store mean accuracy for this combination

    # Plot the heatmap of the results
    plt.figure(figsize=(8, 6))
    plt.imshow(mean_accuracies, interpolation='nearest', cmap = 'jet',aspect='auto')
    plt.colorbar(label='Mean Accuracy Score')

    # Set axis labels and ticks
    plt.xticks(np.arange(len(n_estimators_values)), n_estimators_values, rotation=45)
    plt.yticks(np.arange(len(max_depth_values)), max_depth_values)

    # Label the plot
    plt.xlabel('Min Samples Leaf (min_samples_leaf)')
    plt.ylabel('Min Samples Split (min_samples_split)')
    plt.title('Random Forest Hyperparameter Tuning (Grid Search)')
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def boxplot_augmentations(accuracy_vals, accuracy_std_vals, config_names, title):
    import matplotlib.pyplot as plt

    # Calculate boxplot statistics for each configuration
    def calculate_box(mean, std):
        q1 = mean - 0.675 * std  # Approximate 25th percentile
        q3 = mean + 0.675 * std  # Approximate 75th percentile
        whisker_low = mean - 1.5 * std  # Lower whisker
        whisker_high = mean + 1.5 * std  # Upper whisker
        return {
            'whislo': whisker_low,  # Bottom whisker
            'q1': q1,               # 25th percentile
            'med': mean,            # Median
            'q3': q3,               # 75th percentile
            'whishi': whisker_high  # Top whisker
        }

    # Generate box data for all configurations
    boxes = [calculate_box(mean, std) for mean, std in zip(accuracy_vals, accuracy_std_vals)]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Adjust positions of boxplots with spacings
    positions = range(1, len(boxes) + 1)  # Automatically space box plots
    ax.bxp(boxes, positions=positions, showfliers=False, patch_artist=True)

    # Annotate mean and std for each configuration
    for pos, mean, std, name in zip(positions, accuracy_vals, accuracy_std_vals, config_names):
        ax.text(pos, mean + 1.5 * std, f"Mean: {mean:.2f}\nStd: {std:.2f}", 
                ha='center', va='center', fontsize=8, color='black', bbox=dict(facecolor='white', alpha=0.8))

    # Customize the plot
    ax.set_title(title, fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Final layout adjustments
    plt.tight_layout()
    plt.show()
