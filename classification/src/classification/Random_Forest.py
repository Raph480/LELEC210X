import matplotlib.pyplot as plt
import numpy as np
import os
import time


"Machine learning tools"
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
from classification.datasets import Dataset
from classification.utils.audio_student import AudioUtil, Feature_vector_DS
from classification.utils.plots import (
    plot_decision_boundaries,
    plot_specgram,
    show_confusion_matrix,
)
from classification.utils.utils import accuracy


#All the functions are in the model_utils.py file
from classification.model_utils import *
import glob

dB_mismatch = 10
### 1. Model without hyperparameters tuning, without normalization & without reduction (PCA)
###--------------------------------------------
def model_original():
    print("\n-----------------------------------------------\n\
    Random Forest Classifier, without hyperparameters tuning, without normalization & without reduction (PCA)")
    #Get dataset matrix
    #------------------

    FV_normalization = False #Normalization directly feature vectors
    normalization = False #Normalization for the model with standartScaler
    reduction = False #Reduction with PCA
    PCA_components = 7 #for kfold only if reduction is True
    print("FV normalization: ", FV_normalization)

    X, y, classnames = get_dataset_matrix(normalization=FV_normalization)
    #Verify that the dataset is balanced (it is)
    print("Number of samples per class:")
    print(np.unique(y, return_counts=True))


    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # Stratify enable to have the same proportion of each class in the training and testing set

    #Verify that the dataset is balanced (it is, thanks to stratify)
    print("Number of samples per class in the training set:")
    print(np.unique(y_train, return_counts=True))
    print("Number of samples per class in the test set:")
    print(np.unique(y_test, return_counts=True))
    print(f"Shape of the training matrix : {X_train.shape}")
    print(f"Number of training labels : {len(y_train)}")

    #Model creation
    #------------------
    params = {'bootstrap': True, 'max_depth': 4, 'min_samples_leaf': 2,
            'min_samples_split': 4, 'n_estimators': 150}

    model_rf = RandomForestClassifier(**params)

    #0. Analysis of the number of folds to decide if n_splits = 5 or 10

    analyse_5_or_10_Kfold = False
    if analyse_5_or_10_Kfold:
        analyse_nb_splits_Kfold(X_train, y_train, model_rf, normalization, reduction, PCA_components, verbose=True)


    #1. K Fold evaluation (the most rigorous)
    #DECISION: n_splits = 10 so less variance in the validation set (but more computation time)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    _,_, accuracy_val_classic, accuracy_std_val_classic = \
        perform_Kfold(X_train, y_train, model_rf, kf, normalization, reduction, PCA_components, verbose=True)

    #TODO: Adapt K-fold function so that it shows mean prediction, recall and F1 score

    #2. Performance metrics
    # On the whole train and test set (no kfold, less accurate)
    #--------------------------------------------
    model_rf.fit(X_train, y_train)
    y_pred_train = model_rf.predict(X_train)
    y_pred_test = model_rf.predict(X_test)
    # Accuracy
    accuracy_train = accuracy(y_train, y_pred_train)
    accuracy_test = accuracy(y_test, y_pred_test)

    print("Training results:\n---------------------")
    print(f"Accuracy on train set: {accuracy_train:.5f}")
    print(f"Accuracy on test set: {accuracy_test:.5f}")

    # Confusion matrix
    #TODO: correct so that it saves correctly!
    show_confusion_matrix(y_train, y_pred_train, classnames, title="confusion_matrix_train")
    show_confusion_matrix(y_test, y_pred_test, classnames, title="confusion_matrix_test")

    #Precision, recall, F1-score (better to do it directly with K_fold?)
    print_classification_report = False
    if print_classification_report:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test, target_names=classnames))

    #3. Add dB mismatch
    #------------------

    #On k fold (30% of dataset is mismatched, then kfold is done)
    _,_, accuracy_val_classic_db, accuracy_std_val_classic_db = \
        perform_Kfold(X_train, y_train, model_rf, kf, normalization, \
                    reduction, PCA_components, verbose=False, dB_mismatch=dB_mismatch)


    X_test_scaled = X_test* 10 ** (-dB_mismatch / 20)
    y_pred_test_db = model_rf.predict(X_test_scaled)
    accuracy_test_db_clasic = accuracy(y_test, y_pred_test_db)


    return accuracy_val_classic, accuracy_std_val_classic, \
            accuracy_val_classic_db, accuracy_std_val_classic_db, \
            accuracy_test_db_clasic, 


### 2. Model with FV normalization, without hyperparameters tuning & without reduction (PCA)
###--------------------------------------------

def model_FVNnorm():
    print("\n-----------------------------------------------\n\
    Random Forest Classifier, with FV normalization & without reduction (PCA)")
    #Get dataset matrix
    #------------------

    FV_normalization = True #Normalization directly feature vectors
    normalization = False #Normalization for the model with standartScaler
    reduction = False #Reduction with PCA
    PCA_components = 7 #for kfold only if reduction is True
    print("FV normalization: ", FV_normalization)

    X, y, classnames = get_dataset_matrix(normalization=FV_normalization)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    #Model creation
    #------------------

    #From previous best hyperparameters
    params = {'bootstrap': True, 'max_depth': 4, 'min_samples_leaf': 2,
            'min_samples_split': 4, 'n_estimators': 150}

    model_rf = RandomForestClassifier(**params)

    #1. K Fold evaluation (the most rigorous)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    _,_, accuracy_val_FVNorm, accuracy_std_val_FVnorm = \
        perform_Kfold(X_train, y_train, model_rf, kf, normalization, reduction, PCA_components, verbose=True)


    #TODO: Adapt K-fold function so that it shows mean prediction, recall and F1 score

    #2. Performance metrics
    # On the whole train and test set (no kfold, less accurate)
    #--------------------------------------------
    model_rf.fit(X_train, y_train)
    y_pred_train = model_rf.predict(X_train)
    y_pred_test = model_rf.predict(X_test)
    # Accuracy
    accuracy_train = accuracy(y_train, y_pred_train)
    accuracy_test = accuracy(y_test, y_pred_test)

    print("Training results:\n---------------------")
    print(f"Accuracy on train set: {accuracy_train:.5f}")
    print(f"Accuracy on test set: {accuracy_test:.5f}")

    # Confusion matrix
    #TODO: correct so that it saves correctly!
    #show_confusion_matrix(y_train, y_pred_train, classnames, title="confusion_matrix_train", save=False)
    #show_confusion_matrix(y_test, y_pred_test, classnames, title="confusion_matrix_test", save=True)

    #Precision, recall, F1-score (better to do it directly with K_fold?)
    print_classification_report = False
    if print_classification_report:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test, target_names=classnames))

    #3. Add dB mismatch
    #------------------

    _,_, accuracy_val_FVNorm_db, accuracy_std_val_FVnorm_db = \
        perform_Kfold(X_train, y_train, model_rf, kf, normalization, \
                    reduction, PCA_components, verbose=False, dB_mismatch=dB_mismatch, FV_normalization=FV_normalization)


    X_train_normalised = X_train/ np.linalg.norm(
        X_train, axis=1, keepdims=True
    )

    X_test_scaled = X_test* 10 ** (-dB_mismatch / 20)
    X_test_normalised = X_test_scaled / np.linalg.norm(X_test_scaled, axis=1, keepdims=True)

    model_rf.fit(X_train_normalised, y_train)
    y_pred_test_db = model_rf.predict(X_test_normalised)
    accuracy_test_db_FVnorm = accuracy(y_test, y_pred_test_db)
    return accuracy_val_FVNorm, accuracy_std_val_FVnorm, \
        accuracy_val_FVNorm_db, accuracy_std_val_FVnorm_db, \
        accuracy_test_db_FVnorm


### 3. Model without FV normalization but standartScaler normalization,without reduction (PCA)
###--------------------------------------------
def model_standartnorm():
    print("\n-----------------------------------------------\n\
    Random Forest Classifier, with standartScaler normalization & without reduction (PCA)")
    #Get dataset matrix
    #------------------

    FV_normalization = False #Normalization directly feature vectors
    normalization = True #Normalization for the model with standartScaler
    reduction = False #Reduction with PCA
    PCA_components = 7 #for kfold only if reduction is True
    print("FV normalization: ", FV_normalization)

    X, y, classnames = get_dataset_matrix(normalization=FV_normalization)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    #Model creation
    #------------------

    #From previous best hyperparameters
    params = {'bootstrap': True, 'max_depth': 4, 'min_samples_leaf': 2,
            'min_samples_split': 4, 'n_estimators': 150}

    model_rf = RandomForestClassifier(**params)

    #1. K Fold evaluation (the most rigorous)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    _,_, accuracy_val_standardScaler, accuracy_std_val_standardScaler = \
        perform_Kfold(X_train, y_train, model_rf, kf, normalization, reduction, PCA_components, verbose=True)

    #TODO: Adapt K-fold function so that it shows mean prediction, recall and F1 score

    #2. Performance metrics
    # On the whole train and test set (no kfold, less accurate)
    #--------------------------------------------
    model_rf.fit(X_train, y_train)
    y_pred_train = model_rf.predict(X_train)
    y_pred_test = model_rf.predict(X_test)
    # Accuracy
    accuracy_train = accuracy(y_train, y_pred_train)
    accuracy_test = accuracy(y_test, y_pred_test)

    print("Training results:\n---------------------")
    print(f"Accuracy on train set: {accuracy_train:.5f}")
    print(f"Accuracy on test set: {accuracy_test:.5f}")

    # Confusion matrix
    #TODO: correct so that it saves correctly!
    #show_confusion_matrix(y_train, y_pred_train, classnames, title="confusion_matrix_train", save=False)
    #show_confusion_matrix(y_test, y_pred_test, classnames, title="confusion_matrix_test", save=True)

    #Precision, recall, F1-score (better to do it directly with K_fold?)
    print_classification_report = False
    if print_classification_report:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test, target_names=classnames))

    #3. Add dB mismatch
    #------------------

    _,_, accuracy_val_StandartScaler_db, accuracy_std_val_StandartScaler_db = \
        perform_Kfold(X_train, y_train, model_rf, kf, normalization, \
                    reduction, PCA_components, verbose=False, dB_mismatch=dB_mismatch)


    X_test_scaled = X_test* 10 ** (-dB_mismatch / 20)
    """
    #Re-create a X_train X_test with new X_test scaled
    #X_new = np.concatenate((X_train, X_test_scaled), axis=0)
    #y_new = np.concatenate((y_train, y_test), axis=0)

    #X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.3, random_state=42, stratify=y_new)
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_standardscaled = scaler.transform(X_test_scaled)

    model_rf.fit(X_train, y_train)
    y_pred_test_db = model_rf.predict(X_test_standardscaled)
    accuracy_test_db_StandartScaler = accuracy(y_test, y_pred_test_db)

    return accuracy_val_standardScaler, accuracy_std_val_standardScaler, \
        accuracy_val_StandartScaler_db, accuracy_std_val_StandartScaler_db, \
        accuracy_test_db_StandartScaler

### 4. Analyse three models with boxplot
###--------------------------------------------
def boxplot_and_mismatch(acc_val_classic, acc_val_FVNorm, acc_val_standardScaler,
                        acc_std_val_classic, acc_std_val_FVnorm, acc_std_val_standardScaler,
                        acc_val_classic_db, acc_val_FVNorm_db, acc_val_StandartScaler_db,
                        acc_std_val_classic_db, acc_std_val_FVnorm_db, acc_std_val_StandartScaler_db,
                        accuracy_test_db_clasic, accuracy_test_db_FVnorm, accuracy_test_db_StandartScaler,
                        title=""):
    boxplot_models(acc_val_classic, acc_val_FVNorm, acc_val_standardScaler,
                acc_std_val_classic, acc_std_val_FVnorm, acc_std_val_standardScaler,
                "No normalization", "FV Normalization", "StandardScaler", title= "Without dB mismatch")

    #Add dB mismatch
    boxplot_models(acc_val_classic_db, acc_val_FVNorm_db, acc_val_StandartScaler_db,
                acc_std_val_classic_db, acc_std_val_FVnorm_db, acc_std_val_StandartScaler_db,
                "No normalization", "FV Normalization", "StandardScaler (Worst case)", title= "With dB mismatch")

    #Add dB mismatch
    print("\---------------------------------------\n\
        Accuracy of three models with dB mismatch on test set:", dB_mismatch)
    print(f"Without normalization: {accuracy_test_db_clasic:.5f}")
    print(f"With FV normalization: {accuracy_test_db_FVnorm:.5f}")
    print(f"With StandardScaler normalization: {accuracy_test_db_StandartScaler:.5f}")


### 5. Reduction analysis with PCA
###--------------------------------------------

def model_PCA():
    print("\n-----------------------------------------------\n\
    Random Forest Classifier, with FV normalization & with reduction (PCA)")

    #Chosen model: FV normalization
    FV_normalization = True #Normalization directly feature vectors
    normalization = False #Normalization for the model with standartScaler
    reduction = True #Reduction with PCA

    X, y, classnames = get_dataset_matrix(normalization=FV_normalization)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    #Model creation

    params = {'bootstrap': True, 'max_depth': 4, 'min_samples_leaf': 2,
            'min_samples_split': 4, 'n_estimators': 150}

    model_rf = RandomForestClassifier(**params)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    #Get results without PCA
    _,_,mean_without_PCA,_ = perform_Kfold(X_train, y_train, model_rf, kf, normalization, 
                                        False, PCA_components=0, verbose=True, 
                                        dB_mismatch=0, FV_normalization=True)

    PC_start = 1
    PC_end = 50

    perform_KFold_eval_n_PC(X_train,y_train,model_rf,kf, PC_start, PC_end,mean_without_PCA, verbose=False)

### 6. Hyperparameters tuning
###--------------------------------------------
def hyperparam_plot():
    print("\n-----------------------------------------------\n\
    Random Forest Classifier, with hyperparameters tuning, with FV normalization & without reduction (PCA)")

    FV_normalization = True 
    normalization = False 
    reduction = False 

    X, y, classnames = get_dataset_matrix(normalization=FV_normalization)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Model creation
    # ------------------

    # Heatmap plot

    # Define the hyperparameter grid (only for min_samples_leaf and min_samples_split)
    param_grid = {
        'n_estimators': [90],  # Single value as a list
        'max_depth': [8],  # Single value as a list
        'min_samples_leaf': list(range(1, 20)),  # From 1 to 10, step of 1
        'min_samples_split': list(range(2, 20)),  # From 2 to 10, step of 1
    }

    # Perform grid search and plot the results as a heatmap
    plot_grid_search_heatmap(X_train, y_train, param_grid)  # This might take time

def hyperparameters_tuning():
    print("\n-----------------------------------------------\n\
    Random Forest Classifier, with hyperparameters tuning, with FV normalization & without reduction (PCA)")
    FV_normalization = True 
    normalization = False 
    reduction = False 

    augmentations = ["original", "scaling", "time_shift", "add_noise", "add_echo"]
    X, y, classnames = get_dataset_matrix_augmented(augmentations)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    #Model creation
    #------------------

    # Random Forest Grid Search
    param_grid = {
        'n_estimators': [80,90,100,110, 120, 130,140,150, 200],
        'max_depth': [5,6, 7,8,9,10],
        'min_samples_split': [3, 4, 5, 7, 10],
        'min_samples_leaf': [13, 15, 18,19,20, 25],
        'bootstrap': [True]
    }
    
    model = RandomForestClassifier()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rf_grid = perform_grid_search(model, param_grid, kf, X_train, y_train, "Random Forest")
    accuracy_train = rf_grid.best_score_
    accuracy_test = accuracy(y_test, rf_grid.predict(X_test))
    print_result(rf_grid, "Random Forest", accuracy_train, accuracy_test)
    return rf_grid.best_params_



# 7. Data augmentation analysis
#----------------------------------
def data_augmentation():
    print("\n-----------------------------------------------\n\
    Random Forest Classifier, with FV normalization & without reduction (PCA)")

    #Get dataset matrix
    #------------------

    FV_normalization = True #Normalization directly feature vectors
    normalization = False #Normalization for the model with standartScaler
    reduction = False #Reduction with PCA
    PCA_components = 7 #for kfold only if reduction is True
    print("FV normalization: ", FV_normalization)

    augmentations = ["original","add_bg", "scaling", "time_shift", "add_noise", "add_echo"]
    X, y = get_dataset_matrix_augmented(augmentations)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y_aug)

    #Model creation
    #------------------

    #From previous best hyperparameters
    params = {'bootstrap': True, 'max_depth': 7, 'min_samples_leaf': 1,
            'min_samples_split': 2, 'n_estimators': 100}

    model_rf = RandomForestClassifier(**params)

    #1. K Fold evaluation (the most rigorous)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    _,_, accuracy_val_FVNorm, accuracy_std_val_FVnorm = \
        perform_Kfold(X_train, y_train, model_rf, kf, normalization, reduction, PCA_components, verbose=True)



def evaluate_data_augmentations():
    print("\n-----------------------------------------------\n\
    Evaluating Data Augmentation Configurations")

    # Data Preparation
    FV_normalization = True  # Feature vector normalization
    normalization = False  # Model-specific normalization
    reduction = False  # No PCA reduction
    PCA_components = 7
    print("FV normalization: ", FV_normalization)

    dataset = Dataset()
    classnames = dataset.list_classes()

    myds = Feature_vector_DS(dataset, Nft=512, nmel=20, duration=950*5, shift_pct=0.2)

    # Define data augmentation configurations
    data_aug_configs = [
        None,  # 1. No data augmentation
        ["original"],  # 2. Only "original" data augmentation
        ["add_bg"],  # 3. Only "add_bg" data augmentation
        ["scaling"],  # 4. Only "scaling"
        ["time_shift"],  # 5. Only "time shift"
        ["add_noise"],  # 6. Only "add_noise"
        ["add_echo"],  # 7. Only "add_echo"
        ["original","add_bg","scaling","time_shift","add_noise","add_echo"],  # 8. All
        ["original","scaling","add_noise","time shift","add_echo"],  # 9. Best combination

    ]

    # Parameters for the Random Forest model
    params = {'bootstrap': True, 'max_depth': 8, 'min_samples_leaf': 13,
              'min_samples_split': 3, 'n_estimators': 150}
    model_rf = RandomForestClassifier(**params)

    # K-Fold configuration
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Lists to store results
    accuracy_vals = []
    accuracy_std_vals = []

    # Evaluate each data augmentation configuration
    for idx, data_aug in enumerate(data_aug_configs):
        print(f"\nEvaluating configuration {idx + 1}: {data_aug if data_aug else 'No Augmentation'}")

        # Apply the data augmentation setting
        myds.mod_data_aug(data_aug)

        # Generate feature matrix and labels
        X_aug = np.zeros((myds.data_aug_factor * dataset.nclass * dataset.naudio, len(myds["fire", 0])))
        y_aug = np.empty((myds.data_aug_factor * dataset.nclass * dataset.naudio,), dtype=object)

        for s in range(myds.data_aug_factor):
            for i in range(dataset.naudio):
                for class_idx, classname in enumerate(classnames):
                    featvec = myds[classname, i]
                    X_aug[s * dataset.nclass * dataset.naudio + class_idx * dataset.naudio + i, :] = featvec
                    y_aug[s * dataset.nclass * dataset.naudio + class_idx * dataset.naudio + i] = classname

        y_aug = np.array(y_aug)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_aug, y_aug, test_size=0.3, random_state=42, stratify=y_aug
        )

        # Perform K-Fold evaluation
        _, _, accuracy_val, accuracy_std_val = perform_Kfold(
            X_train, y_train, model_rf, kf, normalization, reduction, PCA_components, verbose=False
        )

        # Store results
        accuracy_vals.append(accuracy_val)
        accuracy_std_vals.append(accuracy_std_val)

        print(f"Accuracy: {accuracy_val:.4f}, Std Dev: {accuracy_std_val:.4f}")

    # Return results
    return accuracy_vals, accuracy_std_vals



# 8. Final model: FV Normalization, No PCA and best hyperparameters
#------------------
def final_model(verbose=True):
    print("\n-----------------------------------------------\n\
    Final Classifier, with FV normalization, without reduction (PCA) & with data augmentation")

    #------------------
    normalization = False
    FVnormalization = True
    reduction = False
    PCA_components = 7
    

    augmentations = ["original", "scaling", "time_shift", "add_noise", "add_echo"]
    
    # Add real melvecs to X and y
    additionnal_melvecs = [[],[]]
    melvecs_txt_path = "melvecs_txt" # folder name
    # Get the list of folder names in the folder
    additionnal_melvecs_classes = [name for name in os.listdir(melvecs_txt_path) if os.path.isdir(os.path.join(melvecs_txt_path, name))]
    for class_name in additionnal_melvecs_classes:
        num_txt_files = len(glob.glob(os.path.join(melvecs_txt_path+f"/{class_name}", "*.txt")))
        print(f"Number of .txt files in {melvecs_txt_path}/{class_name}: {num_txt_files}")
        for num in range(1, num_txt_files+1):
            melvec = np.loadtxt(f"{melvecs_txt_path}/{class_name}/melvec_{num}.txt", 
                            dtype=str, 
                            converters={0: lambda x: int(x, 16)})
            melvec = np.array(melvec, dtype=np.uint16)
            additionnal_melvecs[0].append(melvec)
            additionnal_melvecs[1].append(class_name)

    X, y, classnames = get_dataset_matrix_augmented(augmentations, additionnal_melvecs)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    #Model creation
    #------------------

    #From previous best hyperparameters
    #params = {'bootstrap': True, 'max_depth': 8, 'min_samples_leaf': 19,
    #        'min_samples_split': 4, 'n_estimators': 110}

    params = {'bootstrap': True, 'max_depth': 8, 'min_samples_leaf': 13,
            'min_samples_split': 3, 'n_estimators': 150}
    #print the parameters
    print("Parameters of the model:")
    print(params)

    model_rf = RandomForestClassifier(**params)

    #1. K Fold evaluation (the most rigorous)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    _,_, accuracy_val_FVNorm, accuracy_std_val_FVnorm = \
        perform_Kfold(X_train, y_train, model_rf, kf, normalization, reduction, PCA_components, verbose=verbose, FV_normalization=FVnormalization)


    #TODO: Adapt K-fold function so that it shows mean prediction, recall and F1 score ? 

    #2. Performance metrics
    # On the whole train and test set (no kfold, less accurate)
    #--------------------------------------------
    model_rf.fit(X_train, y_train)
    if (verbose):
        y_pred_train = model_rf.predict(X_train)
        y_pred_test = model_rf.predict(X_test)

        # Accuracy
        accuracy_train = accuracy(y_train, y_pred_train)
        accuracy_test = accuracy(y_test, y_pred_test)

        print("Training results:\n---------------------")
        print(f"Accuracy on train set: {accuracy_train:.5f}")
        print(f"Accuracy on test set: {accuracy_test:.5f}")

        # Confusion matrix
        #TODO: correct so that it saves correctly!
        show_confusion_matrix(y_train, y_pred_train, classnames, title="confusion_matrix_train")
        show_confusion_matrix(y_test, y_pred_test, classnames, title="confusion_matrix_test")

        #Precision, recall, F1-score (better to do it directly with K_fold?)
        print_classification_report = True
        if print_classification_report:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_test, target_names=classnames))
    
    return model_rf




if __name__ == '__main__':
        
    #acc_val_classic, acc_std_val_classic, acc_val_classic_db, acc_std_val_classic_db, accuracy_test_db_clasic = model_original()
    #acc_val_FVNorm, acc_std_val_FVnorm, acc_val_FVNorm_db, acc_std_val_FVnorm_db, accuracy_test_db_FVnorm = model_FVNnorm()
    #acc_val_standardScaler, acc_std_val_standardScaler, acc_val_StandartScaler_db, acc_std_val_StandartScaler_db, accuracy_test_db_StandartScaler = model_standartnorm()

                        
    #boxplot_and_mismatch(acc_val_classic, acc_val_FVNorm, acc_val_standardScaler,
    #                        acc_std_val_classic, acc_std_val_FVnorm, acc_std_val_standardScaler,
    #                        acc_val_classic_db, acc_val_FVNorm_db, acc_val_StandartScaler_db,
    #                        acc_std_val_classic_db, acc_std_val_FVnorm_db, acc_std_val_StandartScaler_db,
    #                        accuracy_test_db_clasic, accuracy_test_db_FVnorm, accuracy_test_db_StandartScaler,
    #                        )

    #model_PCA()
    #hyperparam_plot()
    #hyperparam_compute()

    #data_augmentation()
    #accuracy_vals, accuracy_std_vals = evaluate_data_augmentations()
    #config_names = [
    #    "No Augmentation", "Original", "Add BG", "Scaling", 
    #    "Time Shift", "Add Noise", "Echo", "All", "Best"
    #]

    # Call the function
    #boxplot_augmentations(accuracy_vals, accuracy_std_vals, config_names, "Model Performance by Data Augmentation")

    #hyperparameters_tuning()
    model = final_model(verbose=True)
    # Save the model
    with open("final_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved as 'final_model.pkl'")