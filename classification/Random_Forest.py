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
from model_utils import *


### 1. Model without hyperparameters tuning, without normalization & without reduction (PCA)
###--------------------------------------------

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

analyse_5_or_10_Kfold = True
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
#show_confusion_matrix(y_train, y_pred_train, classnames, title="confusion_matrix_train", save=False)
#show_confusion_matrix(y_test, y_pred_test, classnames, title="confusion_matrix_test", save=True)

#Precision, recall, F1-score (better to do it directly with K_fold?)
print_classification_report = False
if print_classification_report:
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=classnames))

#3. Add dB mismatch
#------------------
dB_mismatch = 10
#On k fold (30% of dataset is mismatched, then kfold is done)
_,_, accuracy_val_classic_db, accuracy_std_val_classic_db = \
    perform_Kfold(X_train, y_train, model_rf, kf, normalization, \
                  reduction, PCA_components, verbose=False, dB_mismatch=dB_mismatch)


X_test_scaled = X_test* 10 ** (-dB_mismatch / 20)
y_pred_test_db = model_rf.predict(X_test_scaled)
accuracy_test_db_clasic = accuracy(y_test, y_pred_test_db)
#TODO: Find if possible to compare dB mismatch with other models on a graph


### 2. Model with FV normalization, without hyperparameters tuning & without reduction (PCA)
###--------------------------------------------

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

### 3. Model without FV normalization but standartScaler normalization,without reduction (PCA)
###--------------------------------------------
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

### 4. Analyse three models with boxplot
###--------------------------------------------

boxplot_models(accuracy_val_classic, accuracy_val_FVNorm, accuracy_val_standardScaler,
               accuracy_std_val_classic, accuracy_std_val_FVnorm, accuracy_std_val_standardScaler,
               "No normalization", "FV Normalization", "StandardScaler", title= "Without dB mismatch")

#Add dB mismatch
boxplot_models(accuracy_val_classic_db, accuracy_val_FVNorm_db, accuracy_val_StandartScaler_db,
               accuracy_std_val_classic_db, accuracy_std_val_FVnorm_db, accuracy_std_val_StandartScaler_db,
               "No normalization", "FV Normalization", "StandardScaler (Worst case)", title= "With dB mismatch")

#Add dB mismatch
print("\---------------------------------------\n\
      Accuracy of three models with dB mismatch on test set:", dB_mismatch)
print(f"Without normalization: {accuracy_test_db_clasic:.5f}")
print(f"With FV normalization: {accuracy_test_db_FVnorm:.5f}")
print(f"With StandardScaler normalization: {accuracy_test_db_StandartScaler:.5f}")


### 5. Reduction analysis with PCA
###--------------------------------------------
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
PC_end = 20

perform_KFold_eval_n_PC(X_train,y_train,model_rf,kf, PC_start, PC_end,mean_without_PCA, verbose=False)

### 6. Hyperparameters tuning
###--------------------------------------------
print("\n-----------------------------------------------\n\
Random Forest Classifier, with hyperparameters tuning, with FV normalization & without reduction (PCA)")


FV_normalization = True 
normalization = False 
reduction = False 

X, y, classnames = get_dataset_matrix(normalization=FV_normalization)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#Model creation
#------------------

# Random Forest Grid Search
def Random_Forest_Model_GridSearch(X_train, X_test, y_train, y_test, model, kf):
    param_grid = {
        'n_estimators': [50, 75, 100, 125, 150, 200],
        'max_depth': [3, 4, 5, 7, None],
        'min_samples_split': [2, 4, 5, 6, 10],
        'min_samples_leaf': [1, 2, 4, 5, 6],
        'bootstrap': [True]
    }
    
    rf = RandomForestClassifier()
    rf_grid = perform_grid_search(model, param_grid, kf, X_train, y_train, "Random Forest")
    accuracy_train = rf_grid.best_score_
    accuracy_test = accuracy(y_test, rf_grid.predict(X_test))
    print_result(rf_grid, "Random Forest", accuracy_train, accuracy_test)
    return rf_grid.best_params_

model_rf = RandomForestClassifier()
kf = KFold(n_splits=10, shuffle=True, random_state=42)
#To compute the best hyperparameters
#Random_Forest_Model_GridSearch(X_train, X_test, y_train, y_test, model_rf, kf) # Uncomment to compute the best hyperparameters (it takes time!!!)

#Heatmap plot

# Define the hyperparameter grid (only for n_estimators and max_depth)
param_grid = {
    'n_estimators': list(range(10, 201, 10)),  # From 10 to 200, step of 10
    'max_depth': list(range(1, 21))  # From 1 to 20, step of 1
}

# Perform grid search and plot the results as a heatm
plot_grid_search_heatmap(X_train, y_train, param_grid) #it takes time!!

# 7. Final model: FV Normalization, No PCA and best hyperparameters
#------------------

print("\n-----------------------------------------------\n\
Final Classifier, with FV normalization & without reduction (PCA)")
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
params = {'bootstrap': True, 'max_depth': 7, 'min_samples_leaf': 1,
           'min_samples_split': 2, 'n_estimators': 100}

model_rf = RandomForestClassifier(**params)

#1. K Fold evaluation (the most rigorous)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
_,_, accuracy_val_FVNorm, accuracy_std_val_FVnorm = \
    perform_Kfold(X_train, y_train, model_rf, kf, normalization, reduction, PCA_components, verbose=True)


#TODO: Adapt K-fold function so that it shows mean prediction, recall and F1 score ? 

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
print_classification_report = True
if print_classification_report:
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=classnames))


Random_Forest_Model_GridSearch(X_train, X_test, y_train, y_test, model_rf, kf) # Uncomment to compute the best hyperparameters (it takes time!!!)
