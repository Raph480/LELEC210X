import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import set_config
from sklearn.decomposition import PCA

from classification.model_utils import *

"""
#Pipeline
#-------------------------
1. Dataset import -> get-dataset_matrix()
3. Data preparation -> get_dataset_augmented()
2. Data split -> train_test_split()
3. Data scaling -> StandardScaler()
4. Data processing -> PCA?
5. Modelling -> Tensorflow CNNs (HyperEfficientNet, HyperResNet, HyperDenseNet, 
                HyperXception, HyperMobileNet), self-made)
6. Training and 7. Evaluation -> Tensorflow Keras tuner
8. Final_model -> perform_KFold, fit, predict, accuracy, confusion_matrix, classification_report
#-------------------------

Wanted to use sklearn pipeline but seems too restrictive compared to the benefits
"""
verbose = True

#1 + #3. Dataset import & preparation

#TODO: TO IMPROVE: data augmentation
## + Sound analysis: if soud not loud enough, set its label to "garbage"
## Also garbage if only noise / ambiant sound
augmentations = ["original","add_bg", "scaling", "time_shift", "add_noise", "add_echo"]
X, y = get_dataset_matrix_augmented(augmentations)

#2. Data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

if (verbose):
    print(f"Shape of the training matrix : {X_train.shape}")
    print(f"Number of training labels : {len(y_train)}")

#3. Data preparation - Find if something to do here

#4. Data processing - Create FV from dataset : time duration, potential 
# PCA
#TODO

#5. Modelling
#TODO: Tensorflow CNNs (HyperEfficientNet, HyperResNet, HyperDenseNet, HyperXception, HyperMobileNet), self-made)

#6. Training and 7. Evaluation
#TODO: Tensorflow Keras tuner

#8. Final_model
#TODO: perform_KFold, fit, predict, accuracy, confusion_matrix, classification_report