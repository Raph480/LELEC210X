import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import set_config
from sklearn.decomposition import PCA


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import set_config
from sklearn.decomposition import PCA

##from ..utils.audio_student import AudioUtil, Feature_vector_DS
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras_tuner.applications import HyperResNet, HyperEfficientNet, HyperXception


##from ..Q1.model_utils import *
##from ..Q2.CNN_utils import AudioUtil, Feature_vector_DS


# Now you can use absolute imports
#from classification.utils.audio_student import AudioUtil, Feature_vector_DS
#from classification.Q1.model_utils import *


from audio_student import AudioUtil, Feature_vector_DS
from model_utils import *
from CNN_utils import AudioUtil, Feature_vector_DS

#from utils.audio_student import AudioUtil, Feature_vector_DS
#from ..Q1.model_utils import *
#from ..Q2.CNN_utils import AudioUtil, Feature_vector_DS

"""
#Pipeline
#-------------------------
1. Dataset import -> get-dataset_matrix()
3. Data preparation -> get_dataset_()
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


from classification.src.classification.Q2.model_utils import *


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
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras_tuner.applications import HyperResNet, HyperEfficientNet, HyperXception

opt_1 = Adam(learning_rate=0.001)


#A. HyperResNet
hyperResNet = HyperResNet(input_shape=X_train.shape[1:], classes=4)

#B. HyperEfficientNet
hyperEfficientNet = HyperEfficientNet(input_shape=X_train.shape[1:], classes=4)

#C. HyperXception
hyperXception = HyperXception(input_shape=X_train.shape[1:], classes=4)


#D Self-made 1
def self_made_builder(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = X_train.shape[1:]))

    hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
    hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=512, step=32)
    hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=512, step=32)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) 

    model.add(tf.keras.layers.Dense(units=hp_layer_1 , activation=hp_activation))
    model.add(tf.keras.layers.Dense(units=hp_layer_2 , activation=hp_activation))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer=opt_1,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#6. Training and 7. Evaluation
#Using Tensorflow Keras tuner

import keras_tuner as kt

def create_tuner(hypermodel):
    return kt.Hyperband(hypermodel,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='hp_dir',
                     project_name='tuner1')

#(A: HyperResNet, B: HyperEfficientNet, C: HyperXception, D: self_made_builder)
tuner = create_tuner(hyperResNet)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

#8. Final_model
#TODO: perform_KFold, fit, predict, accuracy, confusion_matrix, classification_report

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2)


predictions = hypermodel.predict(X_test)
#Get probabilities
print(tf.nn.softmax(predictions).numpy())

eval_result = hypermodel.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
