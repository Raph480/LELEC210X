from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import set_config
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras_tuner.applications import HyperResNet, HyperEfficientNet, HyperXception


from audio_student import AudioUtil, Feature_vector_DS
from model_utils import *
from CNN_utils import *
#from classification.Q2.audio_student import AudioUtil, Feature_vector_DS
#from classification.Q1.model_utils import *
#from classification.Q2.CNN_utils import AudioUtil, Feature_vector_DS



def get_dataset(path="../datasets/sounds/recorded_sounds/classes/",filter_str=None,  Nft=512, n_melvec=20, melvec_height = 20, samples_per_melvec=512, window_type = "hamming", sr = 10200, flag_8bit = False, bit_sensitivity=0, normalize=True, shift_pct=0, img_idx = 0, verbose=False, play_sound = False):
    """
    Load and initialize the dataset with necessary attributes.
    """

    # Dataset class containing all sound files
    dataset = Dataset(folder=path, filter_str = filter_str)
    classnames = dataset.list_classes()
    if verbose:
        print("Number of elements in each class: ", dataset.naudio)
        print("Number of sounds in dataset: ", len(dataset))

    myds = Feature_vector_DS(dataset, n_melvec=n_melvec, melvec_height=melvec_height, Nft = Nft,
                             samples_per_melvec=samples_per_melvec, window_type=window_type, sr=sr, 
                             flag_8bit=flag_8bit, bit_sensitivity=bit_sensitivity, normalize=normalize) #Vérif si bons paramètres
    myds.data_aug = None  # Ensure no augmentation initially

    if verbose:
        fig, axs = plt.subplots(1, len(classnames), figsize=(len(classnames) * 4, 3))
        for i, ax in zip(range(len(classnames)), axs):
            ax.imshow(myds[classnames[i], img_idx].reshape((melvec_height, -1)), 
                      cmap="jet", origin="lower", aspect="auto")
            ax.set_title(classnames[i])
            ax.set_xlabel("")
            ax.set_ylabel("Mel bins")
        plt.colorbar(axs[-1].images[0], ax=axs, orientation='vertical')
        plt.show()
        if play_sound:
            for i in range(len(classnames)):
                myds.display([classnames[i], img_idx])
                time.sleep(1)
    
    return myds, dataset, classnames


def augment_dataset(myds, dataset, classnames, augmentations, melvec_height=20,
                    bg_dataset = None, bg_amplitude_limit=[0.1], physical_bg_dataset = None, my_phy_ds = None, shift_nb = 1, 
                    load_matrix = False, verbose=False, img_idx=0, aug_indexes = [0], play_sound=False,pickle_name = "feature_matrix_2D_aug.npy"):
    """
    Augment dataset and compute feature matrix.
    """
    train_pct = 1
    featveclen = len(myds["gun", 0])  # Number of items in a feature vector
    nitems = len(myds)  # Number of sounds in dataset
    naudio = dataset.naudio  # Number of audio files per class
    nclass = dataset.nclass  # Number of classes
    nlearn = round(naudio * train_pct)  # Training sample count

    n_physical = 0 if physical_bg_dataset is None else physical_bg_dataset.naudio

    myds.data_aug = None  # Ensure no augmentation initially
    data_aug_factor = len(augmentations)
    myds.data_aug_factor = data_aug_factor
    #myds.mod_data_aug(augmentations)

    myds.bg_dataset = bg_dataset
    myds.bg_amplitude_limit = bg_amplitude_limit

    create = True
    if load_matrix:
        create = False
        try:
            X_aug = np.load(pickle_name +'.npy', allow_pickle=True)
            y_aug = np.load(pickle_name + "labels.npy", allow_pickle=True)
            print("Previously computed dataset successfully loaded!")
        except FileNotFoundError:
            print(f"File {pickle_name} not found. Generating new dataset.")
            create = True

    if create:
        nb_phi = 1 if "physical_bg" in augmentations else 0
        start_time = time.time()
        X_aug = np.zeros((nclass * naudio  + shift_nb * nclass * naudio +  len(bg_amplitude_limit) * nclass * naudio + nb_phi * nclass * n_physical, featveclen))
        y_aug = np.empty((nclass * naudio  + shift_nb * nclass * naudio +  len(bg_amplitude_limit) * nclass * naudio + nb_phi * nclass * n_physical,), dtype=object)
        print("Number of shifts: ", shift_nb)
        print("Number of bg_amplitude_limit and values: ", len(bg_amplitude_limit), bg_amplitude_limit)
        print("Physical augmentation: ", (my_phy_ds is not None))
        print("X_aug shape: ", X_aug.shape)

        #original dataset
        for idx in range(naudio):
            for class_idx, classname in enumerate(classnames):
                try:
                    featvec = myds[classname, idx]
                    X_aug[class_idx * naudio + idx, :] = featvec
                    y_aug[class_idx * naudio + idx] = classname
                except:
                    print(f"Error at {classname}, {idx}")
                    continue
        
        
        # Individual augmentations
        if len(augmentations) != 0:
            for current_aug in augmentations:
                myds.data_aug = current_aug
                print(f"Augmenting with {current_aug}")

                if current_aug == "time_shift":
                    myds.data_aug = "time_shift"

                    for j in range(0,shift_nb):
                        print("Shift number: ", j+1)
                        myds.shift_pct = np.random.uniform(0, 1)
                        print("Shift percentage: ", myds.shift_pct)                        
                        for idx in range(dataset.naudio):
                            print(f"%d/%d" % (idx, dataset.naudio))
                            for class_idx, classname in enumerate(classnames):
                                #print(classname, idx)
                                featvec = myds[classname, idx]
                                #print((s+1) * nclass * naudio + j * nclass * naudio + class_idx * naudio + idx)
                                X_aug[nclass * naudio + j * nclass * naudio + class_idx * naudio + idx, :] = featvec
                                y_aug[nclass * naudio + j * nclass * naudio + class_idx * naudio + idx] = classname

                elif current_aug == "add_bg":
                    myds.data_aug = "add_bg"

                    for j in range(0,len(bg_amplitude_limit)):
                        print("Background number: ", j)
                        myds.bg_amplitude_limit = bg_amplitude_limit[j]
                        for idx in range(dataset.naudio):
                            print(f"%d/%d" % (idx, dataset.naudio))
                            for class_idx, classname in enumerate(classnames):
                                sound_name = dataset.__getname__((classname, idx))  # HUGE CORRECTION
                                # Skip processing if "background" is in the filename
                                if "background" in sound_name.lower():  
                                    continue 
                                #print(classname, idx)
                                featvec = myds[classname, idx]
                                #print(nclass * naudio + shift_nb * nclass * naudio + j * naudio * nclass + class_idx * naudio + idx)
                                X_aug[nclass * naudio + shift_nb * nclass * naudio + j * naudio * nclass + class_idx * naudio + idx, :] = featvec
                                y_aug[nclass * naudio + shift_nb * nclass * naudio + j * naudio * nclass+ class_idx * naudio + idx] = classname
                
                elif current_aug == "physical_bg":
                    if physical_bg_dataset is not None:
                        for idx in range(n_physical):
                            for class_idx, classname in enumerate(classnames):
                                featvec = my_phy_ds[classname, idx]
                                X_aug[nclass * naudio + shift_nb * nclass * naudio + len(bg_amplitude_limit) * naudio * nclass + class_idx
                                        * n_physical + idx, :] = featvec
                                y_aug[nclass * naudio + shift_nb * nclass * naudio + len(bg_amplitude_limit) * naudio * nclass + class_idx
                                        * n_physical + idx] = classname
                else:
                    print("ERROR - Wrong augmentation name!")

                        
        # Combined augmentations - TODO if necessary

        #X_aug = X_aug / np.linalg.norm(X_aug, axis=1, keepdims=True)  # Normalize - already done
        
        np.save(pickle_name, X_aug, allow_pickle=True)
        np.save(pickle_name + "labels", y_aug, allow_pickle=True)
        end_time = time.time()
        #print the time in minutes
        print("Time taken to augment the dataset: ", (end_time - start_time)/60, " minutes")
    if verbose:
        print(f"Shape of feature matrix: {X_aug.shape}")
        print(f"Number of labels: {len(y_aug)}")

        for aug_idx in aug_indexes:
            fig, axs = plt.subplots(1, len(classnames), figsize=(len(classnames) * 4, 3))
        
            for class_idx, ax in enumerate(axs):
                # Compute the correct index based on augmentation type
                if aug_idx == 0:
                    # Original dataset
                    data_index = class_idx * naudio + img_idx
                elif 1 <= aug_idx <= shift_nb:
                    # Time shift augmentations
                    shift_offset = nclass * naudio
                    data_index = shift_offset + (aug_idx - 1) * nclass * naudio + class_idx * naudio + img_idx
                elif shift_nb < aug_idx <= shift_nb + len(bg_amplitude_limit):
                    # Background noise augmentations
                    bg_offset = (1 + shift_nb) * nclass * naudio
                    bg_idx = aug_idx - (shift_nb + 1)
                    data_index = bg_offset + bg_idx * nclass * naudio + class_idx * naudio + img_idx
                elif aug_idx > shift_nb + len(bg_amplitude_limit):
                    # Physical augmentation
                    phys_offset = (1 + shift_nb + len(bg_amplitude_limit)) * nclass * naudio
                    phys_idx = aug_idx - (shift_nb + len(bg_amplitude_limit) + 1)
                    data_index = phys_offset + phys_idx * nclass * n_physical + class_idx * n_physical + img_idx
                else:
                    print(f"Skipping invalid aug_idx {aug_idx}")
                    continue


                # Ensure index is within bounds
                if data_index >= X_aug.shape[0]:
                    print(f"Skipping aug_idx {aug_idx}, class_idx {class_idx}, img_idx {img_idx}: Index out of bounds.")
                    continue

                print(f"Plotting index: {data_index} (Aug: {aug_idx}, Class: {classnames[class_idx]}, Img: {img_idx})")

                # Plot the spectrogram/mel features

                #Transform
                ax.imshow(
                    X_aug[data_index].reshape((melvec_height, -1)), 
                    cmap="jet",
                    origin="lower",
                    aspect="auto",
                )
                ax.set_title(classnames[class_idx])
                ax.set_xlabel("")
                ax.set_ylabel("Mel bins")
            
        #Ensure axis is within bounds to add the colorbar
        #No more necessary as max always 1
        #if len(axs) > 0:
        #    plt.colorbar(axs[0].images[0], ax=axs, orientation='vertical')
        plt.show()
                
    return X_aug, y_aug


def get_picklename(
    flag_8bit,
    bit_sensitivity,
    Nft,
    samples_per_melvec,
    n_melvec,
    melvec_height,
    window_type,
    sr,
    augmentations,
    shift_nb,
    bg_amplitude_limit,
    physical_aug,
    prefix="../datasets/melvecs/",
    purpose="melvecs"  # or "model"
):
    """
    Generate a unique pickle name based on dataset parameters and augmentations.
    """
    # Represent precision info as a string
    if flag_8bit:
        precision_str = f"int8s{bit_sensitivity}"  # int8 with sensitivity
    else:
        precision_str = "int16"

    pickle_name = prefix

    # Base name with physical parameters, excluding duration
    pickle_name += (
        f"{purpose}_{precision_str}_{Nft}_{samples_per_melvec}_"
        f"{n_melvec}_{melvec_height}_{window_type}_{int(sr)}"
    )

    # Append augmentation-specific parameters
    for aug in augmentations:
        pickle_name += f"_{aug}"
        if aug == "time_shift":
            pickle_name += f"_{shift_nb}"
        if aug == "add_bg":
            pickle_name += f"_{len(bg_amplitude_limit)}"

    # Add physical augmentation flag (only once)
    if physical_aug and "_physical_bg" not in pickle_name:
        pickle_name += "_physical_bg"

    return pickle_name



from seaborn import heatmap

def show_confusion_matrix(y_predict, y_true, classnames, title=""):
    """
    From target labels and prediction arrays, sort them appropriately and plot confusion matrix.
    The arrays can contain either ints or str quantities, as long as classnames contains all the elements present in them.
    """
    plt.figure(figsize=(6, 6))  # Enlarged the figure size for better visibility
    
    # Compute confusion matrix
    confmat = confusion_matrix(y_true, y_predict)
    
    # Plot confusion matrix with enlarged annotations
    heatmap(
        confmat.T,
        square=True,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=classnames,
        yticklabels=classnames,
        annot_kws={"size": 14},  # Increase font size for annotations
        ax=plt.gca(),
    )
    
    # Set larger font sizes for labels and title
    plt.xlabel("True label", fontsize=16)
    plt.ylabel("Predicted label", fontsize=16)
    plt.title(title, fontsize=18)
    
    # Enlarge tick labels
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    # Display the plot
    plt.show()
    return None

def perform_kfold(model, X, y, k=5, epochs=5, batch_size=32):
    print("K-fold cross-validation")
    # Ensure the model is compiled
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores, recall_scores, f1_scores = [], [], []
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        y_pred = np.argmax(model.predict(X_val), axis=1)
        y_true = y_val if len(y_val.shape) == 1 else np.argmax(y_val, axis=1)
        
        report = classification_report(y_true, y_pred, output_dict=True)
        accuracy_scores.append(report['accuracy'])
        recall_scores.append(report['macro avg']['recall'])
        f1_scores.append(report['macro avg']['f1-score'])
    
    kfold_acc = np.mean(accuracy_scores)
    kfold_recall = np.mean(recall_scores)
    kfold_f1 = np.mean(f1_scores)
    print(f"KFold Accuracy: {kfold_acc:.5f}")
    print(f"KFold Recall: {kfold_recall:.5f}")
    print(f"KFold F1-score: {kfold_f1:.5f}")

    return kfold_acc, kfold_recall, kfold_f1

def evaluate_model(model, X_train, y_train, classnames, X_test=None, y_test=None, show_confusion = False):
    model.fit(X_train, y_train, epochs=5, validation_split=0.2, verbose=1)
    
    y_pred_train = np.argmax(model.predict(X_train), axis=1)
    y_true_train = y_train if len(y_train.shape) == 1 else np.argmax(y_train, axis=1)

    if (X_test is not None):
        y_pred_test = np.argmax(model.predict(X_test), axis=1)
        y_true_test = y_test if len(y_test.shape) == 1 else np.argmax(y_test, axis=1)
    
    if (X_test is not None):
        test_accuracy = np.mean(y_pred_test == y_true_test)
        print(f"Accuracy on test set: {test_accuracy:.5f}")
    train_confmat = confusion_matrix(y_true_train, y_pred_train)
    if not show_confusion:
        print("Confusion matrix on train set: \n", train_confmat)
    test_confmat = None
    if (X_test is not None):
        test_confmat = confusion_matrix(y_true_test, y_pred_test)
        if not show_confusion:
            print("Confusion matrix on test set: \n", test_confmat)
    if show_confusion:
        print("Confusion Matrix (Train):")
        show_confusion_matrix(y_pred_train, y_true_train, classnames, title="Confusion Matrix - Train")
        print("Confusion Matrix (Test):")
        show_confusion_matrix(y_pred_test, y_true_test, classnames, title="Confusion Matrix - Test")
    
    print("\nClassification Report:")
    print("Train: ")
    train_report = classification_report(y_true_train, y_pred_train, target_names=classnames)
    print(train_report)
    test_report = None
    if (X_test is not None):
        print("Test: ")
        test_report = classification_report(y_true_test, y_pred_test, target_names=classnames)
        print(test_report)
    return train_confmat, train_report, test_accuracy, test_confmat, test_report

def save_results(hp_list, kfold_acc_res, mean_recal_res, kfold_f1_res, comp_t, prefix, name,
                 test_accuracy_res=None, verbose=True):
    
    
    if verbose:
        print("Mean accuracy: ", [float(x) for x in kfold_acc_res])
        print("Mean recall: ", [float(x) for x in mean_recal_res])
        print("Mean F1 score: ", [float(x) for x in kfold_f1_res])

        if test_accuracy_res is not None:
            print("Test accuracy: ", [float(x) for x in test_accuracy_res])

    # Base result dict
    results_dict = {
        'HPs': hp_list,
        'kfold_accuracy': kfold_acc_res,
        'kfold_recall': mean_recal_res,
        'kfold_f1': kfold_f1_res,
        'time': comp_t
    }

    if test_accuracy_res is not None:
        results_dict['test_accuracy'] = test_accuracy_res


    # Convert to DataFrame and save
    results_df = pd.DataFrame(results_dict)

    # Ensure the folder exists
    os.makedirs(prefix, exist_ok=True)

    filename = f'{prefix}/results_{name}_{hp_list[0]}_{hp_list[-1]}.csv'
    results_df.to_csv(filename, index=False)
    print(f"Results saved in {filename}")

def plot_results(hp_list, kfold_acc_res=None,mean_recal_res=None,kfold_f1_res=None, test_accuracy_res=None, xlabel= "x", title="title", name="noname", prefix= "../datasets/GSresults/", verbose=False):
    
     #Include name in path
    prefix = f"{prefix}/{name}"
    
    plt.figure(figsize=(10, 6))
    if kfold_acc_res is not None:
        plt.plot(hp_list, kfold_acc_res, label='KFold Accuracy', marker='o')
    if mean_recal_res is not None:
        plt.plot(hp_list, mean_recal_res, label='KFold Recall', marker='o')
    if kfold_f1_res is not None:
        plt.plot(hp_list, kfold_f1_res, label='KFold F1 Score', marker='o')
    if test_accuracy_res is not None:
        plt.plot(hp_list, test_accuracy_res, label='Test Accuracy', marker='o', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid()
    #save as svg
    
    plt.savefig(f'{prefix}.svg', format='svg')
    if verbose:
        plt.show()
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_csv_files(file_paths, xlabel="HPs", descending=False, show=True, save = 'None'):
    # List to collect DataFrames
    all_data = []

    # Read all CSVs and append to the list
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        all_data.append(df[['HPs', 'kfold_accuracy', 'test_accuracy']])
    
    # Concatenate all DataFrames
    combined_df = pd.concat(all_data)

    # Group by HPs and calculate mean and std
    stats = combined_df.groupby('HPs').agg(
        kfold_accuracy_mean=('kfold_accuracy', 'mean'),
        kfold_accuracy_std=('kfold_accuracy', 'std'),
        test_accuracy_mean=('test_accuracy', 'mean'),
        test_accuracy_std=('test_accuracy', 'std')
    ).reset_index()

    # Sort by HPs in ascending or descending order
    stats = stats.sort_values('HPs', ascending=not descending)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot K-Fold Accuracy with error bars
    plt.errorbar(stats['HPs'], stats['kfold_accuracy_mean'],
                 yerr=stats['kfold_accuracy_std'], fmt='-o',
                 capsize=5, label='K-Fold Accuracy', color='blue')

    # Plot Test Accuracy with error bars
    plt.errorbar(stats['HPs'], stats['test_accuracy_mean'],
                 yerr=stats['test_accuracy_std'], fmt='-o',
                 capsize=5, label='Test Accuracy', color='green')

    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.title("Mean and Std Dev of Accuracies vs " + xlabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
   
    if save:
        plt.savefig(save + '.svg')
    
    if show:
        plt.show()