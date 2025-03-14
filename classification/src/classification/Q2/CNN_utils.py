from classification.Q2.audio_student import AudioUtil, Feature_vector_DS
import matplotlib.pyplot as plt
import numpy as np

#from classification.Q1.model_utils import *
from classification.Q2.audio_student import AudioUtil, Feature_vector_DS

def get_dataset(Nft=512, nmel=20, duration=950, normalize=True, shift_pct=0, verbose=False):
    """
    Load and initialize the dataset with necessary attributes.
    """
    dataset = Dataset()
    classnames = dataset.list_classes()

    myds = Feature_vector_DS(dataset, Nft=Nft, nmel=nmel, duration=duration, shift_pct=shift_pct, normalize=normalize)
    myds.data_aug = None  # Ensure no augmentation initially

    if verbose:
        fig, axs = plt.subplots(1, len(classnames), figsize=(len(classnames) * 4, 3))
        for i, ax in zip(range(len(classnames)), axs):
            ax.imshow(myds[classnames[i], 0].reshape((nmel, -1)), cmap="jet", origin="lower", aspect="auto")
            ax.set_title(classnames[i])
            ax.set_xlabel("")
            ax.set_ylabel("Mel bins")
        plt.colorbar(axs[-1].images[0], ax=axs, orientation='vertical')
        plt.show()
    
    return myds, dataset, classnames

def augment_dataset(myds, dataset,augmentations, verbose=False):
    """
    Augment dataset and compute feature matrix.
    """
    train_pct = 1
    featveclen = len(myds["gun", 0])  # Number of items in a feature vector
    nitems = len(myds)  # Number of sounds in dataset
    naudio = dataset.naudio  # Number of audio files per class
    nclass = dataset.nclass  # Number of classes
    nlearn = round(naudio * train_pct)  # Training sample count

    X_aug = np.zeros((myds.data_aug_factor * nclass * naudio, featveclen))
    y_aug = np.empty((myds.data_aug_factor * nclass * naudio,), dtype=object)
    
    for s in range(myds.data_aug_factor):
        for idx in range(dataset.naudio):
            for class_idx, classname in enumerate(dataset.list_classes()):
                featvec = myds[classname, class_idx]
                X_aug[s * nclass * naudio + class_idx * naudio + idx, :] = featvec
                y_aug[s * nclass * naudio + class_idx * naudio + idx] = classname
    
    X_aug = X_aug / np.linalg.norm(X_aug, axis=1, keepdims=True)  # Normalize
    np.save("feature_matrix_2D_aug.npy", X_aug, allow_pickle=True)
    np.save("labels_aug.npy", y_aug, allow_pickle=True)

    print(f"Shape of feature matrix: {X_aug.shape}")
    print(f"Number of labels: {len(y_aug)}")

    if verbose:
        fig, axs = plt.subplots(1, len(dataset.list_classes()), figsize=(len(dataset.list_classes()) * 4, 3))
        for i, ax in zip(range(len(dataset.list_classes())), axs):
            ax.imshow(X_aug[i].reshape((myds.nmel, -1)), cmap="jet", origin="lower", aspect="auto")
            ax.set_title(dataset.list_classes()[i])
            ax.set_xlabel("")
            ax.set_ylabel("Mel bins")
        plt.colorbar(axs[-1].images[0], ax=axs, orientation='vertical')
        plt.show()
    
    return X_aug, y_aug