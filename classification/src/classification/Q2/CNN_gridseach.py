from CNN_utils import *
import pandas as pd
import keras_tuner as kt
from sklearn.preprocessing import LabelEncoder

tf.config.run_functions_eagerly(True)  # Force eager execution

from tensorflow.keras.utils import to_categorical
import numpy as np


def self_made_builder_factory(input_shape):
    #Enable to pass input_shape to the function
    def self_made_builder(hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))

        hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])
        hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=512, step=32)
        hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=512, step=32)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) 

        model.add(tf.keras.layers.Dense(units=hp_layer_1, activation=hp_activation))
        model.add(tf.keras.layers.Dense(units=hp_layer_2, activation=hp_activation))
        model.add(tf.keras.layers.Dense(4, activation='softmax'))

        opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
    return self_made_builder



def model_creation_evaluation(dtype, Nft, n_melvec, melvec_height, window_type, sr, shift_nb=0,
        bg_amplitude_limit=[], physical_bg=False):

    print("NEW MODEL")
    print("------------------------------------")
    print("Physical HP: ")
    print("dtype: ", dtype)
    print("Nft: ", Nft)
    print("n_melvec: ", n_melvec)
    print("melvec_height: ", melvec_height)
    print("window_type: ", window_type)
    print("sampling rate: ", sr)


    #1. Import classical  - #2. Train-test split
    #--------------------------------
    myds, dataset, classnames = get_dataset(path="../datasets/sounds/recorded_sounds/trainset/", filter_str=None, #"_1_"
        Nft=Nft, n_melvec=n_melvec, melvec_height=melvec_height, samples_per_melvec=samples_per_melvec,
        window_type=window_type, sr = sr, dtype=dtype,
        normalize=True, shift_pct=0, verbose=False, img_idx = img_idx, play_sound=False)
    
    myds_test, dataset_test, _ = get_dataset(path="../datasets/sounds/recorded_sounds/testset/", filter_str=None, #"_1_"
        Nft=Nft, n_melvec=n_melvec, melvec_height=melvec_height, samples_per_melvec=samples_per_melvec,
        window_type=window_type, sr = sr, dtype=dtype,
        normalize=True, shift_pct=0, verbose=False, img_idx = img_idx, play_sound=False)



    #3. Dataset augmentations
    #----------------------------
    print("\nDataset Augmentations")
    if "add_bg" in augmentations:
        bg_dataset = Dataset(folder="../datasets/sounds/recorded_sounds/background/", filter_str=None)
    else :
        bg_dataset = None

    if "physical_bg" in augmentations:
        my_phy_ds, phy_bg_dataset, classnames = get_dataset(path="../datasets/sounds/recorded_sounds/trainset/", filter_str="_background_",
        Nft=Nft, n_melvec=n_melvec, melvec_height=melvec_height, samples_per_melvec=samples_per_melvec,
        window_type=window_type, sr = sr, dtype=dtype,
        normalize=True, shift_pct=0, verbose=False, img_idx = img_idx, play_sound=False)
    else :
        my_phy_ds = None
        phy_bg_dataset = None

    pickle_name = get_picklename(
        dtype,
        Nft,
        samples_per_melvec,
        n_melvec,
        melvec_height,
        window_type,
        sr,
        augmentations,
        shift_nb,
        bg_amplitude_limit,
        physical_bg,
        prefix="../datasets/melvecs/HP_tuning/",
        purpose = "melvecs" # "model"
    )
    print("pickle_name: ", pickle_name)

    #Visualisation purposes
    verbose = False
    play_sound = False


    X_train, y_train = augment_dataset(myds, dataset, classnames, augmentations, melvec_height=melvec_height,
                        shift_nb = shift_nb, #numbers of shifts done
                        bg_dataset = bg_dataset, bg_amplitude_limit=bg_amplitude_limit, #dataset used for background noise, background amplitudes
                        physical_bg_dataset = phy_bg_dataset,my_phy_ds = my_phy_ds, #dataset used for physical background noises
                        verbose=verbose, img_idx=img_idx, aug_indexes=plot_indexes, play_sound=play_sound, #verbose parameters
                        load_matrix=False, pickle_name=pickle_name) #load and save parameters

    pickle_name_test = get_picklename(
    dtype,
    Nft,
    samples_per_melvec,
    n_melvec,
    melvec_height,
    window_type,
    sr,
    augmentations,
    shift_nb,
    bg_amplitude_limit,
    physical_bg,
    prefix="../datasets/melvecs/HP_tuning/",
    purpose = "melvecs_test" # "model"
    )

    X_test, y_test = augment_dataset(myds_test, dataset_test, classnames, augmentations = [], melvec_height=melvec_height,    
                    shift_nb = 0, #numbers of shifts done
                    bg_dataset = None, bg_amplitude_limit=[], #dataset used for background noise, background amplitudes
                    physical_bg_dataset = phy_bg_dataset,my_phy_ds = my_phy_ds, #dataset used for physical background noises
                    verbose=verbose, img_idx=img_idx, aug_indexes=plot_indexes, play_sound=play_sound, #verbose parameters
                load_matrix=False, pickle_name=pickle_name_test) #load and save parameters

    # Transform the labels to integers and save mapping
    label_to_id = {label: i for i, label in enumerate(classnames)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    y_train = np.array([label_to_id[label] for label in y_train])
    y_test = np.array([label_to_id[label] for label in y_test])


    #4. Modelling
    #----------------------------
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)  # Convert labels to numbers

    def create_tuner(hypermodel):
        return kt.Hyperband(hypermodel,
                        objective='val_accuracy',
                        max_epochs=epochs_tuner,
                        factor=3,
                        directory='hp_dir',
                        project_name='tuner1',
                        overwrite=False)


    # D: self_made_builder)
    input_shape = X_train.shape[1:]
    builder = self_made_builder_factory(input_shape)
    tuner = create_tuner(builder)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    tuner.search(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[stop_early])

    #5. Final model
    model_name = get_picklename(
        dtype,
        Nft,
        samples_per_melvec,
        n_melvec,
        melvec_height,
        window_type,
        sr,
        augmentations,
        shift_nb,
        bg_amplitude_limit,
        physical_bg,
        prefix="../datasets/models/",
        purpose = "model"
    )

    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[stop_early])

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=validation_split)

    #Final results
    #------------------------------

    # Train evaluation -  K-Fold cross-validation
    kfold_acc, kfold_recall, kfold_f1 = perform_kfold(model, X_train, y_train, k=k, epochs=epochs_kfold)

    # Test evaluation
    train_confmat, train_report, \
    test_accuracy, test_confmat, test_report = evaluate_model(model, X_train, y_train, classnames, X_test, y_test)

    if (X_test is not None):
        eval_result = hypermodel.evaluate(X_test, y_test)
        acc_test = eval_result[1]
        print("[test loss, test accuracy]:", eval_result)

    return kfold_acc, kfold_recall, kfold_f1, train_confmat, train_report, test_accuracy, test_confmat, test_report


def search_hp(dtype, Nft, n_melvec, melvec_height, window_type, sr, 
              shift_nb, bg_amplitude_limit, physical_bg,      
        hp_list, name, preprefix, hp_name, verbose=False):
    """
    Generalized hyperparameter search function.

    Parameters:
    - hp_list: list of values to search over
    - idx: experiment index or identifier
    - preprefix: base directory or path prefix
    - hp_name: string name of the hyperparameter (e.g. 'sr', 'n_melvec')
    - verbose: whether to print and plot results
    """
    prefix = f"{preprefix}{hp_name}"

    kfold_acc_res = []
    mean_recal_res = []
    kfold_f1_res = []
    comp_t = []

    train_confmat_res = []
    train_report_res = []
    test_accuracy_res = []
    test_confmat_res = []
    test_report_res = []


    search_initial_time = time.time()
    for hp_value in hp_list:

        # Set the hyperparameter value dynamically in the global or config context
        # This is a placeholder - replace with how you actually set hyperparameters
        # Dynamically assign the current hyperparameter to the correct variable
        local_n_melvec = n_melvec
        local_sr = sr
        local_melvec_height = melvec_height
        local_window_type = window_type
        local_Nft = Nft

        local_shift_nb = shift_nb
        local_bg_amplitude_limit = bg_amplitude_limit
        local_physical_bg = physical_bg

        if hp_name == 'n_melvec':
            local_n_melvec = hp_value
        elif hp_name == 'sr':
            print("original_duration: ", original_duration)
            local_sr = hp_value
            local_n_melvec = int(original_duration * local_sr / local_Nft)
            print('local sr: ', local_sr)
            print("local n_melvec: ", local_n_melvec)
            print("local duration: ", local_n_melvec * local_Nft / local_sr )
        elif hp_name == 'melvec_height':
            local_melvec_height = hp_value
        elif hp_name == 'window_type':
            local_window_type = hp_value
        elif hp_name == 'Nft':
            local_Nft = hp_value

        elif hp_name == 'shift_nb':
            local_shift_nb = hp_value
            print("Shift number: ", shift_nb)
        elif hp_name == 'bg_amplitude_limit':
            local_bg_amplitude_limit = hp_value
            print("Background amplitude limit: ", bg_amplitude_limit)
        elif hp_name == 'physical_bg':
            local_physical_bg = hp_value
            print("Physical background noise: ", physical_bg)

        else:
            raise ValueError(f"Unsupported hyperparameter name: {hp_name}")

        print(f"Testing {hp_name} = {hp_value}")

        individual_time_start = time.time()
        kfold_acc, kfold_recall, kfold_f1, train_confmat, train_report, \
            test_accuracy, test_confmat, test_report = model_creation_evaluation(dtype, local_Nft, local_n_melvec, local_melvec_height, local_window_type, local_sr,
                                                                                 local_shift_nb, local_bg_amplitude_limit, local_physical_bg)
    
        
        individual_time_stop = time.time()

        print(f"Necessary time for {hp_name}={hp_value}: {(individual_time_stop - individual_time_start)/60:.2f} min")
        
        kfold_acc_res.append(kfold_acc)
        mean_recal_res.append(kfold_recall)
        kfold_f1_res.append(kfold_f1)
        comp_t.append((individual_time_stop - individual_time_start)/60)
        train_confmat_res.append(train_confmat)
        train_report_res.append(train_report)
        test_accuracy_res.append(test_accuracy)
        test_confmat_res.append(test_confmat)
        test_report_res.append(test_report)

    search_final_time = time.time()
    print("Total search time: ", (search_final_time - search_initial_time)/60, "min")

    save_results(hp_list, kfold_acc_res, mean_recal_res, kfold_f1_res, comp_t, prefix, name,test_accuracy_res)

    plot_results(hp_list, kfold_acc_res, mean_recal_res, kfold_f1_res, test_accuracy_res,
                    xlabel=hp_name, title=f'Model performance vs. {hp_name}',
                    name=name, prefix=prefix, verbose=verbose)


def plot_mean_results(HP):
    if HP == "melvec_height":
        path1 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/melvec_height/results_wednesday16_global_tests_same_tuner_2_2_30.csv"
        path2 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/melvec_height/results_wednesday16_global_tests_same_tuner_2_30.csv"
        path3=  "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/melvec_height/results_tusday17_global_tests_same_tuner_2_30.csv"
        path4 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/melvec_height/results_tusday17_global_tests_same_tuner_2_2_30.csv"
        path5 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/melvec_height/results_tuesday15_global_tests_same_tuner_2_30.csv"
        path6 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/melvec_height/results_monday14_global_tests_same_tuner_2_30.csv"
        path7 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/melvec_height/results_friday18_global_tests_same_tuner_2_30.csv"
        xlabel = "melvec_height"

    elif HP == "n_melvec":
        path1 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/n_melvec/results_wednesday16_global_tests_same_tuner_2_2_32.csv"
        path2 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/n_melvec/results_wednesday16_global_tests_same_tuner_2_32.csv"
        path3=  "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/n_melvec/results_tusday17_global_tests_same_tuner_2_32.csv"
        path4 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/n_melvec/results_tusday17_global_tests_same_tuner_2_2_32.csv"
        path5 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/n_melvec/results_tuesday15_global_tests_same_tuner_2_32.csv"
        path6 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/n_melvec/results_monday14_global_tests_same_tuner_2_32.csv"
        path7 = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/GSresults/n_melvec/results_friday18_global_tests_same_tuner_2_32.csv"
        xlabel = "n_melvec"


    paths = [path1, path2, path3, path4, path5, path6, path7]

    plot_csv_files(paths, xlabel)


if __name__ == '__main__':

    plot_mean_results("n_melvec")

    do_melvec = False
    do_sr = False
    do_melvec_height = False
    do_window_type = False
    do_Nft = False

    do_shifts = False
    do_bg = False
    do_physical_bg = False


    verbose = False #To show graphs
    preprefix = "../datasets/GSresults/"
    name = "friday18_global_tests_same_tuner"

    #Physical HP
    #------------------------------------
    original_dtype = np.int16
    original_Nft = 512
    original_samples_per_melvec = original_Nft
    original_n_melvec = 10
    original_melvec_height = 10
    original_window_type = "hamming"
    original_sr = 10200
    original_duration = original_n_melvec * original_Nft / original_sr 

    original_shift_nb = 0
    original_bg_amplitude_limit = []
    original_physical_bg = False

    dtype = original_dtype
    Nft = original_Nft
    samples_per_melvec = original_samples_per_melvec
    n_melvec = original_n_melvec
    melvec_height = original_melvec_height
    window_type = original_window_type
    sr = original_sr

    #Visualisation
    img_idx = 100


    #Augmentations
    #-----------------------------------------------

    #CAREFUL: always respect the order
    augmentations = []
    #augmentations = ["time_shift", "add_bg", "physical_bg"] 

    shift_nb = original_shift_nb                  if "time_shift" in augmentations else 0
    bg_amplitude_limit = original_bg_amplitude_limit if "add_bg" in augmentations else []      #0.1: 20dB, 0.316: 5dB
    physical_bg = original_physical_bg

    plot_indexes = [0,1]
    load_matrix = False


    #CNN architecture
    #----------------------------------------------
    #Tuner
    epochs= 10
    validation_split=0.2
    patience  = 5

    epochs_tuner = 7
    #Validation
    #-----------------------------------------------
    k = 5
    epochs_kfold = 5


    prefix = preprefix
            #1. N MELVEC
    #-----------------------------------------------
    if do_melvec:

        print("N MELVEC")
        print("------------------------------------")
        n_melvec_list = [2,4,8,12,16,20,24,28,32]
        search_hp(dtype, Nft, n_melvec, melvec_height, window_type, sr,   
                  shift_nb, bg_amplitude_limit, physical_bg,     
                  n_melvec_list, name, prefix, hp_name='n_melvec', verbose=verbose)
        n_melvec = original_n_melvec


     #2. SAMPLING RATE
    #-----------------------------------------------
    if do_sr:
        print("SAMPLING RATE")
        print("------------------------------------")
        sr_list= [25000,12000,10200,8400,6800,5100,3400]
        search_hp(dtype, Nft, n_melvec, melvec_height, window_type, sr,     
                  shift_nb, bg_amplitude_limit, physical_bg,   
                  sr_list, name, preprefix, hp_name='sr', verbose=verbose)
        sr = original_sr


    #3. MELVEC HEIGHT
    #-----------------------------------------------
    if do_melvec_height:

        print("MELVEC HEIGHT")
        print("------------------------------------")
        melvec_height_list = [2,4,8,10,16,22,30]
        search_hp(dtype, Nft, n_melvec, melvec_height, window_type, sr,        
                  shift_nb, bg_amplitude_limit, physical_bg,
                  melvec_height_list, name, prefix, hp_name='melvec_height', verbose=verbose)
        melvec_height = original_melvec_height

    #4. WINDOW TYPE
    #-----------------------------------------------
    if do_window_type:

        print("WINDOW TYPE")
        print("------------------------------------")
        window_type_list = ["hamming", "hanning", "blackman", "rectangular", "triangular"]
        search_hp(dtype, Nft, n_melvec, melvec_height, window_type, sr,       
                  shift_nb, bg_amplitude_limit, physical_bg, 
                  window_type_list, name, prefix, hp_name='window_type', verbose=verbose)
        window_type = original_window_type
    
    #5. NFFT
    #-----------------------------------------------
    if do_Nft:

        print("NFFT")
        print("------------------------------------")
        Nft_list = [128, 256, 512, 1024, 2048]
        search_hp(dtype, Nft, n_melvec, melvec_height, window_type, sr,        
                  shift_nb, bg_amplitude_limit, physical_bg,
                  Nft_list, name, prefix, hp_name='Nft', verbose=verbose)
        Nft = original_Nft

    # AUGMENTATION PARAMETERS
    # -----------------------------------------------

    #6. TIME SHIFT
    #-----------------------------------------------
    if do_shifts:
        print("TIME SHIFT")
        print("------------------------------------")
        
        shift_nb_list = [0,1,2,3,4,5,6,7]

        augmentations = ["time_shift"] 
        shift_nb = original_shift_nb                  if "time_shift" in augmentations else 0
                
        search_hp(dtype, Nft, n_melvec, melvec_height, window_type, sr,    
                  shift_nb, bg_amplitude_limit, physical_bg,    
                  shift_nb_list, name, prefix, hp_name='shift_nb', verbose=verbose)
        shift_nb = original_shift_nb


    #8. PHYSICAL BACKGROUND NOISE
    #-----------------------------------------------
    if do_physical_bg:
        print("PHYSICAL BACKGROUND NOISE")
        print("------------------------------------")
        augmentations = ["physical_bg"]
        physical_bg = [False, True]
        search_hp(dtype, Nft, n_melvec, melvec_height, window_type, sr,  
                  shift_nb, bg_amplitude_limit, physical_bg,      
                  physical_bg, name, prefix, hp_name='physical_bg', verbose=verbose)
        physical_bg = original_physical_bg
    


    #7. BACKGROUND NOISE
    #-----------------------------------------------
    if do_bg:
        print("BACKGROUND NOISE")
        print("------------------------------------")
        bg_amplitude_limit_list = [[], [0.1], [0.1, 0.316], [0.1,0.316,0.5], [0.1,0.316,0.5,1]]

        augmentations = ["add_bg"]

        search_hp(dtype, Nft, n_melvec, melvec_height, window_type, sr,     
                  shift_nb, bg_amplitude_limit, physical_bg,   
                  bg_amplitude_limit_list, name, prefix, hp_name='bg_amplitude_limit', verbose=verbose)
        bg_amplitude_limit = original_bg_amplitude_limit
    