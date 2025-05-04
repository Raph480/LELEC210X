from CNN_utils import *
import pandas as pd
import keras_tuner as kt
from sklearn.preprocessing import LabelEncoder

tf.config.run_functions_eagerly(True)  # Force eager execution

from tensorflow.keras.utils import to_categorical
import numpy as np


#D. Self-made 1
def self_made_builder_factory(input_shape):
    #Enable to pass input_shape to the function
    def self_made_builder(hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape = input_shape))
        model.add(tf.keras.layers.Flatten())

        hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])

        for i in range(hp.Int("num_layers", 1,3)):
            model.add(
                tf.keras.layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"layer_{i}", min_value=32, max_value=512, step=32),
                    activation=hp_activation,
                )
            )
            ## Add dropout after each dense layer
            #model.add(
            #    tf.keras.layers.Dropout(
            #        rate=hp.Float(f"dropout_rate_{i}", min_value=0.1, max_value=0.5, step=0.1)
            #    )
            #)
        #hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=512, step=32)
        #hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=512, step=32)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) 

        #model.add(tf.keras.layers.Dense(units=hp_layer_1, activation=hp_activation))
        #model.add(tf.keras.layers.Dense(units=hp_layer_2, activation=hp_activation))
        model.add(tf.keras.layers.Dense(4, activation='softmax'))

        opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
    return self_made_builder




def model_creation_evaluation(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr, shift_nb=0,
        bg_amplitude_limit=[], physical_bg=False, augmentations = [], CNN_dataset = False, load_matrix = False, load_matrix_test = False):

    print("NEW CREATION EVALUATION")
    print("------------------------------------")
    print("Physical HP: ")
    print("flag_8bit,: ", flag_8bit)
    print("bit_sensitivity: ", bit_sensitivity)
    print("Nft: ", Nft)
    print("n_melvec: ", n_melvec)
    print("melvec_height: ", melvec_height)
    print("window_type: ", window_type)
    print("sampling rate: ", sr)
    img_idx = 0
    test_img_idx = 0
    verbose = False
    plot_indexes = []
    play_sound = False
    samples_per_melvec = Nft

    #1. Import classical  - #2. Train-test split
    #--------------------------------
    #Does not take the background noise and youtube sounds into account
    myds, dataset, classnames = get_dataset(path="../datasets/sounds/recorded_sounds/trainset_with_new/", filter_str=None,
        Nft=Nft, n_melvec=n_melvec, melvec_height=melvec_height, samples_per_melvec=samples_per_melvec,
        window_type=window_type, sr = sr, flag_8bit = flag_8bit, bit_sensitivity=bit_sensitivity,
        normalize=True, shift_pct=0, verbose=False, img_idx = img_idx, play_sound=True, CNN_dataset = CNN_dataset)

    myds_test, dataset_test, _ = get_dataset(path="../datasets/sounds/recorded_sounds/testset_with_new/", filter_str=None, #"_background_",
        Nft=Nft, n_melvec=n_melvec, melvec_height=melvec_height, samples_per_melvec=samples_per_melvec,
        window_type=window_type, sr = sr,  flag_8bit = flag_8bit, bit_sensitivity=bit_sensitivity,
        normalize=True, shift_pct=0, verbose=False, img_idx = test_img_idx, play_sound=False, CNN_dataset = CNN_dataset)


    #3. Dataset augmentations
    #----------------------------
    print("\nDataset Augmentations")
    if "add_bg" in augmentations:
        bg_dataset = Dataset(folder="../datasets/sounds/recorded_sounds/background/", filter_str=None)
    else :
        bg_dataset = None

    if  physical_bg:
        my_phy_ds, phy_bg_dataset, classnames = get_dataset(path="../datasets/sounds/recorded_sounds/totalset_with_new/", filter_str="_background_",
        Nft=Nft, n_melvec=n_melvec, melvec_height=melvec_height, samples_per_melvec=samples_per_melvec,
        window_type=window_type, sr = sr,  flag_8bit = flag_8bit, bit_sensitivity=bit_sensitivity,
        normalize=True, shift_pct=0, verbose=False, img_idx = img_idx, play_sound=False, CNN_dataset = CNN_dataset)
    else :
        my_phy_ds = None
        phy_bg_dataset = None

    physical_aug = ("physical_bg" in augmentations) or physical_bg


    pickle_name = get_picklename( flag_8bit, bit_sensitivity,Nft,samples_per_melvec,n_melvec,melvec_height,window_type,sr,augmentations,shift_nb,bg_amplitude_limit,physical_aug,
        prefix="../datasets/melvecs/HP_tuning/",
        purpose = "melvecs" # "model"
    )
    #pickle_name+="_1_"
    if CNN_dataset:
        pickle_name+="_2D_"
    print("pickle_name: ", pickle_name)

    #---------------------------------

    #Visualisation purposes
    img_idx = 10
    print("TRAIN SET")
    X_train, y_train = augment_dataset(myds, dataset, classnames, augmentations, n_melvec = n_melvec, melvec_height=melvec_height,
                        shift_nb = shift_nb, #numbers of shifts done
                        bg_dataset = bg_dataset, bg_amplitude_limit=bg_amplitude_limit, #dataset used for background noise, background amplitudes
                        physical_bg_dataset = phy_bg_dataset,my_phy_ds = my_phy_ds, #dataset used for physical background noises
                        verbose=verbose, img_idx=img_idx, aug_indexes=plot_indexes, play_sound=play_sound, #verbose parameters
                        load_matrix=load_matrix, pickle_name=pickle_name, CNN_dataset = CNN_dataset) #load and save parameters


    print("TEST SET")
    plot_indexes = [0]
    verbose = False
    play_sound = False


    augmentations_test = [] 
    nb_shift_test = 0

    pickle_name_test = get_picklename(
        flag_8bit, bit_sensitivity,Nft,samples_per_melvec,n_melvec,melvec_height,window_type,sr,augmentations_test,nb_shift_test,bg_amplitude_limit,physical_aug,
        prefix="../datasets/melvecs/HP_tuning/",
        purpose = "melvecs_test" # "model"
    )
    #pickle_name_test+="_1_"
    if CNN_dataset:
        pickle_name_test+="_2D_"
    print("pickle_name_test: ", pickle_name_test)


    X_test, y_test = augment_dataset(myds_test, dataset_test, classnames, augmentations = augmentations_test
                                    , n_melvec=n_melvec, melvec_height=melvec_height,
                                    
                    shift_nb = nb_shift_test, #numbers of shifts done
                    bg_dataset = None, bg_amplitude_limit=[], #dataset used for background noise, background amplitudes
                    physical_bg_dataset = phy_bg_dataset,my_phy_ds = my_phy_ds, #dataset used for physical background noises
                    verbose=verbose, img_idx=img_idx, aug_indexes=plot_indexes, play_sound=play_sound, #verbose parameters
                load_matrix=load_matrix_test, pickle_name=pickle_name_test, CNN_dataset = CNN_dataset) #load and save parameters

    #Print shape of the test set
    print("X_test shape: ", X_test.shape)

    # Transform the labels to integers and save mapping
    label_to_id = {label: i for i, label in enumerate(classnames)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    y_train = np.array([label_to_id[label] for label in y_train])
    y_test = np.array([label_to_id[label] for label in y_test])


    #Shuffle X_train and y_train
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Identify invalid samples (all zeros or NaNs)
    invalid_mask = np.any(np.isnan(X_train), axis=tuple(range(1, X_train.ndim))) | \
                np.all(X_train == 0, axis=tuple(range(1, X_train.ndim)))

    # Invert the mask to get valid entries
    valid_mask = ~invalid_mask

    # Filter X_train and y_train
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]

    print(f"Removed {np.sum(invalid_mask)} invalid samples!")
    print(f"X_train shape after removing invalid samples: {X_train.shape}")



    #4. Modelling
    #----------------------------
    #label_encoder = LabelEncoder()
    #y_train = label_encoder.fit_transform(y_train)  # Convert labels to numbers
    validation_split = 0.2

    if CNN_dataset:
        def self_made_builder_factory(input_shape):
            def self_made_builder(hp):
                try:
                    model = tf.keras.Sequential()
                    model.add(tf.keras.Input(shape=input_shape))  # Ensure this is (height, width, channels)
                    model.add(tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1)))

                    num_conv_layers = hp.Int("num_conv_layers", 1, 4)

                    for i in range(num_conv_layers):
                        kernel_size = hp.Choice(f"kernel_size_{i}", values=[3, 5, 7, 9])
                        pool_size = hp.Choice(f"pool_size_{i}", values=[2, 3, 4])

                        model.add(
                            tf.keras.layers.Conv2D(
                                filters=hp.Int(f"filters_{i}", min_value=16, max_value=256, step=16),
                                kernel_size=(kernel_size, kernel_size),
                                activation=hp.Choice(f"activation_{i}", values=['relu', 'tanh', 'sigmoid']),
                            )
                        )
                        model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size)))
                        model.add(tf.keras.layers.BatchNormalization())
                    
                    # After convolutional layers, flatten the output to feed into dense layers
                    model.add(tf.keras.layers.Flatten())

                    # Dense layers
                    for i in range(hp.Int("num_dense_layers", 1, 3)):  # Flexible Dense layers
                        model.add(
                            tf.keras.layers.Dense(
                                units=hp.Int(f"dense_layer_{i}", min_value=32, max_value=256, step=32),
                                activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']),
                                kernel_regularizer=tf.keras.regularizers.l2(0.01)
                            )
                        )
                        model.add(tf.keras.layers.Dropout(rate=hp.Float(f"dropout_rate_{i}", min_value=0.1, max_value=0.6, step=0.1)))

                    model.add(tf.keras.layers.Dense(4, activation='softmax'))

                    # Learning rate and optimizer
                    hp_learning_rate = hp.Choice('learning_rate', values=[0.5e-1,1e-2, 1e-3, 1e-4])
                    opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

                    model.compile(optimizer=opt,
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
                    return model
                except Exception as e:
                    print("\n[ERROR] Model creation failed.")
                    print(f"Reason: {e}")
                    print("Full traceback:")

                    # Create a terrible dummy model to ensure very low accuracy
                    bad_model = tf.keras.Sequential([
                        tf.keras.Input(shape=input_shape),
                        tf.keras.layers.Lambda(lambda x: tf.zeros_like(x)),  # Kills all features
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(4, activation='softmax',
                                            kernel_initializer='zeros',
                                            bias_initializer='zeros')
                    ])
                    bad_model.compile(
                        optimizer='sgd',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    return bad_model
            return self_made_builder
    else:      
        #D. Self-made 1
        def self_made_builder_factory(input_shape):
            #Enable to pass input_shape to the function
            def self_made_builder(hp):
                
                model = tf.keras.Sequential()
                model.add(tf.keras.Input(shape = input_shape))
                model.add(tf.keras.layers.Flatten())

                hp_activation = hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])

                for i in range(hp.Int("num_layers", 1,4)):
                    model.add(
                        tf.keras.layers.Dense(
                            # Tune number of units separately.
                            units=hp.Int(f"layer_{i}", min_value=1, max_value=526, step=32),
                            activation=hp_activation,
                        )
                    )
                    ## Add dropout after each dense layer
                    model.add(
                        tf.keras.layers.Dropout(
                            rate=hp.Float(f"dropout_rate_{i}", min_value=0.1, max_value=0.5, step=0.1)
                        )
                    )
                #hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=512, step=32)
                #hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=512, step=32)
                hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) 

                #model.add(tf.keras.layers.Dense(units=hp_layer_1, activation=hp_activation))
                #model.add(tf.keras.layers.Dense(units=hp_layer_2, activation=hp_activation))
                model.add(tf.keras.layers.Dense(4, activation='softmax'))

                opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

                model.compile(optimizer=opt,
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

                return model
            return self_made_builder

    if CNN_dataset:
        project_name = "basic_ref_model_2D_20melvecs_gridseach_sat"
    else:
        project_name = "basic_ref_model"
        #project_name = "test_opti"


    def create_tuner(hypermodel):
        return kt.Hyperband(hypermodel,
                        objective='val_accuracy',
                        max_epochs=epochs_tuner,
                        factor=3,
                        directory='hp_dir',
                        project_name=project_name,
                        overwrite= False)

    input_shape = X_train[0].shape 
    print("input shape: ", input_shape)

    builder = self_made_builder_factory(input_shape)
    tuner = create_tuner(builder)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    tuner.search(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[stop_early])

    #5. Final model
    validation_split = 0.3
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)


    model_name = get_picklename(
        flag_8bit, bit_sensitivity,Nft,samples_per_melvec,n_melvec,melvec_height,window_type,sr,augmentations,shift_nb,
        bg_amplitude_limit,physical_aug,
        prefix="../datasets/models/",
        purpose = "model"
    )

    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    print("Best hyperparameters: ", best_hps.values)

    #Temporary model to find the best number of epochs - avoid overfitting
    temp_model = tuner.hypermodel.build(best_hps)
    history = temp_model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[stop_early])
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    #Final results
    #------------------------------
    #Remove tensorflow warnings
    tf.data.experimental.enable_debug_mode()

    #Train the model with the best hyperparameters and best epoch
    hypermodel = tuner.hypermodel.build(best_hps)
    hp_learning_rate = best_hps.get('learning_rate')
    opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    hypermodel.compile(optimizer=opt,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=validation_split) #validation_split=validation_split)
    k = 5
    epochs_kfold = 20
    save_model = False

    #Clone the model for k-fold
    kfold_model = tf.keras.models.clone_model(hypermodel)
    #Get the best tuning rate from the best hyperparameters
    hp_learning_rate = best_hps.get('learning_rate')
    opt = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    kfold_model.compile(optimizer=opt,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])


    # Make sure to recompile
    # Train of the model & evaluation -  K-Fold cross-validation
    kfold_acc, kfold_recall, kfold_f1 = perform_kfold(kfold_model, X_train, y_train, k=k, epochs=epochs_kfold)




    #Train- Test evaluation
    train_confmat, train_report, test_accuracy, test_confmat, test_report= evaluate_model(hypermodel, 
                            X_train, y_train, classnames, X_test, y_test,show_confusion=False)


    if (X_test is not None):
        predictions = hypermodel.predict(X_test)
        eval_result = hypermodel.evaluate(X_test, y_test)
        print("[test loss, test accuracy]:", eval_result)


    return kfold_acc, kfold_recall, kfold_f1, train_confmat, train_report, test_accuracy, test_confmat, test_report


def search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr, 
              shift_nb, bg_amplitude_limit, physical_bg,      
        hp_list, name, preprefix, hp_name, verbose=False, augmentations = []):
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
        local_flag_8bit = flag_8bit
        local_bit_sensitivity = bit_sensitivity

        local_shift_nb = shift_nb
        local_bg_amplitude_limit = bg_amplitude_limit
        local_physical_bg = physical_bg

        if hp_name == 'n_melvec':
            local_n_melvec = hp_value
        elif hp_name == 'bit_sensitivity':
            local_flag_8bit = True
            local_bit_sensitivity = hp_value
            print("Bit Sensitivity: ", bit_sensitivity)
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
            local_shift_nb = 4
            print("Background amplitude limit: ", bg_amplitude_limit)
        elif hp_name == 'physical_bg':
            local_physical_bg = hp_value
            print("Physical background noise: ", physical_bg)

        else:
            raise ValueError(f"Unsupported hyperparameter name: {hp_name}")

        print(f"Testing {hp_name} = {hp_value}")

        individual_time_start = time.time()
        kfold_acc, kfold_recall, kfold_f1, train_confmat, train_report, \
            test_accuracy, test_confmat, test_report = model_creation_evaluation(local_flag_8bit, local_bit_sensitivity, local_Nft, local_n_melvec, local_melvec_height, local_window_type, local_sr,
                                                                                 local_shift_nb, local_bg_amplitude_limit, local_physical_bg, augmentations = augmentations, CNN_dataset=CNN_dataset,
                                                                                 load_matrix = load_matrix, load_matrix_test = load_matrix_test)
    
        
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

import itertools
import json
from datetime import datetime
import os


def exhaustive_grid_search(hyperparams_grid, sr_dic, sr_lim_dic, log_dir="logs", CNN_dataset=False):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_log_file = os.path.join(log_dir, f"grid_search_raw_{timestamp}.log")
    sorted_log_file = os.path.join(log_dir, f"grid_search_sorted_{timestamp}.log")

    # Generate all hyperparameter combinations
    keys = list(hyperparams_grid.keys())
    values = list(hyperparams_grid.values())
    combinations = list(itertools.product(*values))

    #Fixed parameters
    flag_8bit = True
    Nft = 512

    #Augmentations parameters
    #augmentations = []
    augmentations = ["time_shift"]
    shift_nb = 3 if "time_shift" in augmentations else 0
    bg_amplitude_limit = [] if "add_bg" in augmentations else []
    physical_bg = False if "physical_bg" in augmentations else False

    results = []

    with open(raw_log_file, 'w', buffering=1) as raw_log:
        for combo in combinations:
            hyperparams = dict(zip(keys, combo))
            #try:
            # Unpack hyperparameters
            window_type = hyperparams["window_type"]
            n_melvec = hyperparams["n_melvec"]
            melvec_height = hyperparams["melvec_height"]
            bit_sensitivity = hyperparams["bit_sensitivity"]

            #sr is defined from n_melvec
            sr = 512 * n_melvec / 1.3
            #sr = sr_dic[n_melvec]
            #sr_limit = sr_lim_dic[melvec_height]
            sr_limit = 9217
            if sr > sr_limit:
                print(f"Skipping combination due to sr limit: {sr} > {sr_limit}")
                kfold_acc, test_acc = 0, 0
                continue

            kfold_acc, _, _, _, _, test_acc, _, _ = model_creation_evaluation(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr, 
                                                                                shift_nb, bg_amplitude_limit, physical_bg, augmentations, CNN_dataset,
                                                                                load_matrix = load_matrix, load_matrix_test = load_matrix_test)
            metrics = {
                "kfold_acc": kfold_acc,
                "test_accuracy": test_acc,
            }
            #except Exception as e:
            #    #get the name of the exception
            #    print(f"Error occurred for hyperparameters {hyperparams}: {e}")
            #    metrics = {"error": str(e)}
            result = {
                "hyperparameters": hyperparams,
                "results": metrics
            }
            results.append(result)
            raw_log.write(json.dumps(result) + "\n")
            raw_log.flush()  # Ensure the log is written immediately

    # Filter out failed runs
    successful_results = [r for r in results if "error" not in r["results"]]

    # Sort by training score (assuming higher is better)
    sorted_results = sorted(successful_results, key=lambda x: x["results"].get("kfold_acc", 0), reverse=True)

    with open(sorted_log_file, 'w') as sorted_log:
        for result in sorted_results:
            sorted_log.write(json.dumps(result) + "\n")

    print(f"Grid search completed. Raw log: {raw_log_file}, Sorted log: {sorted_log_file}")


if __name__ == '__main__':

    do_exhaustive_grid_search = False
    do_idv_grid_search = True
    do_mean_results = False

    CNN_dataset = True

    #EXHAUSTIVE GRID SEARCH
    #------------------------------------
    if do_exhaustive_grid_search:
        preprefix = "../datasets/exhaustiveGSresults/"
        hyperparams_grid = {
        "window_type": ["triangular","hanning", "rectangular", "hamming", "blackman"],
        "n_melvec": [10],
        "melvec_height": [20,25],
        "bit_sensitivity": [4,5],
        }

        #sr_dic = {5: 1969,10: 3938,15: 5908,20: 7877, 25: 9318}
        #sr_lim_dic = {5: 9619,10:9552,15:9487,20:9408,25:9318,30:9217}
        load_matrix = False
        load_matrix_test = False
        verbose = False
        img_idx = 1
        plot_indexes = []
        epochs = 30
        epochs_tuner = 8
        patience = 3
        exhaustive_grid_search(hyperparams_grid, None, None, log_dir=preprefix, CNN_dataset=CNN_dataset)

    #MEAN RESULTS OF INDIVIDUAL GRID SEARCH
    #--------------------------------------------------
    if do_mean_results:
        mean_name = "MEAN_friday25"
        HPs = ["bit_sensitivity", "melvec_height", "n_melvec", "Nft", "physical_bg", "shift_nb", "sr", "window_type"]
        #for HP in HPs:
        #    plot_mean_results(HP, mean_name, show=True)
        plot_mean_results("shift_nb", mean_name, show=True)

    #INDIVIDUAL GRID SEARCH CODE
    #---------------------------------------------------
    if do_idv_grid_search == False:
        exit()

    do_bit_sensitivity = False
    do_melvec = False
    do_sr = False
    do_melvec_height = False
    do_window_type = False
    do_Nft = False

    do_shifts = True
    do_bg = True
    do_physical_bg = True


    verbose = False #To show graphs
    preprefix = "../datasets/GSresults/"
    original_name = "sat03_CNN_results_"

    #Physical HP
    #------------------------------------
    original_flag_8bit = True
    original_bit_sensitivity = 5
    original_Nft = 512
    original_samples_per_melvec = original_Nft
    original_n_melvec = 20
    original_melvec_height = 10
    original_window_type = "hamming"
    original_sr =  512 * original_n_melvec / 1.3
    #original_sr = 10200
    original_duration = original_n_melvec * original_Nft / original_sr 

    original_shift_nb = 3
    original_bg_amplitude_limit = []
    original_physical_bg = False


    flag_8bit = original_flag_8bit
    bit_sensitivity = original_bit_sensitivity
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
    augmentations = ["time_shift"]
    #augmentations = ["time_shift", "add_bg", "physical_bg"] 

    shift_nb = original_shift_nb                  if "time_shift" in augmentations else 0
    bg_amplitude_limit = original_bg_amplitude_limit if "add_bg" in augmentations else []      #0.1: 20dB, 0.316: 5dB
    physical_bg = original_physical_bg

    plot_indexes = [0,1]
    load_matrix = False
    load_matrix_test = False


    #CNN architecture
    #----------------------------------------------
    #Tuner
    epochs= 50
    validation_split=0.2
    patience  = 10

    epochs_tuner = 8
    #Validation
    #-----------------------------------------------
    k = 5
    epochs_kfold = 5


    prefix = preprefix
    name = original_name
    
    #0  BIT SENSITIVITY
    #-----------------------------------------------
    if do_bit_sensitivity:
        print("BIT SENSITIVITY")
        print("------------------------------------")
        bit_sensitivity_list = [0,1,2,3,4,5,6,7,8]
        flag_8bit = True
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,
                  shift_nb, bg_amplitude_limit, physical_bg,
                  bit_sensitivity_list, name, prefix, hp_name='bit_sensitivity', verbose=verbose, augmentations=augmentations)
        bit_sensitivity = original_bit_sensitivity
        flag_8bit = original_flag_8bit

        name  = original_name + "_2_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,
                  shift_nb, bg_amplitude_limit, physical_bg,
                  bit_sensitivity_list, name, prefix, hp_name='bit_sensitivity', verbose=verbose, augmentations=augmentations)
        bit_sensitivity = original_bit_sensitivity
        flag_8bit = original_flag_8bit

        name = original_name + "_3_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,
                  shift_nb, bg_amplitude_limit, physical_bg,
                  bit_sensitivity_list, name, prefix, hp_name='bit_sensitivity', verbose=verbose, augmentations=augmentations)
        bit_sensitivity = original_bit_sensitivity
        flag_8bit = original_flag_8bit
        name = original_name
    
    #1. N MELVEC
    #-----------------------------------------------
    if do_melvec:

        print("N MELVEC")
        print("------------------------------------")
        n_melvec_list = [2,4,8,12,16,20,24,28,32]
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,   
                  shift_nb, bg_amplitude_limit, physical_bg,     
                  n_melvec_list, name, prefix, hp_name='n_melvec', verbose=verbose, augmentations=augmentations)
        n_melvec = original_n_melvec

        name = original_name + "_2_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,   
                  shift_nb, bg_amplitude_limit, physical_bg,     
                  n_melvec_list, name, prefix, hp_name='n_melvec', verbose=verbose, augmentations=augmentations)
        n_melvec = original_n_melvec

        name = original_name + "_3_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,   
                  shift_nb, bg_amplitude_limit, physical_bg,     
                  n_melvec_list, name, prefix, hp_name='n_melvec', verbose=verbose, augmentations=augmentations)
        n_melvec = original_n_melvec
        name = original_name


     #2. SAMPLING RATE
    #-----------------------------------------------
    if do_sr:
        print("SAMPLING RATE")
        print("------------------------------------")
        sr_list= [25000,12000,10200,8400,6800,5100,3400]
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,     
                  shift_nb, bg_amplitude_limit, physical_bg,   
                  sr_list, name, preprefix, hp_name='sr', verbose=verbose, augmentations=augmentations)
        sr = original_sr
        name = original_name + "_2_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,     
                  shift_nb, bg_amplitude_limit, physical_bg,   
                  sr_list, name, preprefix, hp_name='sr', verbose=verbose, augmentations=augmentations)
        sr = original_sr
        name = original_name + "_3_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,     
                  shift_nb, bg_amplitude_limit, physical_bg,   
                  sr_list, name, preprefix, hp_name='sr', verbose=verbose, augmentations=augmentations)
        sr = original_sr
        name = original_name


    #3. MELVEC HEIGHT
    #-----------------------------------------------
    if do_melvec_height:

        print("MELVEC HEIGHT")
        print("------------------------------------")
        melvec_height_list = [2,4,8,10,16,22,30]
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,        
                  shift_nb, bg_amplitude_limit, physical_bg,
                  melvec_height_list, name, prefix, hp_name='melvec_height', verbose=verbose, augmentations=augmentations)
        melvec_height = original_melvec_height

        name = original_name + "_2_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,        
                  shift_nb, bg_amplitude_limit, physical_bg,
                  melvec_height_list, name, prefix, hp_name='melvec_height', verbose=verbose, augmentations=augmentations)
        melvec_height = original_melvec_height

        name = original_name + "_3_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,        
                  shift_nb, bg_amplitude_limit, physical_bg,
                  melvec_height_list, name, prefix, hp_name='melvec_height', verbose=verbose, augmentations=augmentations)
        melvec_height = original_melvec_height
        name = original_name


    #4. WINDOW TYPE
    #-----------------------------------------------
    if do_window_type:

        print("WINDOW TYPE")
        print("------------------------------------")
        window_type_list = ["hamming", "hanning", "blackman", "rectangular", "triangular"]
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,       
                  shift_nb, bg_amplitude_limit, physical_bg, 
                  window_type_list, name, prefix, hp_name='window_type', verbose=verbose, augmentations=augmentations)
        window_type = original_window_type

        name = original_name + "_2_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,       
                  shift_nb, bg_amplitude_limit, physical_bg, 
                  window_type_list, name, prefix, hp_name='window_type', verbose=verbose, augmentations=augmentations)
        window_type = original_window_type

        name = original_name + "_3_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,       
                  shift_nb, bg_amplitude_limit, physical_bg, 
                  window_type_list, name, prefix, hp_name='window_type', verbose=verbose, augmentations=augmentations)
        window_type = original_window_type
        name = original_name

    
    #5. NFFT
    #-----------------------------------------------
    if do_Nft:

        print("NFFT")
        print("------------------------------------")
        Nft_list = [128, 256, 512, 1024, 2048]
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,        
                  shift_nb, bg_amplitude_limit, physical_bg,
                  Nft_list, name, prefix, hp_name='Nft', verbose=verbose, augmentations=augmentations)
        Nft = original_Nft
        name = original_name + "_2_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,        
                  shift_nb, bg_amplitude_limit, physical_bg,
                  Nft_list, name, prefix, hp_name='Nft', verbose=verbose, augmentations=augmentations)
        Nft = original_Nft
        name = original_name + "_3_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,        
                  shift_nb, bg_amplitude_limit, physical_bg,
                  Nft_list, name, prefix, hp_name='Nft', verbose=verbose, augmentations=augmentations)
        Nft = original_Nft
        name = original_name


    # AUGMENTATION PARAMETERS
    # -----------------------------------------------

        #8. PHYSICAL BACKGROUND NOISE
    #-----------------------------------------------
    if do_physical_bg:
        print("PHYSICAL BACKGROUND NOISE")
        print("------------------------------------")
        augmentations = ["time_shift", "physical_bg"]
        shift_nb = 2
        physical_bg = [False, True]
        name = original_name + "_2_"

        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,  
                  shift_nb, bg_amplitude_limit, physical_bg,      
                  physical_bg, name, prefix, hp_name='physical_bg', verbose=verbose, augmentations=augmentations)

        name = original_name + "_3_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,  
                  shift_nb, bg_amplitude_limit, physical_bg,      
                  physical_bg, name, prefix, hp_name='physical_bg', verbose=verbose, augmentations=augmentations)

        name = original_name + "_4_"
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,  
                  shift_nb, bg_amplitude_limit, physical_bg,      
                  physical_bg, name, prefix, hp_name='physical_bg', verbose=verbose, augmentations=augmentations)
        physical_bg = original_physical_bg
        name = original_name    


    #6. TIME SHIFT
    #-----------------------------------------------
    if do_shifts:
        print("TIME SHIFT")
        print("------------------------------------")
        
        shift_nb_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        #shift_nb_list = [13,14,15]
        original_shift_nb = 0


        augmentations = ["time_shift"] 
        shift_nb = original_shift_nb                  if "time_shift" in augmentations else 0
                
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,    
                  shift_nb, bg_amplitude_limit, physical_bg,    
                  shift_nb_list, name, prefix, hp_name='shift_nb', verbose=verbose, augmentations=augmentations)
        shift_nb = original_shift_nb


    #7. BACKGROUND NOISE
    #-----------------------------------------------
    if do_bg:
        print("BACKGROUND NOISE")
        print("------------------------------------")
        shift_nb = 3
        bg_amplitude_limit_list = [[], [0.1], [0.316], [0.1,0.316], [0.1,0.1,0.316,0.316]]
        #bg_amplitude_limit_list = [[], [0.1], [0.1,0.1], [0.1,0.1,0.1], [0.1,0.1,0.1,0.1], [0.1,0.1,0.1,0.1,0.1]]
        #bg_amplitude_limit_list = [[], [0.316], [0.316,0.316], [0.316,0.316,0.316], [0.316,0.316,0.316,0.316], [0.316,0.316,0.316,0.316,0.316]]
        #bg_amplitude_limit_list = [[], 
        #                           [0.1, 0.316], 
        #                           [0.1,0.1,0.316,0.316]]
                                    #[0.1, 0.1, 0.1, 0.316,0.316,0.316],
                                   #[0.1, 0.1, 0.1, 0.1, 0.316,0.316,0.316,0.316]]
        augmentations = ["time_shift","add_bg"]

        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,     
                  shift_nb, bg_amplitude_limit, physical_bg,   
                  bg_amplitude_limit_list, name, prefix, hp_name='bg_amplitude_limit', verbose=verbose, augmentations=augmentations)
        bg_amplitude_limit = original_bg_amplitude_limit

        name = original_name + "_2_"
        bg_amplitude_limit_list = [[], [0.1], [0.1,0.1], [0.1,0.1,0.1], [0.1,0.1,0.1,0.1], [0.1,0.1,0.1,0.1,0.1]]
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,     
                  shift_nb, bg_amplitude_limit, physical_bg,   
                  bg_amplitude_limit_list, name, prefix, hp_name='bg_amplitude_limit', verbose=verbose, augmentations=augmentations)
        bg_amplitude_limit = original_bg_amplitude_limit

        name = original_name + "_3_"
        bg_amplitude_limit_list = [[], [0.316], [0.316,0.316], [0.316,0.316,0.316], [0.316,0.316,0.316,0.316], [0.316,0.316,0.316,0.316,0.316]]
        search_hp(flag_8bit, bit_sensitivity, Nft, n_melvec, melvec_height, window_type, sr,     
                  shift_nb, bg_amplitude_limit, physical_bg,   
                  bg_amplitude_limit_list, name, prefix, hp_name='bg_amplitude_limit', verbose=verbose, augmentations=augmentations)
        bg_amplitude_limit = original_bg_amplitude_limit
        name = original_name
