import pickle
from pathlib import Path
from typing import Optional
import time
import click
import numpy as np
import matplotlib.pyplot as plt
import requests
import json

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
from classification.utils import payload_to_melvecs
from classification.utils.plots import plot_specgram

from uncertainity_algo import UncertaintyTracker, DoubleRunningSumDetector, decision
from tensorflow.keras.models import load_model

load_dotenv()

# Constants
PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20
DTYPE = np.dtype(np.uint16).newbyteorder("<")


@click.command()
@click.option(
    "-i",
    "--input",
    "_input",
    default="-",
    type=click.File("r"),
    help="Where to read the input stream. Default to '-', a.k.a. stdin.",
)
@click.option(
    "-m",
    "--model",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the trained classification model.",
)
# Added to save the payload in a text file:
@click.option(
    "-s",
    "--save-payloads",
    default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Optional: Path to save valid input payloads for offline testing.",
)
@common.click.melvec_length
@common.click.n_melvecs
@common.click.verbosity
def main(
    _input: Optional[click.File],
    model: Optional[Path],
    melvec_length: int,
    n_melvecs: int,

    #Added to save the payload in a text file:
    save_payloads: Optional[Path],
) -> None:
    """
    Extract Mel vectors from payloads and perform classification on them every 5 seconds.
    Logs predictions to a file with an index.
    """
    if model:
        with open(model, "rb") as file:
            #try:
                model = load_model(model)
                print(f"Model {type(model).__name__} loaded successfully.")
            #except:
            #    model_rf = pickle.load(file)
            #    print(f"Model {type(model_rf).__name__} loaded successfully.")
    else:
        logger.warning("No model provided, skipping classification.")
        return

    msg_counter = 0
    log_file = "predictions_log.txt"

    with open(log_file, "w") as log:
        log.write("Index\tPrediction\n")
        log.write("=" * 30 + "\n")

    class_names = ["chainsaw", "fire", "fireworks", "gunshot"]

    
    # DOUBLE RUNNING SUM
    N_STS = 50
    N_LTS = 50000
    K = 3
    BASELINE = 2048 # Vérifier si nécessaire

    doublesum_detector = DoubleRunningSumDetector(N_STS=N_STS, N_LTS=N_LTS, K=K, BASELINE=BASELINE)

    # MEMORY TRACKER
    window_size = 3
    alpha = 0.7
    reset_interval = 2.5 #s

    memory_tracker = UncertaintyTracker(window_size=window_size, alpha=alpha, reset_interval= reset_interval) #NEED TO CHOOSE THE WINDOW SIZE
    
    #DECISION 
    a_sum = 1
    b_entropy = 1
    c_memory = 0

    for i, payload in enumerate(_input): #loop over input payloads (_input is a list of strings (hexadecimal payloads))
        
        if PRINT_PREFIX in payload: # verfiy if the payload is a valid payload (it must start with PRINT_PREFIX)
            payload = payload[len(PRINT_PREFIX):] # remove the PRINT_PREFIX from the payload to isolate the actual data
            
            # Save the payload to a text file if the option is provided
            payload = payload.strip()

            #if save_payloads:
            #    with open(save_payloads, "a") as save_file: #mode a to append the payload to the file without overwriting it
            #        save_file.write(payload.strip() + "\n") #strip removes leading and trailing whitespaces

            melvec = payload_to_melvecs(payload, melvec_length, n_melvecs) # *32768 to match the original q15_t values
            #logger.info(f"Parsed payload into Mel vectors: {melvec}")

            msg_counter += 1
            print(f"MEL Spectrogram #{msg_counter}")

            # Reshape and normalize feature vector
            fv = melvec.reshape(1, -1)
            fv = fv / np.linalg.norm(fv)

            # Make prediction
            #pred = model_rf.predict(fv)
            #proba = model_rf.predict_proba(fv)
            #predicted_class = pred[0]

            # Make prediction
            probas = model.predict(fv)
            predicted_class = np.argmax(probas, axis=1)[0]
            predicted_class_name = class_names[predicted_class]

            # Uncertainty calculation

            send_packet = decision(payload, probas, i, DTYPE, predicted_class, doublesum_detector, memory_tracker, 
                                   a_sum=a_sum, b_entropy=b_entropy, c_memory=c_memory, 
                                   save_payloads=save_payloads)

            #send_packet = True
            # Log prediction
            #log.write(f"{msg_counter}\t{predicted_class}\n")

            # Optional: Visualization
            
            #class_names = model_rf.classes_
            probabilities = np.round(probas[0] * 100, 2)
            max_len = max(len(name) for name in class_names)
            class_names_str = " ".join([f"{name:<{max_len}}" for name in class_names])
            probabilities_str = " ".join([f"{prob:.2f}%".ljust(max_len) for prob in probabilities])
            textlabel = f"{class_names_str}\n{probabilities_str}"
            textlabel += f"\n\Theoretical class: {predicted_class_name}"
            textlabel += f""

            
            plot_specgram(
                melvec.reshape((N_MELVECS, MELVEC_LENGTH)),
                ax=plt.gca(),
                is_mel=True,
                title=f"MEL Spectrogram #{msg_counter}",
                xlabel="Mel vector",
                textlabel=textlabel,
            )
            #plt.show()
            #Save the plot
            #plt.savefig(f"demo_mels/demo_round1_{msg_counter}.png")
            #plt.savefig(f"/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/recorded_mels/mel_{msg_counter}.png")
            plt.draw()
            plt.pause(0.1)
            
            plt.clf()
            

            if send_packet:
                print("SEND TRUE: ", predicted_class_name)
                #hostname = "http://localhost:5000"
                #key = "v_JARGcUgZRmCgToaK9Y-uZgmCYNBSgekgQk21cm"
                hostname = "http://lelec210x.sipr.ucl.ac.be"
                key = "c6yuOmUKKbKAS4lRa7vQ9clME3bGTSWUvS4N26af"
                guess = predicted_class_name
                #time.sleep(4)
                response = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{key}/{guess}", timeout=1)
                

                # N.B.: the timeout is generally a good idea to avoid blocking infinitely (if an error occurs)
                # but you can change its value. Note a too small value may not give the server enough time
                # to reply.

                # All responses are JSON dictionaries
                #response_as_dict = json.loads(response.text)

            # Wait for 5 seconds before processing the next prediction
            #time.sleep(5)
            else:
                print("SEND FALSE: ", predicted_class_name)



if __name__ == "__main__":
    main()