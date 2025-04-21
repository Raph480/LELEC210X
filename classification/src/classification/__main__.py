import pickle
from pathlib import Path
from typing import Optional
import time
import click
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import threading

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
from classification.utils import payload_to_melvecs
from classification.utils.plots import plot_specgram
np.set_printoptions(precision=2, suppress=True)


from uncertainity_algo import DoubleRunningSumDetector, compute_uncertainity
from tensorflow.keras.models import load_model

load_dotenv()

PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_HEIGHT = 20
N_MELVECS = 20
DTYPE = np.dtype(np.uint16).newbyteorder("<")


import sys

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@click.command()
@click.option(
    "-i", "--input", "_input", default="-", type=click.File("r"),
    help="Where to read the input stream. Default to '-', a.k.a. stdin.",
)
@click.option(
    "-m", "--model_1", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the trained classification model.",
)
@click.option(
    "-m2", "--model_2", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the second model."
)
@click.option(
    "-m3", "--model_3", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the third model."
)
@click.option(
    "-s", "--save-payloads", default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Optional: Path to save valid input payloads for offline testing.",
)
@click.option(
    "-L", "--log-file", default=None,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Optional: Path to a file where all stdout (print) will be written.",
)

@common.click.melvec_length
@common.click.n_melvecs
@common.click.verbosity
def main(
    _input: Optional[click.File],
    model_1: Optional[Path],
    model_2: Optional[Path],
    model_3: Optional[Path],
    save_payloads: Optional[Path],
    log_file: Optional[Path],
    melvec_length: int,
    n_melvecs: int,
) -> None:
    models = {}

    if model_1:
        with open(model_1, "rb") as file:
            model_1 = load_model(model_1)
            print(f"Model {type(model_1).__name__} loaded successfully.")
    else:
        logger.warning("Model 1 not provided.")
    
    if model_2:
        with open(model_2, "rb") as file:
            model_2 = load_model(model_2)
            print(f"Model 2 {type(model_2).__name__} loaded successfully.")
    else:
        logger.warning("Model 2 not provided.")

    if model_3:
        with open(model_3, "rb") as file:
            model_3 = load_model(model_3)
            print(f"Model 3 {type(model_3).__name__} loaded successfully.")
    else:
        logger.warning("Model 3 not provided.")
        
    if log_file:
        log_stream = open(log_file, "w")
        sys.stdout = Tee(sys.__stdout__, log_stream)
    
    msg_counter = 0
    log_file = "predictions_log.txt"
    with open(log_file, "w") as log:
        log.write("Index\tPrediction\n")
        log.write("=" * 30 + "\n")

    class_names = ["chainsaw", "fire", "fireworks", "gunshot"]

    up_time = 3.5 #s
    min_time_between_listening = 6 #s


    N_STS = 50
    N_LTS = 50000
    K = 3
    BASELINE = 2048

    doublesum_detector = DoubleRunningSumDetector(N_STS=N_STS, N_LTS=N_LTS, K=K, BASELINE=BASELINE)

    listening = False
    last_activation_time = 0

        
    def wait_take_decision_and_send(fv_history,payloads_history, doublesum_detector, idx, up_time=3, model_1=None, model_2=None, model_3=None):

        class_names = ["chainsaw", "fire", "fireworks", "gunshot"]
        #1. Listening time: wiate for up_time s to recieve packets
        #----------------------
        time.sleep(up_time)
        nonlocal listening
        listening = False


        #2. Decision time: take decision and send the packet
        #----------------------

        #Get model_1 information about recieved packets
        nb_packets = len(fv_history)
        if nb_packets == 0:
            print("No packets received during listening time!")
            return
        
        idv_probas_reduced = np.zeros((nb_packets, 4))

        for i,fv in enumerate(fv_history):
            print("Packet #", i)
            #normalized_fv = fv / np.linalg.norm(fv)
            #Normalize by the maximum value of the fv
            normalized_fv = fv / np.max(fv)
            idv_probas = model_1.predict(normalized_fv)[0]
            idv_uncertainity = compute_uncertainity(payloads_history[i], idv_probas, idx, DTYPE, \
                                                            detector = doublesum_detector, a_sum=1, b_entropy=1, save_payloads=False)
            #do the difference of each proba and the uncertainty
            idv_probas_reduced[i] = idv_probas - idv_uncertainity  

        # If 2 paquets, get model_2 information on 2 concatenated packets
        #TODO: if 3 packets, apply 2 times the model_2?
        
        multiple_probas_reduced = np.zeros((1, 4))
        fv_combined = None
        n_melvecs = N_MELVECS
        if nb_packets == 2 and model_2:
            print("MODEL 2")
            fv_combined = np.hstack(fv_history[:2])
            #normalized_fv_combined = fv_combined / np.linalg.norm(fv_combined)
            #Normalize by the maximum value of the fv
            normalized_fv_combined = fv_combined / np.max(fv_combined)
            multiple_probas = model_2.predict(normalized_fv_combined)[0]
            multiple_uncertainty = compute_uncertainity(fv_combined, multiple_probas, idx, DTYPE, \
                                                        detector=None, a_sum=0, b_entropy=1, save_payloads=False)
            #do the difference of each proba and the uncertainty
            multiple_probas_reduced = multiple_probas - multiple_uncertainty
            n_melvecs = N_MELVECS * 2
            
            
        # If 3 packets, get model_3 information on 3 concatenated packets
        elif nb_packets >= 3 and model_3:
            print("MODEL 3")
            fv_combined = np.hstack(fv_history[:3])
            #normalized_fv_combined = fv_combined / np.linalg.norm(fv_combined)
            #Normalize by the maximum value of the fv
            normalized_fv_combined = fv_combined / np.max(fv_combined)
            multiple_probas = model_3.predict(normalized_fv_combined)[0]
            multiple_uncertainty = compute_uncertainity(fv_combined, multiple_probas, idx, DTYPE, \
                                                        detector=None, a_sum=0, b_entropy=1, save_payloads=False)
            #do the difference of each proba and the uncertainty
            multiple_probas_reduced = multiple_probas - multiple_uncertainty
            #Reduce the probability of gun (3rd index) by 0.5 as it is very unlikely with 3 packets
            multiple_probas_reduced[3] -= 0.5
            n_melvecs = N_MELVECS * 3

            
        #3. Decision making
        #----------------------

        #Sum each reduced idv probas one to one (i.e sum all probas 1 together, all probas 2 together, etc)
        final_probas = np.sum(idv_probas_reduced, axis=0)
        if nb_packets == 2 and model_2:
            final_probas += multiple_probas_reduced
        elif nb_packets >= 3 and model_3:
            final_probas += multiple_probas_reduced
        
        predicted_class = np.argmax(final_probas)
        predicted_class_name = class_names[predicted_class]
        print("Final probabilities: ", final_probas)
        print("Final Sent class: ", predicted_class_name)
        print("-------------------------------\n")

        fv_history.clear()
        payloads_history.clear()

        #4. Send the packet
        #----------------------

        hostname = "http://lelec210x.sipr.ucl.ac.be"
        key = "c6yuOmUKKbKAS4lRa7vQ9clME3bGTSWUvS4N26af"
        try:
            response = requests.post(
                f"{hostname}/lelec210x/leaderboard/submit/{key}/{predicted_class_name}",
                timeout=1
            )
        except requests.RequestException:
            print("Error sending prediction")


    fv_history = []
    payloads_history = []

    for idx, payload in enumerate(_input):
        current_time = time.time()
        print("PACKET #", idx)
        print("----------------")

        #Launch the decision thread if enough time 
        if not listening and (last_activation_time == 0 or (current_time - last_activation_time >= min_time_between_listening)):
            listening = True
            last_activation_time = current_time
            threading.Thread(target=wait_take_decision_and_send, args=(fv_history, payloads_history, doublesum_detector, idx, up_time, model_1, model_2, model_3), daemon=True).start()

        if not listening:
            print("Skipping packet (listening=False)")
            continue

        print("Processing packet (listening=True)")

        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX):].strip()

            melvec = payload_to_melvecs(payload, melvec_length, n_melvecs)
            msg_counter += 1
            #print(f"MEL Spectrogram #{msg_counter}")

            fv = melvec.reshape(1, -1)
            #fv = fv / np.linalg.norm(fv) NOT NORMALIZE BEFORE AS AMPLITUDE RECQUIRED FOR THE DOUBLE SUM
            fv_history.append(fv)
            payloads_history.append(payload)

            #Concatenate payloads in payloads_history

            #normalize the fv by its maximum value
            fv = fv / np.max(fv)
            #plot concatenated spectrogram
            plot_specgram(
                fv.reshape((n_melvecs, MELVEC_HEIGHT)),
                ax=plt.gca(),
                is_mel=True,
                title=f"MEL Spectrogram #{msg_counter}",
                xlabel="Mel vector",
                #textlabel=textlabel,
            )
            plt.draw()
            plt.pause(0.1)
            plt.clf()
    


if __name__ == "__main__":
    main()