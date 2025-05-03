import pickle
from pathlib import Path
from typing import Optional
import time
import click
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import threading  # For handling work deactivation in a separate thread

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
    save_payloads: Optional[Path],
) -> None:
    """
    Extract Mel vectors from payloads and perform classification on them every 5 seconds.
    Logs predictions to a file with an index.
    """

    # Load the model
    if model:
        with open(model, "rb") as file:
            model = load_model(model)
            print(f"Model {type(model).__name__} loaded successfully.")
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
    BASELINE = 2048  # Verify necessity

    doublesum_detector = DoubleRunningSumDetector(N_STS=N_STS, N_LTS=N_LTS, K=K, BASELINE=BASELINE)

    # MEMORY TRACKER
    window_size = 3
    alpha = 0.7
    reset_interval = 2.5  # seconds

    memory_tracker = UncertaintyTracker(window_size=window_size, alpha=alpha, reset_interval=reset_interval)

    # DECISION PARAMETERS
    a_sum = 1
    b_entropy = 1
    c_memory = 0

    # WORK VARIABLE MANAGEMENT
    work = False
    last_activation_time = 0  # Tracks when work was last activated

    def deactivate_work():
        """Function to disable `work` after 3 seconds."""
        time.sleep(3)
        nonlocal work
        work = False
        print("Work set to FALSE")

    for i, payload in enumerate(_input):  # Loop over input payloads
        current_time = time.time() # Get the current time 

        # If work is False and enough time has passed, allow reactivation
        if not work and (last_activation_time == 0 or (current_time - last_activation_time >= 5)): # Work was activated during 3s and must be deactivated during at least 2s
            work = True
            print("Work set to TRUE")
            last_activation_time = current_time  # Update activation time
            threading.Thread(target=deactivate_work, daemon=True).start()

        if not work:
            print("Skipping packet (work=False)")
            continue  # Skip processing if work is False

        print("Processing packet (work=True)")

        if PRINT_PREFIX in payload:  # Verify if the payload is valid
            payload = payload[len(PRINT_PREFIX):]  # Remove the prefix
            payload = payload.strip()

            # Convert payload to Mel vectors
            melvec = payload_to_melvecs(payload, melvec_length, n_melvecs)

            msg_counter += 1
            print(f"MEL Spectrogram #{msg_counter}")

            # Reshape and normalize feature vector
            fv = melvec.reshape(1, -1)
            fv = fv / np.linalg.norm(fv)

            # Make prediction
            probas = model.predict(fv)
            predicted_class = np.argmax(probas, axis=1)[0]
            predicted_class_name = class_names[predicted_class]

            # Uncertainty calculation
            send_packet = decision(payload, probas, i, DTYPE, predicted_class, doublesum_detector, memory_tracker,
                                   a_sum=a_sum, b_entropy=b_entropy, c_memory=c_memory,
                                   save_payloads=save_payloads)

            if send_packet:
                print("SEND TRUE: ", predicted_class_name)
                hostname = "http://lelec210x.sipr.ucl.ac.be"
                key = "c6yuOmUKKbKAS4lRa7vQ9clME3bGTSWUvS4N26af"
                guess = predicted_class_name
                response = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{key}/{guess}", timeout=1)
            else:
                print("SEND FALSE: ", predicted_class_name)


if __name__ == "__main__":
    main()
