import pickle
from pathlib import Path
from typing import Optional
import time
import click
import numpy as np
import matplotlib.pyplot as plt

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger
from classification.utils import payload_to_melvecs
from classification.utils.plots import plot_specgram

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
@common.click.melvec_length
@common.click.n_melvecs
@common.click.verbosity
def main(
    _input: Optional[click.File],
    model: Optional[Path],
    melvec_length: int,
    n_melvecs: int,
) -> None:
    """
    Extract Mel vectors from payloads and perform classification on them every 5 seconds.
    Logs predictions to a file with an index.
    """
    if model:
        with open(model, "rb") as file:
            try:
                model = load_model('model.h5')
                print(f"Model {type(model).__name__} loaded successfully.")
            except:
                model_rf = pickle.load(file)
                print(f"Model {type(model_rf).__name__} loaded successfully.")
    else:
        logger.warning("No model provided, skipping classification.")
        return

    msg_counter = 0
    log_file = "predictions_log.txt"

    with open(log_file, "w") as log:
        log.write("Index\tPrediction\n")
        log.write("=" * 30 + "\n")

        for payload in _input:
            if PRINT_PREFIX in payload:
                payload = payload[len(PRINT_PREFIX):]
                melvec = payload_to_melvecs(payload, melvec_length, n_melvecs)
                logger.info(f"Parsed payload into Mel vectors: {melvec}")

                msg_counter += 1
                print(f"MEL Spectrogram #{msg_counter}")

                # Reshape and normalize feature vector
                fv = melvec.reshape(1, -1)
                fv = fv / np.linalg.norm(fv)

                # Make prediction
                pred = model_rf.predict(fv)
                proba = model_rf.predict_proba(fv)
                predicted_class = pred[0]
                print(f"Predicted class: {predicted_class}")
                print(f"Predicted probabilities: {proba}")

                # Log prediction
                log.write(f"{msg_counter}\t{predicted_class}\n")

                # Optional: Visualization
                class_names = model_rf.classes_
                probabilities = np.round(proba[0] * 100, 2)
                max_len = max(len(name) for name in class_names)
                class_names_str = " ".join([f"{name:<{max_len}}" for name in class_names])
                probabilities_str = " ".join([f"{prob:.2f}%".ljust(max_len) for prob in probabilities])
                textlabel = f"{class_names_str}\n{probabilities_str}"
                textlabel += f"\n\nPredicted class: {predicted_class}"

                plot_specgram(
                    melvec.reshape((N_MELVECS, MELVEC_LENGTH)),
                    ax=plt.gca(),
                    is_mel=True,
                    title=f"MEL Spectrogram #{msg_counter}",
                    xlabel="Mel vector",
                    textlabel=textlabel,
                )
                plt.draw()
                plt.pause(0.1)
                plt.clf()

                # Wait for 5 seconds before processing the next prediction
                time.sleep(5)


if __name__ == "__main__":
    main()
