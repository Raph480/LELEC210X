import pickle
from pathlib import Path
from typing import Optional

import click

import common
from auth import PRINT_PREFIX
from common.env import load_dotenv
from common.logging import logger

from classification.utils import payload_to_melvecs

import numpy as np
import matplotlib.pyplot as plt
from classification.utils.plots import plot_specgram

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
    Extract Mel vectors from payloads and perform classification on them.
    Classify MELVECs contained in payloads (from packets).

    Most likely, you want to pipe this script after running authentification
    on the packets:

        rye run auth | rye run classify

    This way, you will directly receive the authentified packets from STDIN
    (standard input, i.e., the terminal).
    """
    if model:
        with open(model, "rb") as file:
            m = pickle.load(file)
    else:
        m = None

    msg_counter = 0
    for payload in _input:
        if PRINT_PREFIX in payload:
            payload = payload[len(PRINT_PREFIX) :]

            melvec = payload_to_melvecs(payload, melvec_length, n_melvecs)
            logger.info(f"Parsed payload into Mel vectors: {melvec}")


            if m:
                 model_rf = pickle.load(open(model, "rb"))
                 print(f"Model {type(model_rf).__name__} has been loaded from pickle file.\n")
                 pass
            else:
                logger.warning("No model provided, skipping classification.")
                continue

            msg_counter += 1

            print(f"MEL Spectrogram #{msg_counter}")

            #np.savetxt(f"melspectrograms_plots/melvec_{msg_counter}.txt", melvec, fmt="%04x", delimiter="\n")

            # Predict the class of the mel vector
            
            fv = melvec.reshape(1, -1)
            fv = fv / np.linalg.norm(fv)

            pred = model_rf.predict(fv)
            proba = model_rf.predict_proba(fv)
            print(f"Predicted class: {pred[0]}\n")
            print(f"Predicted probabilities: {proba}\n")
            
            class_names = model_rf.classes_
            probabilities = np.round(proba[0] * 100, 2)
            max_len = max(len(name) for name in class_names)
            class_names_str = " ".join([f"{name:<{max_len}}" for name in class_names])
            probabilities_str = " ".join([f"{prob:.2f}%".ljust(max_len) for prob in probabilities])
            textlabel = f"{class_names_str}\n{probabilities_str}"
            # For column text: textlabel = "\n".join([f"{name:<11}: {prob:>6.2f}%" for name, prob in zip(class_names, probabilities)])
            textlabel = textlabel + f"\n\nPredicted class: {pred[0]}\n" 
            
            #textlabel = ""
            plot_specgram(
                melvec.reshape((N_MELVECS, MELVEC_LENGTH)), #.T 
                ax=plt.gca(),
                is_mel=True,
                title=f"MEL Spectrogram #{msg_counter}",
                xlabel="Mel vector",
                textlabel=textlabel,
            )
            plt.draw()
            #plt.savefig(f"melspectrograms_plots/melspec_{msg_counter}.pdf")
            plt.pause(0.1)
            plt.clf()

