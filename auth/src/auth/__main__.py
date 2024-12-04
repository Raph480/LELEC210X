import argparse
import pickle
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from serial import Serial
import click
from common.env import load_dotenv
from common.logging import logger
from .utils import payload_to_melvecs
from classification.utils.plots import plot_specgram

# Constants
PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 102
DTYPE = np.dtype(np.uint16).newbyteorder("<")

load_dotenv()


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
    "--serial-port",
    default=None,
    help="Port for serial communication.",
)
@click.option(
    "--auth-output",
    is_flag=True,
    default=False,
    help="Specify if the input is the output of the authentication script.",
)
@common.click.melvec_length
@common.click.n_melvecs
@common.click.verbosity
def main(
    _input: Optional[click.File],
    model: Optional[Path],
    serial_port: Optional[str],
    auth_output: bool,
    melvec_length: int,
    n_melvecs: int,
) -> None:
    """
    Process data from input, classify Mel spectrograms, and optionally visualize them.
    """
    if model:
        with open(model, "rb") as file:
            classifier = pickle.load(file)
        logger.info(f"Model {type(classifier).__name__} loaded from {model}.")
    else:
        logger.warning("No classification model provided. Skipping classification.")
        classifier = None

    def parse_authenticated_input(line: str):
        """Parse a line assuming it's in the format produced by the authentication script."""
        if line.startswith(PRINT_PREFIX):
            return bytes.fromhex(line[len(PRINT_PREFIX) :])
        return None

    if serial_port:  # Read directly from a serial port
        ser = Serial(port=serial_port, baudrate=115200)
        ser.reset_input_buffer()
        input_stream = iter(
            lambda: ser.read_until(b"\n").decode("ascii").strip(), ""
        )
    else:  # Read from file or stdin
        input_stream = _input

    plt.figure(figsize=(8, 6))
    msg_counter = 0

    for line in input_stream:
        if auth_output:
            # Parse authenticated output
            buffer = parse_authenticated_input(line.strip())
        else:
            # Directly parse payload to Mel vectors
            if PRINT_PREFIX in line:
                payload = line[len(PRINT_PREFIX) :].strip()
                buffer = payload_to_melvecs(payload, melvec_length, n_melvecs)
            else:
                continue

        if buffer is None:
            continue

        melvec = np.frombuffer(buffer, dtype=DTYPE)[4:-8]
        msg_counter += 1

        logger.info(f"Processing MEL spectrogram #{msg_counter}.")

        # Normalize feature vector
        fv = melvec.reshape(1, -1)
        fv = fv / np.linalg.norm(fv)

        # Classify if a model is provided
        if classifier:
            pred = classifier.predict(fv)
            proba = classifier.predict_proba(fv)
            logger.info(f"Predicted class: {pred[0]}")
            logger.info(f"Class probabilities: {proba}")

            # Prepare probabilities for display
            class_names = classifier.classes_
            probabilities = np.round(proba[0] * 100, 2)
            max_len = max(len(name) for name in class_names)
            class_names_str = " ".join([f"{name:<{max_len}}" for name in class_names])
            probabilities_str = " ".join([f"{prob:.2f}%".ljust(max_len) for prob in probabilities])
            textlabel = f"{class_names_str}\n{probabilities_str}\n\nPredicted class: {pred[0]}"
        else:
            textlabel = "No model loaded for classification."

        # Plot Mel spectrogram
        plot_specgram(
            melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T,
            ax=plt.gca(),
            is_mel=True,
            title=f"MEL Spectrogram #{msg_counter}",
            xlabel="Mel vector",
            textlabel=textlabel,
        )
        plt.draw()
        plt.pause(0.1)
        plt.clf()
