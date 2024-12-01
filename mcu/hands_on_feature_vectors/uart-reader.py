"""
uart-reader.py
ELEC PROJECT - 210x
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports
import pickle

from classification.Random_Forest import final_model
from classification.utils.plots import plot_specgram





PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 102

dt = np.dtype(np.uint16).newbyteorder("<")


def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
    else:
        print(line)
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=2 * N_MELVECS * MELVEC_LENGTH).decode(
                "ascii"
            )
            print(line)
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array


if __name__ == "__main__":    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port for serial communication")
    args = argParser.parse_args()
    print("uart-reader launched...\n")

    if args.port is None:
        print(
            "No port specified, here is a list of serial communication port available"
        )
        print("================")
        port = list(list_ports.comports())
        for p in port:
            print(p.device)
        print("================")
        print("Launch this script with [-p PORT_REF] to access the communication port")

    else:
        input_stream = reader(port=args.port)
        msg_counter = 0

        #model_rf = final_model(verbose=False)
        #print("Random forest classifier fitted...\n")

        # Load the model from pickle file

        #model_rf = pickle.load(open("../../classification/data/models/final_model_5s_norm.pkl", "rb"))
        model_rf = pickle.load(open("../../classification/data/models/final_model_with_txt_aug.pkl", "rb"))
        print(f"Model {type(model_rf).__name__} has been loaded from pickle file.\n")

        plt.figure(figsize=(8, 6))
        for melvec in input_stream:
            melvec = melvec[4:-8]
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
                melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T,
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
