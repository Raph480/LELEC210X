"""
uart-reader.py
ELEC PROJECT - 210x
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports
from classification.utils.plots import plot_specgram


PRINT_PREFIX = "DF:HEX:"
MELVEC_LENGTH = 10
N_MELVECS = 20

dt = np.dtype(np.uint16).newbyteorder("<")

def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        try: 
            return bytes.fromhex(line[len(PRINT_PREFIX) :])
        except ValueError:
            print(f"Error parsing line: {line}")
            return None
    else:
        print(line)
        return None


def reader(port=None):

    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=1 * N_MELVECS * MELVEC_LENGTH).decode(
                "ascii"
            )
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
        
        plt.figure(figsize=(8, 6))
        for melvec in input_stream:

            melvec = melvec[4:-8]
            msg_counter += 1
            print(f"MEL Spectrogram #{msg_counter}")


            if len(melvec) == N_MELVECS * MELVEC_LENGTH /2:   # Probably because 8bit data is sent instead of 16bit
                temp_melvec = np.empty(len(melvec) * 2, dtype=np.uint8)
                temp_melvec[0::2] = (melvec & 0xFF).astype(np.uint8)  # Extract lower byte
                temp_melvec[1::2] = (melvec >> 8).astype(np.uint8)    # Extract upper byte
                melvec = temp_melvec
            else:
                #invert the bytes to go from little endian to big endian
                melvec = melvec.view(np.uint8).reshape(-1, 2)[:, ::-1].flatten()
                melvec = melvec.view(dt)
                

            #np.savetxt(f"melspectrograms_plots/melvec_{msg_counter}.txt", melvec, fmt="%04x", delimiter="\n")

            if melvec.size == N_MELVECS * MELVEC_LENGTH:
                print()
            else:
                print(f"Error: melvec size {melvec.size} does not match expected size {N_MELVECS * MELVEC_LENGTH}")
            print("HERE")
            #textlabel = ""
            #melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T,
            plot_specgram(
                melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T,
                ax=plt.gca(),
                is_mel=True,
                title=f"MEL Spectrogram #{msg_counter}",
                xlabel="Mel vector",
                #textlabel=textlabel,
            )
            plt.draw()
            #plt.savefig(f"melspectrograms_plots/melspec_{msg_counter}.pdf")
            plt.pause(0.3)
            plt.clf()
