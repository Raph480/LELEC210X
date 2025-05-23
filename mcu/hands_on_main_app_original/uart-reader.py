"""
uart-reader.py
ELEC PROJECT - 210x
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import serial
import soundfile as sf
from serial.tools import list_ports

PRINT_PREFIX = "SND:HEX:"
FREQ_SAMPLING = 10200
VAL_MAX_ADC = 4096
VDD = 3.3


def parse_buffer(line):
    """
    #initial version
        line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
    else:
        print(line)
        return None
    """
    #try & except version
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        hex_data = line[len(PRINT_PREFIX):]
        try:
            # Ensure the string only contains valid hex characters
            return bytes.fromhex(hex_data)
        except ValueError as e:
            print(f"Error converting line to hex: {e}")
            return None
    else:
        print(line)
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=1042).decode("ascii")
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            dt = np.dtype(np.uint16)
            dt = dt.newbyteorder("<")
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array


def generate_audio(buf, file_name):
    buf = np.asarray(buf, dtype=np.float64)
    buf = buf - np.mean(buf)
    buf /= max(abs(buf))
    sf.write("audio_files/"+file_name + ".ogg", buf, FREQ_SAMPLING)
    


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port for serial communication")
    argParser.add_argument("-f", "--filename", help="Base name for the output files", default="acq")
    args = argParser.parse_args()
    print("uart-reader launched...\n")

    DURATION_SECONDS = 10
    NUM_SAMPLES = DURATION_SECONDS * FREQ_SAMPLING

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
        plt.figure(figsize=(10, 5))
        input_stream = reader(port=args.port)
        msg_counter = 0
        full_buffer = []  # List to accumulate samples

        for msg in input_stream:
            full_buffer.extend(msg)  # Append new samples

            if len(full_buffer) >= NUM_SAMPLES:
                print(f"Generating audio #{msg_counter} with {NUM_SAMPLES} samples")

                buffer_size = len(full_buffer)
                times = np.linspace(0, buffer_size - 1, buffer_size) * 1 / FREQ_SAMPLING
                voltage_mV = np.array(full_buffer) * VDD / VAL_MAX_ADC * 1e3

                plt.clf()
                plt.plot(times, voltage_mV)
                plt.title(f"Acquisition #{msg_counter}")
                plt.xlabel("Time (s)")
                plt.ylabel("Voltage (mV)")
                plt.ylim([0, 3300])
                plt.draw()
                plt.pause(0.001)

                base_name = f"{args.filename}-{msg_counter}"
                plt.savefig(f"audio_plots/{base_name}.pdf", bbox_inches='tight')

                generate_audio(full_buffer, base_name)  # Generate OGG audio file

                msg_counter += 1
                full_buffer = []  # Clear buffer for the next segment
