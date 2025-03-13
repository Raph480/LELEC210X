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
AUDIO = []
DTYPE = np.int16

def parse_buffer(line):
    """
    #initial version
        line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) 🙂)
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
            print(f"\aError converting line to hex: {e}")
            return None
    else:
        print(line)
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=921600) # OLD: 115200
    while True:
        line = ""
        while not line.endswith("\n"):
            try:
                line += ser.read_until(b"\n", size=1042).decode("ascii")
            except UnicodeDecodeError as e:
                print(f"\aError decoding line: {e}")
                exit(1)
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            dt = np.dtype(np.uint16)
            dt = dt.newbyteorder("<")
            try:
                buffer_array = np.frombuffer(buffer, dtype=dt)
            except ValueError as e:
                print(f"\aError converting buffer to array: {e}")
                exit(1)

            yield buffer_array


def generate_audio(buf, file_name):
    print(buf)
    buf = np.asarray(buf, dtype=DTYPE)
    buf = buf << 3
    buf = buf - (1 << 14)
    #print(buf)
    sf.write("audio_files/"+file_name + ".wav", buf, FREQ_SAMPLING)



if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port for serial communication")
    argParser.add_argument("-f", "--filename", help="Base name for the output files", default="acq")
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
        #plt.figure(figsize=(10, 5))
        input_stream = reader(port=args.port)
        msg_counter = 0

        for msg in input_stream:

            buffer_size = len(msg)

            """
            times = np.linspace(0, buffer_size - 1, buffer_size) * 1 / FREQ_SAMPLING
            voltage_mV = msg * VDD / VAL_MAX_ADC * 1e3

            plt.clf()  # Clear the figure for the next acquisition
            plt.plot(times, voltage_mV)
            plt.title(f"Acquisition #{msg_counter}")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (mV)")
            plt.ylim([0, 3300])
            plt.draw()
            plt.pause(0.001)
            #plt.cla()
            # Use the user-provided filename as the base name
            base_name = f"{args.filename}-{msg_counter}"

            # Save the plot as a PDF
            plt.savefig(f"audio_plots/{base_name}.pdf", bbox_inches='tight')
            """
            # Generate and save the audio as an WAV file
            base_name = f"{args.filename}-{msg_counter:03d}"
            generate_audio(msg, base_name)

            print(f"Audio file #{msg_counter} saved ({buffer_size} samples)")

            msg_counter += 1