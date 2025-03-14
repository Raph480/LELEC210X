import argparse
import matplotlib.pyplot as plt
import numpy as np
import serial
import soundfile as sf
import sounddevice as sd
import time
from serial.tools import list_ports
import os

PRINT_PREFIX = "SND:HEX:"
FREQ_SAMPLING = 10200
VAL_MAX_ADC = 4096
VDD = 3.3

final_audio = []  # Store all acquired buffers
acquisition_active = False  # Track acquisition state



def parse_buffer(line):
    """ Parse hex data from UART """
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        hex_data = line[len(PRINT_PREFIX):]
        try:
            return bytes.fromhex(hex_data)
        except ValueError as e:
            print(f"Error converting line to hex: {e}")
            return None
    else:
        print(line)
        if "Acquisition started" in line:
            global acquisition_active
            acquisition_active = True
            print("Acquisition has started.")
            print(acquisition_active)
        elif "Acquisition stopped" in line:
            acquisition_active = False
            print("Acquisition has stopped. Saving final audio file...")
            save_final_audio()
        return None
        #print(line)
        #return None


def reader(port=None):
    """ Read a single buffer from UART """
    ser = serial.Serial(port=port, baudrate=115200)
    
    while True:
        line = ser.read_until(b"\n", size=1042).decode("ascii").strip()
        buffer = parse_buffer(line)
        if buffer is not None:

            if len(buffer) % 2 != 0:  # Ensure buffer size is a multiple of 2
                print(f"Warning: Buffer size {len(buffer)} is not a multiple of 2. Discarding this buffer.")
                continue  # Skip this iteration

            dt = np.dtype(np.uint16)
            dt = dt.newbyteorder("<")
            buffer_array = np.frombuffer(buffer, dtype=dt)
            yield buffer_array

def save_final_audio():
    """ Save all concatenated audio into a single .wav file """
    #print(final_audio)
    if final_audio:
        full_audio = np.concatenate(final_audio)
        full_audio = full_audio.astype(np.float64)
        full_audio -= np.mean(full_audio)
        full_audio /= np.max(np.abs(full_audio))
        sf.write("final_acquisition.wav", full_audio, FREQ_SAMPLING)
        print("Final audio file saved as 'final_acquisition.wav'.")
    else:
        print("No audio data acquired.")


def generate_audio(buf, file_name):
    """ Normalize and save audio as OGG file """
    buf = np.asarray(buf, dtype=np.float64)
    buf = buf - np.mean(buf)
    buf /= max(abs(buf))
    sf.write(f"audio_files/{file_name}.ogg", buf, FREQ_SAMPLING)


def play_audio(file_path):
    """ Play an audio file """
    if os.path.exists(file_path):
        print(f"Playing {file_path}...")
        data, samplerate = sf.read(file_path)
        sd.play(data, samplerate)
        sd.wait()  # Block until playback is finished
    else:
        print(f"File {file_path} not found, skipping playback.")


def continuous_reader(port):
    """ Continuously collect audio data """
    global final_audio
    input_stream = reader(port=port)
    print("continuous: ", acquisition_active)

    """
    while True:
        msg = next(input_stream)
        if msg is not None:
            print("msg not none")
        if acquisition_active:
            print("acq active")
        if msg is not None and acquisition_active:
            final_audio.append(msg)
    """
    for msg in input_stream:
        print("ok")
        if acquisition_active:
                final_audio.append(msg)
        else:
            break  # Stop reading when acquisition stops
    



if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--port", help="Port for serial communication")
    argParser.add_argument("-f", "--filename", help="Base name for output files", default="acq")
    args = argParser.parse_args()
    print("UART Reader started...\n")

    if args.port is None:
        print("No port specified. Available ports:")
        print("================")
        for p in list(list_ports.comports()):
            print(p.device)
        print("================")
        print("Run script with [-p PORT_REF] to specify the communication port.")
    else:
        continuous_reader(args.port)  # Use continuous reading