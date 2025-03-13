import os
import soundfile as sf
import numpy as np

basename = "sine" # Change this to the basename of the files you want to merge
delete_old_files = True

def merge_audios(input_directory, output_file):
    sample_rate = None
    with sf.SoundFile(output_file, mode='w', samplerate=10200, channels=1, format='WAV', subtype='PCM_16') as output_sf:
        for file_name in os.listdir(input_directory):
            if file_name.startswith(f"{basename}") and file_name.endswith(".wav"):
                file_path = os.path.join(input_directory, file_name)
                print(file_name)
                data, sr = sf.read(file_path, dtype=np.int16)
                print(data, sr)
                if sample_rate is None:
                    sample_rate = sr
                elif sample_rate != sr:
                    raise ValueError(f"Sample rate mismatch: {file_name} has a different sample rate.")
                output_sf.write(data)

def delete_old_files(input_directory):
    for file_name in os.listdir(input_directory):
        if file_name.startswith(f"{basename}") and file_name.endswith(".wav"):
            file_path = os.path.join(input_directory, file_name)
            os.remove(file_path)

if __name__ == "__main__":
    input_directory = "audio_files/"
    output_file = os.path.join(input_directory, f"merged_{basename}.wav")
    merge_audios(input_directory, output_file)
    if delete_old_files:
        delete_old_files(input_directory)