import os
import wave
import contextlib
from pydub import AudioSegment

def split_wav(input_wav_path, segment_length, output_folder, name="segment"):
    """
    Splits a WAV file into smaller segments of given length and saves them to the output folder.

    :param input_wav_path: Path to the input WAV file.
    :param segment_length: Length of each segment in seconds.
    :param output_folder: Folder to save the split WAV files.
    :param name: Base name for the output files (default: 'segment').
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the audio file
    audio = AudioSegment.from_wav(input_wav_path)

    # Get duration of the audio file
    with contextlib.closing(wave.open(input_wav_path, 'r')) as wf:
        duration = wf.getnframes() / wf.getframerate()

    # Calculate the number of segments
    num_segments = int(duration // segment_length)
    if duration % segment_length != 0:
        num_segments += 1  # Include last segment if there's a remainder

    print(f"Splitting {input_wav_path} into {num_segments} segments of {segment_length}s each.")

    # Split and save each segment
    saved_files = []
    for i in range(num_segments):
        start_time = i * segment_length * 1000  # Convert to milliseconds
        end_time = min((i + 1) * segment_length * 1000, len(audio))

        segment = audio[start_time:end_time]
        segment_filename = os.path.join(output_folder, f"{name}_{i+1}.wav")
        segment.export(segment_filename, format="wav")
        
        saved_files.append(segment_filename)
        print(f"Saved: {segment_filename}")

    return saved_files

input_file = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/sounds/recorded_sounds/cropped_recordings/merged_gun_background.wav"
output_folder = "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/sounds/recorded_sounds/gun"
name = "gun_4_background"

seconds = 4.1

# Example usage
split_wav(input_file, seconds, output_folder, name)