import os
import random
import time
from src.classification.utils.audio_student import AudioUtil  # Importing AudioUtil

def play_wav_files(folder_path, play_random=False, delay=0):
    """
    Play all .wav files in the specified folder, one by one sequentially or randomly.
    
    :param folder_path: Path to the folder containing .wav files.
    :param play_random: Boolean to determine if playback is random.
    :param delay: Delay in seconds between consecutive audio plays.
    """
    # Get list of all .wav files in the folder
    wav_files = [file for file in os.listdir(folder_path) if file.endswith(".wav")]
    
    if not wav_files:
        print("No .wav files found in the specified folder.")
        return
    
    # Shuffle the list if play_random is True
    if play_random:
        random.shuffle(wav_files)
    else:
        wav_files.sort()  # Optional: Sort files alphabetically

    print(f"Playing files in {'random' if play_random else 'sequential'} order with a {delay}s delay.")
    
    for wav_file in wav_files:
        file_path = os.path.join(folder_path, wav_file)
        print(f"Playing: {wav_file}")
        try:
            # Load the audio file
            audio = AudioUtil.open(file_path)
            
            # Play the audio file (blocking call)
            AudioUtil.play(audio)
            
            # Wait for the playback to finish before adding the delay
            duration = len(audio[0]) / audio[1]  # Calculate duration (samples / sample_rate)
            time.sleep(duration + delay)  # Add playback time and delay
        except Exception as e:
            print(f"Error playing {wav_file}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Play .wav files in a folder sequentially or randomly.")
    #parser.add_argument("folder", type=str, help="Path to the folder containing .wav files.")
    parser.add_argument("--random", action="store_true", help="Play files in random order if set.")
    parser.add_argument("--delay", type=float, default=0, help="Delay in seconds between plays (default: 0).")

    args = parser.parse_args()
    
    folder = 'src/classification/datasets/soundfiles'
    play_wav_files(folder, args.random, args.delay)




