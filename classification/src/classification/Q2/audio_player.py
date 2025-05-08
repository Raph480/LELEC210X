import os
import random
import time
from classification.Q2.audio_student import AudioUtil  # Importing AudioUtil
#from classification.src.classification.Q2.audio_student import AudioUtil  # Importing AudioUtil

def play_wav_files(folder_path, play_random=False, delay=0, log_file="play_log.txt"):
    """
    Play all .wav files in the specified folder, one by one sequentially or randomly.
    Logs the class name of each played file to a log file.
    
    :param folder_path: Path to the folder containing .wav files.
    :param play_random: Boolean to determine if playback is random.
    :param delay: Delay in seconds between consecutive audio plays.
    :param log_file: Path to the file where playback logs will be written.
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
    
    # Open log file for writing
    with open(log_file, "w") as log:
        log.write("Index\tClass Name\n")
        log.write("=" * 30 + "\n")
        
        for index, wav_file in enumerate(wav_files, start=0):
            if index >= 90:
            #if (0 <= index <= 10) or (40 <= index <= 50) or (80 <= index <= 90) or (120 <= index <= 130):

                file_path = os.path.join(folder_path, wav_file)
                
                # Extract class name (portion before the first '_')
                class_name = wav_file.split("_")[0] if "_" in wav_file else "Unknown"
                
                print(f"Playing: {wav_file} (Class: {class_name})")
                log.write(f"{index}\t{class_name}\n")  # Log the class name
                
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
                    #log.write(f"{index}\t{class_name} - ERROR: {e}\n")  # Log errors

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Play .wav files in a folder sequentially or randomly.")
    parser.add_argument("--random", action="store_true", help="Play files in random order if set.")
    parser.add_argument("--delay", type=float, default=0, help="Delay in seconds between plays (default: 0).")
    parser.add_argument("--log", type=str, default="play_log.txt", help="Path to log file (default: play_log.txt).")

    args = parser.parse_args()
    
    folder = '../datasets/sounds/Q2_sounds'
    play_wav_files(folder, args.random, args.delay, args.log)
