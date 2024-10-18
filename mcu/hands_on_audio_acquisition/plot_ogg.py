import librosa
import numpy as np

import librosa.display
import matplotlib.pyplot as plt

# Load the OGG audio file
file_path = 'audio_files/acq-1.ogg'
y, sr = librosa.load(file_path, sr=None)

# Plot amplitude vs time
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('Amplitude vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
#plt.show()

# plot fft of y

fft = np.fft.fft(y)
# Find frequency corresponding to each point in the FFT
freqs = np.fft.fftfreq(len(fft), 1/sr)

# Find the positive frequencies
mask = freqs > 0
fft = fft[mask]
freqs = freqs[mask]

# Plot the FFT
plt.figure(figsize=(14, 5))
plt.plot(freqs, np.abs(fft))
plt.title('FFT')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.show()




# Plot spectrogram
D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D, sr=sr)
S_dB = librosa.power_to_db(S, ref=np.max)
# in h1, we didn't use dB scale but only log

plt.figure(figsize=(14, 5))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()