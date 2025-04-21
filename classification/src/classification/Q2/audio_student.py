import random
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy import ndarray
from scipy.signal import fftconvolve
from scipy import signal

import mcu_emulation_v2_1 as mcu

# -----------------------------------------------------------------------------
"""
Synthesis of the classes in :
- AudioUtil : util functions to process an audio signal.
- Feature_vector_DS : Create a dataset class for the feature vectors.
"""
# -----------------------------------------------------------------------------


class AudioUtil:
    """
    Define a new class with util functions to process an audio signal.
    """

    def open(audio_file, dtype=np.int16) -> Tuple[ndarray, int]:
        """
        Load an audio file.

        :param audio_file: The path to the audio file.
        :return: The audio signal as a tuple (signal, sample_rate).
        """
        
        sig, sr = sf.read(audio_file)
        if sig.ndim > 1:
            sig = sig[:, 0]
        return (sig, sr)

    def play(audio):
        """
        Play an audio file.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        """
        sig, sr = audio
        sd.play(sig, sr)

    def normalize(audio, target_dB=52) -> Tuple[ndarray, int]:
        """
        Normalize the energy of the signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param target_dB: The target energy in dB.
        """
        sig, sr = audio
        sign = sig / np.sqrt(np.sum(np.abs(sig) ** 2))
        C = np.sqrt(10 ** (target_dB / 10))
        sign *= C
        return (sign, sr)
    
    def resample(audio, newsr=11025) -> Tuple[ndarray, int]:
        """
        Resample to target sampling frequency.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param newsr: The target sampling frequency.
        """
        
        sig, sr = audio

        # Calculate the resampling factor
        M = newsr / sr #M is calculated as the ratio of the target sampling frequency (newsr) to the original sampling frequency (sr)

        # Compute the new length of the signal
        new_len = int(len(sig) * M)

        # Resample the signal using scipy.signal.resample
        resig = signal.resample(sig, new_len)
        #resig = signal.resample_poly(sig, newsr, sr)

        return (resig, newsr)


    @staticmethod
    def compute_energy(signal, frame_size):
        """Compute short-term energy of the signal. Used for the pad_trunc function."""
        energy = np.array([
            np.sum(signal[i : i + frame_size] ** 2)
            for i in range(0, len(signal) - frame_size, frame_size // 2)
        ])
        return energy

    def pad_trunc(audio, max_ms) -> Tuple[ndarray, int]:
        """
        Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param max_ms: The target length in milliseconds.
        """
        sig, sr = audio
        sig_len = len(sig)
        max_len = int(sr * max_ms / 1000)

        if sig_len > max_len:
            # Truncate the signal to the given length at random position
            # begin_len = random.randint(0, max_len)
            #begin_len = 0
            #sig = sig[begin_len : begin_len + max_len]
            
            # Compute short-term energy
            frame_size = int(sr * 0.02)  # 20ms frames
            energy = AudioUtil.compute_energy(sig, frame_size)

            # Find high-energy regions
            threshold = np.percentile(energy, 90)  # Top 30% energy
            high_energy_indices = np.where(energy >= threshold)[0]

            if len(high_energy_indices) > 0:
                best_start = high_energy_indices[0] * (frame_size // 2)
            else:
                best_start = 0  # Fallback to the beginning

            # Randomly select a start point between 10% and 70% of the final signal
            # Calculate the range for random offset
            min_offset = int(0.1 * max_len)  # 10% of max_len
            max_offset = int(0.7 * max_len)  # 70% of max_len

            # Apply a random offset within this range
            random_offset = np.random.randint(min_offset, max_offset)

            # Ensure we don't start before the beginning of the signal
            begin_len = max(0, min(best_start - random_offset, sig_len - max_len))
            sig = sig[begin_len : begin_len + max_len]
            
        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = np.zeros(pad_begin_len)
            pad_end = np.zeros(pad_end_len)

            # sig = np.append([pad_begin, sig, pad_end])
            sig = np.concatenate((pad_begin, sig, pad_end))

        return (sig, sr)

    def time_shift(audio, shift_limit=1) -> Tuple[ndarray, int]:
        """
        Shifts the signal to the left or right by some percent. Values at the end are 'wrapped around' to the start of the transformed signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param shift_limit: The percentage (between 0.0 and 1.0) by which to circularly shift the signal.
        """
        sig, sr = audio
        sig_len = len(sig)

        # Define min and max shift bounds
        min_shift = int(0.1 * sig_len)  # minimum 10% shift
        max_shift = int(shift_limit * sig_len)

        # Avoid edge cases where max_shift < min_shift
        if max_shift <= min_shift:
            shift_amt = min_shift  # or just skip shifting
        else:
            shift_amt = random.randint(min_shift, max_shift)

        return (np.roll(sig, shift_amt), sr)

    def scaling(audio, scaling_limit=5) -> Tuple[ndarray, int]:
        """
        Augment the audio signal by scaling it by a random factor.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param scaling_limit: The maximum scaling factor.
        """
        sig, sr = audio
        scale = np.random.uniform(1 / scaling_limit, scaling_limit)  # Random scaling factor
        sig = sig * scale  # Scale the signal
        return (sig, sr)

    def add_noise(audio, sigma=0.05) -> Tuple[ndarray, int]:
        """
        Augment the audio signal by adding gaussian noise.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param sigma: Standard deviation of the gaussian noise.
        """
        sig, sr = audio
        noise = np.random.normal(0, sigma, sig.shape)  # Generate Gaussian noise
        sig = sig + noise  # Add noise to the signal
        return (sig, sr)

    def add_echo(audio, nechos=2) -> Tuple[ndarray, int]:
        """
        Add echo to the audio signal by convolving it with an impulse response. The taps are regularly spaced in time and each is twice smaller than the previous one.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param nechos: The number of echoes.
        """
        sig, sr = audio
        sig_len = len(sig)
        echo_sig = np.zeros(sig_len)
        echo_sig[0] = 1
        echo_sig[(np.arange(nechos) / nechos * sig_len).astype(int)] = (
            1 / 2
        ) ** np.arange(nechos)

        sig = fftconvolve(sig, echo_sig, mode="full")[:sig_len]
        return (sig, sr)

    def filter(audio, filt) -> Tuple[ndarray, int]:
        """
        Filter the audio signal with a provided filter. Note the filter is given for positive frequencies only and is thus symmetrized in the function.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param filt: The filter to apply.
        """
        sig, sr = audio
        fft_sig = np.fft.fft(sig)  # Perform FFT on the signal
        filt_full = np.concatenate([filt, filt[::-1]])[:fft_sig.shape[0]]  # Symmetrize the filter
        fft_sig = fft_sig * filt_full  # Apply the filter in the frequency domain
        sig = np.fft.ifft(fft_sig).real  # Perform inverse FFT to get the filtered signal
        return (sig, sr)


    def add_bg(audio, dataset, num_sources=1, max_ms=5000, amplitude_limit=0.1) -> Tuple[np.ndarray, int]:
        """
        Adds background sounds from the dataset to the given audio, starting at a random point.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param dataset: The dataset to sample background sounds from.
        :param num_sources: The number of background sounds to mix in.
        :param max_ms: The maximum duration of the background sounds.
        :param amplitude_limit: The maximum amplitude of the added background sounds.
        :return: The mixed audio signal and sample rate.
        """
        
        #Shift without wrapping around
        sig, sr = audio
        sig_len = len(sig)

        # Define min and max shift bounds
        min_shift = int(0.1 * sig_len)  # minimum 10% shift
        max_shift = int(0.9 * sig_len)  #maximum 90% shift

        if max_shift <= min_shift:
            shift_amt = min_shift
        else:
            shift_amt = random.randint(min_shift, max_shift)

        # Randomly decide left or right shift
        direction = random.choice(["left", "right"])

        if direction == "right":
            shifted = np.zeros_like(sig)
            shifted[shift_amt:] = sig[:-shift_amt]
        else:  # left shift
            shifted = np.zeros_like(sig)
            shifted[:-shift_amt] = sig[shift_amt:]
        
        sig = shifted

        for _ in range(num_sources):
            # Randomly select a background sound
            class_name = random.choice(list(dataset.files.keys()))  # Select a random class
            index = random.randint(0, len(dataset.files[class_name]) - 1)  # Select a random index
            bg_audio_path = dataset.__getitem__((class_name, index))  # Retrieve the file path
            
            # Load the background sound
            bg_audio, bg_sr = AudioUtil.open(bg_audio_path)

            # Resample if necessary
            if bg_sr != sr:
                bg_audio = AudioUtil.resample((bg_audio, bg_sr), sr)[0]  # Resample and extract signal
                bg_sr = sr  # Ensure sample rate consistency

            # Convert max duration from milliseconds to samples
            max_samples = int(max_ms / 1000 * sr)

            # Ensure background audio is long enough
            if len(bg_audio) > max_samples:
                # Choose a random start position
                start_idx = random.randint(0, len(bg_audio) - max_samples)
                bg_audio = bg_audio[start_idx : start_idx + max_samples]
            else:
                start_idx = 0  # Use entire background sound if shorter than max_samples


            # Normalize the main signal
            sig_max = np.max(np.abs(sig))  # Get the max amplitude of the main signal
            bg_audio_max = np.max(np.abs(bg_audio))  # Get the max amplitude of the background sound

            # Normalize both signals to the same amplitude
            sig = sig / sig_max  # Normalize main signal
            bg_audio = bg_audio / bg_audio_max  # Normalize background sound
            
            # Scale amplitude
            bg_audio = bg_audio * amplitude_limit


            # Ensure it matches the length of the main signal
            if len(bg_audio) < len(sig):
                bg_audio = np.pad(bg_audio, (0, len(sig) - len(bg_audio)), 'constant')

            # Mix background sound into the main signal
            sig = sig + bg_audio[:len(sig)]

            #Denormalize the signal
            sig = sig * sig_max

        return (sig, sr)

    

    def specgram(audio, Nft=512, fs2=11025) -> ndarray:
        """
        Compute a Spectrogram.

        :param aud: The audio signal as a tuple (signal, sample_rate).
        :param Nft: The number of points of the FFT.
        :param fs2: The sampling frequency.
        """
        sig, sr = audio

        # Resample the signal if the sampling rate is different
        if sr != fs2:
            sig, _ = AudioUtil.resample(audio, fs2)

        # Trim signal to make its length divisible by Nft
        L = len(sig)
        sig = sig[: L - L % Nft]

        # Reshape signal into a matrix with Nft-length segments as rows
        audiomat = np.reshape(sig, (L // Nft, Nft))

        # Apply Hamming window to each row
        audioham = audiomat * np.hamming(Nft)

        # Compute FFT row by row
        stft = np.fft.fft(audioham, axis=1)

        # Take only positive frequencies and compute magnitude
        stft = np.abs(stft[:, : Nft // 2].T)

        return stft

    def get_hz2mel(fs2=11025, Nft=512, Nmel=20) -> ndarray:
        """
        Get the hz2mel conversion matrix.

        :param fs2: The sampling frequency.
        :param Nft: The number of points of the FFT.
        :param Nmel: The number of mel bands.
        """
        mels = librosa.filters.mel(sr=fs2, n_fft=Nft, n_mels=Nmel)
        mels = mels[:, :-1]
        mels = mels / np.max(mels)

        return mels

    def melspectrogram(audio, Nmel=20, Nft=512, fs2=11025) -> ndarray:
        """
        Generate a Melspectrogram.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param Nmel: The number of mel bands.
        :param Nft: The number of points of the FFT.
        :param fs2: The sampling frequency.
        """
        sig, sr = audio

        # Resample the signal to the target sampling frequency
        if sr != fs2:
            sig, _ = AudioUtil.resample(audio, newsr=fs2)

        # Compute the spectrogram using the specgram function
        stft = AudioUtil.specgram((sig, fs2), Nft=Nft, fs2=fs2)

        # Generate the Mel filter bank
        mels = librosa.filters.mel(sr=fs2, n_fft=Nft, n_mels=Nmel)
        mels = mels[:, :-1]  # Remove the last column to match dimensions
        mels = mels / np.max(mels)  # Normalize the Mel filter bank

        # Compute the Melspectrogram
        melspec = np.dot(mels, np.abs(stft))

        return melspec

    def spectro_aug_timefreq_masking(
        spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1
    ) -> ndarray:
        """
        Augment the Spectrogram by masking out some sections of it in both the frequency dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent overfitting and to help the model generalise better. The masked sections are replaced with the mean value.


        :param spec: The spectrogram.
        :param max_mask_pct: The maximum percentage of the spectrogram to mask out.
        :param n_freq_masks: The number of frequency masks to apply.
        :param n_time_masks: The number of time masks to apply.
        """
        Nmel, n_steps = spec.shape
        mask_value = np.mean(spec)
        aug_spec = np.copy(spec)  # avoids modifying spec

        freq_mask_param = max_mask_pct * Nmel
        for _ in range(n_freq_masks):
            height = int(np.round(random.random() * freq_mask_param))
            pos_f = np.random.randint(Nmel - height)
            aug_spec[pos_f : pos_f + height, :] = mask_value

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            width = int(np.round(random.random() * time_mask_param))
            pos_t = np.random.randint(n_steps - width)
            aug_spec[:, pos_t : pos_t + width] = mask_value

        return aug_spec


class Feature_vector_DS:
    """
    Dataset of Feature vectors.
    """

    def __init__(
        #AJOUTER     
        self,
        dataset,

        flag_8bit = False,
        bit_sensitivity = 0,
        n_melvec=20, # Number of mel bands, y axis
        melvec_height=20, # Number of mel bands
        Nft=512, # Number of points of the FFT, x axis = samples_per_melvec
        samples_per_melvec=512, # Number of samples per melvec
        window_type = 'hamming',
        sr=11025, # sampling rate

        shift_pct=0, # percentage of total
        normalize=False, # Normalize the energy of the signal
        data_aug=None, # Data augmentation options
        pca=None, # PCA object
        bg_dataset = None,
        bg_amplitude_limit = 0,
        noise_sigma= 0.001,
        scaling_limit = 5,

    ):

        self.dataset = dataset
        self.Nft = Nft
        self.n_melvec = n_melvec
        self.melvec_height = melvec_height
        #self.samples_per_melvec = samples_per_melvec
        self.samples_per_melvec = Nft
        self.duration = n_melvec * Nft / sr * 1000
        self.sr = sr # sampling rate
        self.window_type = window_type
        self.flag_8bit = flag_8bit
        self.bit_sensitivity = bit_sensitivity


        self.shift_pct = shift_pct  # percentage of total
        self.normalize = normalize
        self.data_aug = data_aug
        self.data_aug_factor = 1

        if isinstance(self.data_aug, list):
            self.data_aug_factor += len(self.data_aug)
        else:
            self.data_aug = [self.data_aug]

        #self.ncol = int(
        #    self.duration * self.sr / (1e3 * self.Nft)
        #)  # number of columns in melspectrogram

        self.pca = pca
        self.bg_dataset = bg_dataset
        self.bg_amplitude_limit = bg_amplitude_limit
        self.noise_sigma = noise_sigma
        self.scaling_limit = scaling_limit

    def __len__(self) -> int:
        """
        Number of items in dataset.
        """
        return len(self.dataset) * self.data_aug_factor

    def get_audiosignal(self, cls_index: Tuple[str, int]) -> Tuple[ndarray, int]:
        """
        Get temporal signal of i'th item in dataset.

        :param cls_index: Class name and index.
        """
        audio_file = self.dataset[cls_index]
        aud_sr = AudioUtil.open(audio_file)
        
        aud_sr = AudioUtil.resample(aud_sr, self.sr)
        aud_sr = AudioUtil.pad_trunc(aud_sr, self.duration)

        
        if self.data_aug is not None:
            if "time_shift" in self.data_aug:
                aud_sr = AudioUtil.time_shift(aud_sr, self.shift_pct)  
                #print("time shift")          
            elif ("add_bg" in self.data_aug):
                ds_bg = self.bg_dataset
                aud_sr = AudioUtil.add_bg(
                    aud_sr,
                    ds_bg,
                    num_sources=1,
                    max_ms=self.duration,
                    amplitude_limit=self.bg_amplitude_limit,
                )
            elif "add_echo" in self.data_aug:
                #print("echo")
                aud_sr = AudioUtil.add_echo(aud_sr)
            elif "add_noise" in self.data_aug:
                #print("noise")
                aud_sr = AudioUtil.add_noise(aud_sr, sigma=self.noise_sigma)
            elif "scaling" in self.data_aug:
                #print("scaling")
                aud_sr = AudioUtil.scaling(aud_sr, scaling_limit=self.scaling_limit)

        # aud = AudioUtil.normalize(aud, target_dB=10)
        # aud = (aud[0] / np.max(np.abs(aud[0])), aud[1]) Problème si normalisation ici!

        aud = aud_sr[0]
        sr = aud_sr[1]

        # Convert to fixed point 16 bits - important
        aud = mcu.float2fixed(aud, maxval=1, dtype=np.int16)

        aud_sr = (aud, sr)  # Convert to tuple (signal, sample_rate)

        return aud_sr

    def __getitem__(self, cls_index: Tuple[str, int]) -> Tuple[ndarray, int]:
        """
        Get i'th item in dataset.

        :param cls_index: Class name and index.
        """
        #aud = self.get_audiosignal(cls_index) #POUR MA VERSION
        #get audio signal donne un tuple( signal, sample_rate)
        aud_sr = self.get_audiosignal(cls_index) #- RAPHAEL
        aud = aud_sr[0] #get the signal
        sr = aud_sr[1] #get the sample rate
        #Set audio type to int16
        aud = aud_sr[0].astype(np.int16)
        
        #melvec = AudioUtil.melspectrogram(aud, Nmel=self.n_melvec, Nft=self.Nft)

        melspec = mcu.melspectrogram(aud, N_melvec=self.n_melvec, melvec_height=self.melvec_height,
                                      samples_per_melvec=self.samples_per_melvec, N_Nft=self.Nft,
                                    window_type=self.window_type, sr=sr, 
                                    flag_8bit=self.flag_8bit, bit_sensitivity=self.bit_sensitivity)    


        if self.data_aug is not None:
            if "aug_melvec" in self.data_aug:
                melspec = AudioUtil.spectro_aug_timefreq_masking(
                    melspec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
                )

        melspec = mcu.fixed_array_to_float(melspec, 15)

        melspec = melspec.T
        melspec = melspec.flatten()
        melspec /= np.linalg.norm(melspec)  # Normalize the feature vector
        
        return melspec

    def display(self, cls_index: Tuple[str, int], show_img = False):
        """
        Play sound and display i'th item in dataset.

        :param cls_index: Class name and index.
        """
        audio = self.get_audiosignal(cls_index)
        AudioUtil.play(audio)
        if show_img:
            plt.figure(figsize=(4, 3))
            plt.imshow(
                mcu.melspectrogram(audio, Nmel=self.n_melvec, Nft=self.Nft),
                cmap="jet",
                origin="lower",
                aspect="auto",
            )
                
            plt.colorbar()
            plt.title(audio)
            plt.title(self.dataset.__getname__(cls_index))
            plt.show()

    def mod_data_aug(self, data_aug) -> None:
        """
        Modify the data augmentation options.

        :param data_aug: The new data augmentation options.
        """
        self.data_aug = data_aug
        self.data_aug_factor = 1
        if isinstance(self.data_aug, list):
            self.data_aug_factor += len(self.data_aug)
        else:
            self.data_aug = [self.data_aug]
