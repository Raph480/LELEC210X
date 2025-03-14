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

    def open(audio_file) -> Tuple[ndarray, int]:
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

        return (resig, newsr)

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
            begin_len = 0
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

    def time_shift(audio, shift_limit=0.4) -> Tuple[ndarray, int]:
        """
        Shifts the signal to the left or right by some percent. Values at the end are 'wrapped around' to the start of the transformed signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param shift_limit: The percentage (between 0.0 and 1.0) by which to circularly shift the signal.
        """
        sig, sr = audio
        sig_len = len(sig)
        shift_amt = int(random.random() * shift_limit * sig_len)
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
        
        sig, sr = audio  # Unpack the signal and sample rate

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

            # Scale amplitude
            bg_audio = bg_audio * amplitude_limit

            # Ensure it matches the length of the main signal
            if len(bg_audio) < len(sig):
                bg_audio = np.pad(bg_audio, (0, len(sig) - len(bg_audio)), 'constant')

            # Mix background sound into the main signal
            sig = sig + bg_audio[:len(sig)]

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
        Nft=512, # Number of points of the FFT, x axis
        nmel=20, # Number of mel bands, y axis
        duration=500, #duration of the audio signal 
        shift_pct=0.4, # percentage of total
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
        self.nmel = nmel
        self.duration = duration  # ms
        self.sr = 10200 # sampling rate
        self.shift_pct = shift_pct  # percentage of total
        self.normalize = normalize
        self.data_aug = data_aug
        self.data_aug_factor = 1
        if isinstance(self.data_aug, list):
            self.data_aug_factor += len(self.data_aug)
        else:
            self.data_aug = [self.data_aug]
        self.ncol = int(
            self.duration * self.sr / (1e3 * self.Nft)
        )  # number of columns in melspectrogram
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
        aud = AudioUtil.open(audio_file)
        aud = AudioUtil.resample(aud, self.sr)
        aud = AudioUtil.pad_trunc(aud, self.duration)
        if self.data_aug is not None:
            if "time_shift" in self.data_aug:
                aud = AudioUtil.time_shift(aud, self.shift_pct)  
                #print("time shift")          
            if ("add_bg" in self.data_aug):
                #print("add bg")
                ds_bg = self.bg_dataset
                aud = AudioUtil.add_bg(
                    aud,
                    ds_bg,
                    num_sources=1,
                    max_ms=self.duration,
                    amplitude_limit=self.bg_amplitude_limit,
                )
            if "add_echo" in self.data_aug:
                #print("echo")
                aud = AudioUtil.add_echo(aud)
            if "add_noise" in self.data_aug:
                #print("noise")
                aud = AudioUtil.add_noise(aud, sigma=self.noise_sigma)
            if "scaling" in self.data_aug:
                #print("scaling")
                aud = AudioUtil.scaling(aud, scaling_limit=self.scaling_limit)

        # aud = AudioUtil.normalize(aud, target_dB=10)
        aud = (aud[0] / np.max(np.abs(aud[0])), aud[1])
        return aud

    def __getitem__(self, cls_index: Tuple[str, int]) -> Tuple[ndarray, int]:
        """
        Get i'th item in dataset.

        :param cls_index: Class name and index.
        """
        aud = self.get_audiosignal(cls_index)
        sgram = AudioUtil.melspectrogram(aud, Nmel=self.nmel, Nft=self.Nft)
        if self.data_aug is not None:
            if "aug_sgram" in self.data_aug:
                sgram = AudioUtil.spectro_aug_timefreq_masking(
                    sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
                )

        sgram_crop = sgram[:, : self.ncol]
        fv = sgram_crop.flatten()  # feature vector
        if self.normalize:
            fv /= np.linalg.norm(fv)
        if self.pca is not None:
            fv = self.pca.transform([fv])[0]
        return fv

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
                AudioUtil.melspectrogram(audio, Nmel=self.nmel, Nft=self.Nft),
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
