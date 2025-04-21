import numpy as np
import librosa
import matplotlib.pyplot as plt


def fixed_to_float(x, e):
    """Convert signal from integer fixed format to floating point."""
    c = abs(x)
    sign = 1
    if x < 0:
        # convert back from two's complement
        c = x - 1
        c = ~c
        sign = -1
    f = (1.0 * c) / (2**e)
    f = f * sign
    return f

def fixed_array_to_float(x: np.ndarray, e: int) -> np.ndarray:
    """
    Convert a NumPy array of fixed-point integers to floating-point values.
    
    Parameters:
    - x (np.ndarray): Input array of integers (e.g., int16).
    - e (int): Exponent used for fixed-point scaling (i.e., 1 / 2**e).
    
    Returns:
    - np.ndarray: Array of float64 values.
    """
    x = np.asarray(x)
    
    # Handle sign: create a mask for negative numbers
    negative_mask = x < 0

    # Convert to unsigned equivalent for bitwise operations
    c = np.abs(x).astype(np.uint64)

    # Apply two's complement conversion manually for negative values
    c[negative_mask] = ~ (x[negative_mask] - 1)

    # Scale to float
    f = c / (2 ** e)

    # Restore sign
    f[negative_mask] *= -1

    return f


def float2fixed(audio, maxval=1, dtype = np.int16):
    """
    Convert signal from float format to integer fixed point, here q15_t.
    Divide by this maximal value to increase range.
    In q15_t, the first bit is for the sign and the 15 others are for quantization of the signal from 0 to 1
    """
    q = dtype(0).nbytes * 8 - 1
    return (np.round((audio / maxval) * (2**q - 1))).astype(dtype)


def Spectrogram_Compute(samples, melvec, window, hz2mel_mat, melvec_height = 20, samples_per_melvec = 512, dtype = np.int16):
    """
    Args:
        samples (input): numpy array of length SAMPLES_PER_MELVEC in DTYPE format
        melvec (output): numpy array of length MELVEC_HEIGHT in DTYPE format
    Post: the melvec array is filled with the mel vector of the input samples
    """
    dtype_fixed = np.int16
    num_bits = dtype_fixed(0).nbytes * 8

    # Buffers
    buf = np.zeros(samples_per_melvec, dtype=dtype_fixed)
    buf_fft = np.zeros(2 * samples_per_melvec, dtype=dtype_fixed)
    # STEP 1 : Windowing of input samples --> Pointwise product
    #vector_mult(samples, hamming_window, buf, SAMPLES_PER_MELVEC)


    buf = (np.int32(samples) * np.int32(window) >> (num_bits-1)).astype(dtype_fixed)  # Perform multiplication in int32
    # STEP 2 : Compute the FFT of the windowed samples
    buf_fft2 = np.fft.fft(buf, n = samples_per_melvec) 
    buf_fft3 = np.zeros(samples_per_melvec*2)
    buf_fft3[0::2] = buf_fft2.real /512    #(8800/4505610) # Mise à l'échelle comme arm_fft, pas //
    buf_fft3[1::2] = buf_fft2.imag /512    #(8800/4505610) 
    buf_fft = buf_fft3.astype(dtype_fixed)


    # STEP 3 : Compute the magnitude of the FFT
    # STEP 3.1 Find the extremum value
    vmax = max(buf_fft)
    pIndex = np.argmax(buf_fft)

    # STEP 3.2 Normalize the FFT
    for i in range(samples_per_melvec):
        buf[i] = ((np.int32(buf_fft[i]) << (num_bits-1)) // np.int32(vmax)).astype(dtype_fixed) #DIVISION BY 0 ERROR

    # STEP 3.3 Compute the complex magnitude
    for i in range(samples_per_melvec//2):
        real = np.int32(buf[2*i])
        imag = np.int32(buf[2*i+1])
        
        #buf[i] = (np.sqrt(real * real + imag * imag) //2).astype(DTYPE)
        acc0 = np.int64(real * real)
        acc1 = np.int64(imag * imag)
        buf[i] = np.sqrt(np.int64(np.int16(np.int64(acc0 + acc1) >> (num_bits+1)))<< (num_bits-1)).astype(dtype_fixed)

    # STEP 3.4 Denormalize the magnitude
    for i in range(samples_per_melvec//2):
        buf[i] = (np.int32(buf[i]) * vmax >> (num_bits-1)).astype(dtype_fixed)
    
    
    # STEP 4 : Compute the Mel vector
    #melvec = hz2mel @ buf
    #hz2mel has MELVEC_HEIGHT rows and SAMPLES_PER_MELVEC//2 columns
    #buf has SAMPLES_PER_MELVEC//2 rows and 1 column
    #melvec has MELVEC_HEIGHT rows and 1 column

    for i in range(melvec_height):
        sum = 0
        for j in range(samples_per_melvec//2):
            sum += (np.int32(hz2mel_mat[i][j]) * np.int32(buf[j]))
        melvec[i] = (sum >> (num_bits - 1)).astype(dtype_fixed)
    
    return


def plot_specgram(
    specgram,
    ax,
    is_mel=False,
    title=None,
    xlabel="Time [s]",
    ylabel="Frequency [Hz]",
    textlabel = "",
    cmap="jet",
    cb=True,
    tf=None,
    invert=True,
):
    """
    Plot a spectrogram (2D matrix) in a chosen axis of a figure.
    Inputs:
        - specgram = spectrogram (2D array)
        - ax       = current axis in figure
        - title
        - xlabel
        - ylabel
        - cmap
        - cb       = show colorbar if True
        - tf       = final time in xaxis of specgram
    """
    if tf is None:
        tf = specgram.shape[1]

    if is_mel:
        ylabel = "Frequency [Mel]"
        im = ax.imshow(
            specgram, cmap=cmap, aspect="auto", extent=[0, tf, specgram.shape[0], 0]
        )
    else:
        im = ax.imshow(
            specgram,
            cmap=cmap,
            aspect="auto",
            extent=[0, tf, int(specgram.size / tf), 0],
        )
    if invert:
        ax.invert_yaxis()
    fig = plt.gcf()
    if cb:
        fig.colorbar(im, ax=ax)
    # cbar.set_label('log scale', rotation=270)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.subplots_adjust(bottom=0.3)
    ax.text(0, -0.15, f'{textlabel}', ha='left', va='top', transform=ax.transAxes, family='monospace')

    return None
    
def melspectrogram(audio, N_melvec=20, melvec_height=20, samples_per_melvec=512, N_Nft=512, window_type="hamming", sr=11025, dtype=np.int16):
    if window_type == "hamming":
        hamming_window = np.hamming(N_Nft)
        window = float2fixed(hamming_window, maxval = 1, dtype=np.int16)
    
    elif window_type == "hanning":
        hanning_window = np.hanning(N_Nft)
        window = float2fixed(hanning_window, maxval = 1, dtype=np.int16)
    
    elif window_type == "blackman":
        blackman_window = np.blackman(N_Nft)
        window = float2fixed(blackman_window, maxval = 1, dtype=np.int16)
    
    elif window_type == "rectangular":
         window = np.ones(N_Nft)
         window = float2fixed(window, maxval = 1, dtype=np.int16)
    
    elif window_type == "triangular":
        triangular_window = np.bartlett(N_Nft)
        window = float2fixed(triangular_window, maxval = 1, dtype=np.int16)
    else:
        raise ValueError("Window type not supported")
    

    # Hz to Mel conversion
    mel = librosa.filters.mel(sr=sr, n_fft=N_Nft, n_mels=melvec_height)  # WARNING: in MCU tables, sr = 11025
    #print("WARNING: Mel filter bank is computed with sr = 11025")
    mel = mel[:,:-1]
    mel = mel/np.max(mel)
    hz2mel_mat = float2fixed(mel, maxval = 1, dtype=np.int16)

    melspec = np.zeros([N_melvec, melvec_height], dtype=np.int16)

    for i in range(N_melvec):
        samples = audio[i*samples_per_melvec:(i+1)*samples_per_melvec]

        #To solve off-by-one error
        if len(samples) < samples_per_melvec:
            pad_width = samples_per_melvec - len(samples)
            samples = np.pad(samples, (0, pad_width), mode='constant')  # or mode='reflect', etc.

        Spectrogram_Compute(samples, melspec[i], window, hz2mel_mat, melvec_height, samples_per_melvec, dtype)
    
    if dtype == np.int8:
        melspec = (melspec >> 0).astype(np.int16)
    return melspec


if __name__ == "__main__":
    import sys
    import soundfile as sf
    from audio_student import AudioUtil
    from classification.datasets import Dataset
    #classification/src/classification/datasets/__init__.py

    np.set_printoptions(threshold=sys.maxsize)

    DTYPE = np.int16
    SAMPLES_PER_MELVEC = 512
    N_MELVEC = 20 #*5
    MELVEC_HEIGHT = 20
    N_NFT = 512
    WINDOW_TYPE = "hamming"
    fs = 10200

    # generate a 512 array of a pure sine at 2500Hz, f=44100Hz
    # f = 2500
    # fs = 44100
    # t = np.arange(0, SAMPLES_PER_MELVEC*20, 1)
    # data = (np.sin(2 * np.pi * f * t / fs)* 32767).astype(DTYPE)

    aud, sr = sf.read("/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/sounds/recorded_sounds/totalset/fire/fire_1_1.wav" )
                      
    
    #dataset = Dataset(folder="/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/classification/src/classification/datasets/sounds/recorded_sounds/classes/test/")
    #classnames = dataset.list_classes()
    #cls_index = (["chainsaw",0])
    #audio_file = dataset[cls_index]
    #aud, sr = AudioUtil.open(audio_file)
    #print("Initial sr: ", sr)

    aud, sr = AudioUtil.resample((aud,sr), fs)
    print("Resampled sr: ", sr)
    #aud = np.array(aud, dtype=DTYPE)
    #aud = AudioUtil.pad_trunc(aud, dyrat)

    # Normalize the audio signal - TO DO NOT
    # aud = (aud - np.mean(aud)) / np.std(aud)


    # Convert to fixed point 16 bits
    aud = float2fixed(aud, maxval=1, dtype=np.int16)

    melspec = melspectrogram(aud, N_melvec=N_MELVEC, melvec_height=MELVEC_HEIGHT, samples_per_melvec=SAMPLES_PER_MELVEC, N_Nft=N_NFT, window_type=WINDOW_TYPE, sr=sr, dtype=DTYPE)    
    DTYPE2 = np.int8
    melspec2 = melspectrogram(aud, N_melvec=N_MELVEC, melvec_height=MELVEC_HEIGHT, samples_per_melvec=SAMPLES_PER_MELVEC, N_Nft=N_NFT, window_type=WINDOW_TYPE, sr=sr, dtype=DTYPE2)    

    
    #if DTYPE == np.int16:
    #    print("ok1")
    melspec = fixed_array_to_float(melspec, 15)
    #if DTYPE2 == np.int8:
    #    print("ok2")
    melspec2 = fixed_array_to_float(melspec2, 15)

    #Compare if both melspecs are the same
    if np.array_equal(melspec, melspec2):
        print("Both melspecs are the same")
    else:
        print("Melspecs are different")

    #Normalize the melspec between 0 and 1
    melspec = melspec.T
    melspec = melspec.flatten()      
    #melspec /= np.linalg.norm(melspec)

    fig, ax = plt.subplots(figsize=(4, 3))

    # Display the mel spectrogram for a single class and image index
    img = ax.imshow(melspec.reshape((MELVEC_HEIGHT, -1)), cmap="jet", origin="lower", aspect="auto")
    ax.set_xlabel("")
    ax.set_ylabel("Mel bins")

    # Add colorbar
    plt.colorbar(img, ax=ax, orientation='vertical')
    plt.show()
    
    melspec2 = melspec2.T
    melspec2 = melspec2.flatten()      
    #melspec2 /= np.linalg.norm(melspec2)


        
    #Compare if both melspecs are the same
    if np.array_equal(melspec, melspec2):
        print("Both melspecs are the same")
    else:
        print("Melspecs are different")
    # Plot the mel spectrogram 2
    fig, ax = plt.subplots(figsize=(4, 3))
    # Display the mel spectrogram for a single class and image index
    img = ax.imshow(melspec2.reshape((MELVEC_HEIGHT, -1)), cmap="jet", origin="lower", aspect="auto")
    ax.set_xlabel("")
    ax.set_ylabel("Mel bins")
    # Add colorbar
    plt.colorbar(img, ax=ax, orientation='vertical')
    plt.show()


