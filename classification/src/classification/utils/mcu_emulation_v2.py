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

def float2fixed(audio, maxval=1, dtype = np.int16):
    """
    Convert signal from float format to integer fixed point, here q15_t.
    Divide by this maximal value to increase range.
    In q15_t, the first bit is for the sign and the 15 others are for quantization of the signal from 0 to 1
    """
    q = dtype(0).nbytes * 8 - 1
    return (np.round((audio / maxval) * (2**q - 1))).astype(dtype)


def Spectrogram_Compute(samples, melvec, window, hz2mel_mat, melvec_length = 20, samples_per_melvec = 512):
    """
    Args:
        samples (input): numpy array of length SAMPLES_PER_MELVEC in DTYPE format
        melvec (output): numpy array of length MELVEC_LENGTH in DTYPE format
    Post: the melvec array is filled with the mel vector of the input samples
    """
    num_bits = (np.int16)(0).nbytes * 8

    # Buffers
    buf = np.zeros(samples_per_melvec, dtype=np.int16)
    buf_fft = np.zeros(2 * samples_per_melvec, dtype=np.int16)

    # STEP 1 : Windowing of input samples --> Pointwise product
    #vector_mult(samples, hamming_window, buf, SAMPLES_PER_MELVEC)
    buf = (np.int32(samples) * np.int32(window) >> (num_bits-1)).astype(np.int16)  # Perform multiplication in int32

    # STEP 2 : Compute the FFT of the windowed samples
    buf_fft2 = np.fft.fft(buf, n = samples_per_melvec) 
    buf_fft3 = np.zeros(samples_per_melvec*2)
    buf_fft3[0::2] = buf_fft2.real /512    #(8800/4505610) # Mise à l'échelle comme arm_fft, pas //
    buf_fft3[1::2] = buf_fft2.imag /512    #(8800/4505610) 
    buf_fft = buf_fft3.astype(np.int16)

    # STEP 3 : Compute the magnitude of the FFT
    # STEP 3.1 Find the extremum value
    vmax = max(buf_fft)
    pIndex = np.argmax(buf_fft)

    # STEP 3.2 Normalize the FFT
    for i in range(samples_per_melvec):
        buf[i] = ((np.int32(buf_fft[i]) << (num_bits-1)) // np.int32(vmax)).astype(np.int16)

    # STEP 3.3 Compute the complex magnitude
    for i in range(samples_per_melvec//2):
        real = np.int32(buf[2*i])
        imag = np.int32(buf[2*i+1])
        #buf[i] = (np.sqrt(real * real + imag * imag) //2).astype(np.int16)
        acc0 = np.int64(real * real)
        acc1 = np.int64(imag * imag)
        buf[i] = np.sqrt(np.int16(np.int64(acc0 + acc1) >> (num_bits+1))<< (num_bits-1)).astype(np.int16)

    # STEP 3.4 Denormalize the magnitude
    for i in range(samples_per_melvec//2):
        buf[i] = (np.int32(buf[i]) * vmax >> (num_bits-1)).astype(np.int16)
    
    # STEP 4 : Compute the Mel vector
    #melvec = hz2mel @ buf
    #hz2mel has MELVEC_LENGTH rows and SAMPLES_PER_MELVEC//2 columns
    #buf has SAMPLES_PER_MELVEC//2 rows and 1 column
    #melvec has MELVEC_LENGTH rows and 1 column

    for i in range(melvec_length):
        sum = 0
        for j in range(samples_per_melvec//2):
            sum += (np.int32(hz2mel_mat[i][j]) * np.int32(buf[j]))
        melvec[i] = (sum >> (num_bits - 1)).astype(np.int16)

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
    
def melspectrogram(audio, N_melvec=20, melvec_length=20, samples_per_melvec=512, N_Nft=512, window_type="hamming", fs2=11025, flag_8bit=False):
    raw_audio_from_adc = (audio + (1 << 14)) >> 3  # Recover the original signal from the ADC 
    audio = (raw_audio_from_adc - (1 << 11)) << 4  # Conversion to the recently used format (to avoid empty melspecs if 8-bit final cast)
    
    if window_type == "hamming":
        hamming_window = np.hamming(N_Nft)
        window = float2fixed(hamming_window, maxval = 1, dtype=np.int16)
    else:
        raise ValueError("Window type not supported")

    # Hz to Mel conversion
    mel = librosa.filters.mel(sr=fs2, n_fft=N_Nft, n_mels=melvec_length)  # WARNING: in MCU tables, sr = 11025
    if (fs2==11025): print("WARNING: Mel filter bank is computed with sr = 11025")
    mel = mel[:,:-1]
    mel = mel/np.max(mel)
    hz2mel_mat = float2fixed(mel, maxval = 1, dtype=np.int16)

    melspec = np.zeros([N_melvec, melvec_length], dtype=np.int16)

    for i in range(N_melvec):
        samples = audio[i*samples_per_melvec:(i+1)*samples_per_melvec]
        Spectrogram_Compute(samples, melspec[i], window, hz2mel_mat, melvec_length, samples_per_melvec)

    if flag_8bit:
        melspec = melspec >> 8  # Convert to 8-bit format

    return melspec


if __name__ == "__main__":
    import sys
    import soundfile as sf

    np.set_printoptions(threshold=sys.maxsize)

    SAMPLES_PER_MELVEC = 512
    N_MELVEC = 20
    MELVEC_LENGTH = 20
    N_NFT = 512
    WINDOW_TYPE = "hamming"
    FLAG_8BIT = False
    # fs = 11025 

    # # generate a 512 array of a pure sine at 2500Hz, f=44100Hz
    # f = 2500
    # fs = 44100
    # t = np.arange(0, SAMPLES_PER_MELVEC, 1)
    # data = (np.sin(2 * np.pi * f * t / fs)* 32767).astype(np.int16)
    

    data, fs = sf.read("chainsaw_low_20-000.wav", dtype=np.int16)

    melspec = melspectrogram(data, N_melvec=N_MELVEC, melvec_length=MELVEC_LENGTH, samples_per_melvec=SAMPLES_PER_MELVEC, N_Nft=N_NFT, window_type=WINDOW_TYPE, fs2=fs, flag_8bit=FLAG_8BIT)    

    print(melspec[0])
    plot_specgram(melspec.T, plt.gca(), is_mel=True, title="Mel Spectrogram", xlabel="Time [s]", ylabel="Frequency [Mel]", textlabel="Mel Spectrogram", cmap="jet", cb=True, tf=N_MELVEC)
    plt.show()