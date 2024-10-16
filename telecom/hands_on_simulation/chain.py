from typing import Optional

import numpy as np

BIT_RATE = 50e3
PREAMBLE = np.array([int(bit) for bit in f"{0xAAAAAAAA:0>32b}"])
SYNC_WORD = np.array([int(bit) for bit in f"{0x3E2A54B7:0>32b}"])


class Chain:
    name: str = ""

    # Communication parameters
    bit_rate: float = BIT_RATE
    freq_dev: float = BIT_RATE / 4

    osr_tx: int = 64
    osr_rx: int = 8

    preamble: np.ndarray = PREAMBLE
    sync_word: np.ndarray = SYNC_WORD

    payload_len: int = 50  # Number of bits per packet

    # Simulation parameters
    n_packets: int = 100  # Number of sent packets

    # Channel parameters
    sto_val: float = 0
    sto_range: float = 10 / BIT_RATE  # defines the delay range when random

    cfo_val: float = 0
    cfo_range: float = (
        10000  # defines the CFO range when random (in Hz) #(1000 in old repo)
    )

    snr_range: np.ndarray = np.arange(-10, 25)

    # Lowpass filter parameters
    numtaps: int = 100
    cutoff: float = BIT_RATE * osr_rx / 2.0001  # or 2*BIT_RATE,...

    # Tx methods

    def modulate(self, bits: np.array) -> np.array:
        """
        Modulates a stream of bits of size N
        with a given TX oversampling factor R (osr_tx).

        Uses Continuous-Phase FSK modulation.

        :param bits: The bit stream, (N,).
        :return: The modulates bit sequence, (N * R,).
        """
        fd = self.freq_dev  # Frequency deviation, Delta_f
        B = self.bit_rate  # B=1/T
        h = 2 * fd / B  # Modulation index
        R = self.osr_tx  # Oversampling factor

        x = np.zeros(len(bits) * R, dtype=np.complex64)
        ph = 2 * np.pi * fd * (np.arange(R) / R) / B  # Phase of reference waveform

        phase_shifts = np.zeros(
            len(bits) + 1
        )  # To store all phase shifts between symbols
        phase_shifts[0] = 0  # Initial phase

        for i, b in enumerate(bits):
            x[i * R : (i + 1) * R] = np.exp(1j * phase_shifts[i]) * np.exp(
                1j * (1 if b else -1) * ph
            )  # Sent waveforms, with starting phase coming from previous symbol
            phase_shifts[i + 1] = phase_shifts[i] + h * np.pi * (
                1 if b else -1
            )  # Update phase to start with for next symbol

        return x

    # Rx methods
    bypass_preamble_detect: bool = False

    def preamble_detect(self, y: np.array) -> Optional[int]:
        """
        Detects the preamlbe in a given received signal.

        :param y: The received signal, (N * R,).
        :return: The index where the preamble starts,
            or None if not found.
        """
        raise NotImplementedError

    bypass_cfo_estimation: bool = False

    def cfo_estimation(self, y: np.array) -> float:
        """
        Estimates the CFO based on the received signal.

        :param y: The received signal, (N * R,).
        :return: The estimated CFO.
        """
        raise NotImplementedError

    bypass_sto_estimation: bool = False

    def sto_estimation(self, y: np.array) -> float:
        """
        Estimates the STO based on the received signal.

        :param y: The received signal, (N * R,).
        :return: The estimated STO.
        """
        raise NotImplementedError

    def demodulate(self, y: np.array) -> np.array:
        """
        Demodulates the received signal using non-coherent detection.

        :param y: The received signal, (N * R,).
        :return: The demodulated bit sequence, (N,).
        """
        fd = self.freq_dev  # Frequency deviation, Delta_f
        B = self.bit_rate  # Bit rate B=1/T
        R = self.osr_rx  # Oversampling factor at RX
        N = len(y) // R  # Number of symbols in the signal

        bits = np.zeros(N, dtype=np.int8)  # Output bits

        # Precompute the phase for symbol waveforms (symbol 0 and symbol 1)
        t = np.arange(R) / (R * B)  # Time vector for one symbol period (normalized)
        s0 = np.exp(1j * 2 * np.pi * (-fd) * t)  # Reference waveform for symbol 0
        s1 = np.exp(1j * 2 * np.pi * fd * t)  # Reference waveform for symbol 1

        #For each symbol
        for k in range(N):
            # Extract the received samples corresponding to the current symbol
            y_k = y[k * R:(k + 1) * R]

            # Compute correlation with both reference waveforms
            r0 = np.sum(y_k * np.conj(s0)) / R  # Correlate with symbol 0
            r1 = np.sum(y_k * np.conj(s1)) / R  # Correlate with symbol 1

            # Non-coherent detection: Compare the magnitudes of r0 and r1
            if np.abs(r1) > np.abs(r0):
                bits[k] = 1  # Symbol 1 is detected
            else:
                bits[k] = 0  # Symbol 0 is detected

        return bits



class BasicChain(Chain):
    name = "Basic Tx/Rx chain"

    cfo_val, sto_val = np.nan, np.nan  # CFO and STO are random

    bypass_preamble_detect = True

    def preamble_detect(self, y):
        """
        Detect a preamble computing the received energy (average on a window).
        """
        L = 4 * self.osr_rx
        y_abs = np.abs(y)

        for i in range(0, int(len(y) / L)):
            sum_abs = np.sum(y_abs[i * L : (i + 1) * L])
            if sum_abs > (L - 1):  # fix threshold
                return i * L

        return None

    bypass_cfo_estimation = True

    def cfo_estimation(self, y: np.array) -> float:
        """
        Estimates CFO using Moose algorithm, on first samples of preamble.
    
        :param y: The received signal, (N * R,).
        :return: The estimated CFO.
        """
        # Constants
        N = 4  # Block size in bits (4 bits)
        RRX = self.osr_rx  # Receiver oversampling factor
        B = self.bit_rate  # Bit rate (B = 1/T)
    
        # Block size in samples
        Nt = N * RRX  # N bits oversampled by RRX factor

        # Extract the first two blocks of size Nt from the received signal
        y1 = y[:Nt]         # First block
        y2 = y[Nt:2*Nt]     # Second block

        # Moose estimator calculation: alpha
        numerator = np.sum(y2 * np.conj(y1))  # Sum of y[l + Nt] * conj(y[l])
        denominator = np.sum(np.abs(y1)**2)   # Sum of |y[l]|^2
        alpha = numerator / denominator       # Estimate of alpha
    
        # Estimate the frequency offset (CFO)
        delta_f_c = np.angle(alpha) / (2 * np.pi * Nt / (RRX * B))  # CFO estimate in Hz

        return delta_f_c


    bypass_sto_estimation = True

    def sto_estimation(self, y):
        """
        Estimates symbol timing (fractional) based on phase shifts.
        """
        R = self.osr_rx

        # Computation of derivatives of phase function
        phase_function = np.unwrap(np.angle(y))
        phase_derivative_1 = phase_function[1:] - phase_function[:-1]
        phase_derivative_2 = np.abs(phase_derivative_1[1:] - phase_derivative_1[:-1])

        sum_der_saved = -np.inf
        save_i = 0
        for i in range(0, R):
            sum_der = np.sum(phase_derivative_2[i::R])  # Sum every R samples

            if sum_der > sum_der_saved:
                sum_der_saved = sum_der
                save_i = i

        return np.mod(save_i + 1, R)

    def demodulate(self, y: np.array) -> np.array:
        """
        Non-coherent demodulator for Continuous-Phase FSK.
    
        :param y: The received signal, (N * R,).
        :return: The demodulated bit sequence, (N,).
        """
        R = self.osr_rx  # Receiver oversampling factor
        nb_syms = len(y) // R  # Number of CPFSK symbols in y

        # Reshape received signal to separate each symbol's samples
        y = np.resize(y, (nb_syms, R))

        # Generate reference waveforms (s0 for symbol 0, s1 for symbol 1)
        fd = self.freq_dev  # Frequency deviation (Delta_f)
        B = self.bit_rate  # Bit rate (B = 1/T)
    
        t = np.arange(R) / (R * B)  # Time vector for one symbol period (normalized)
        s0 = np.exp(1j * 2 * np.pi * (-fd) * t)  # Reference waveform for symbol 0
        s1 = np.exp(1j * 2 * np.pi * fd * t)    # Reference waveform for symbol 1

        # Output bit sequence
        bits_hat = np.zeros(nb_syms, dtype=int)

        # Demodulate each symbol
        for k in range(nb_syms):
            y_k = y[k]  # Get the samples for the k-th symbol
        
            # Compute correlation with both reference waveforms
            r0 = np.sum(y_k * np.conj(s0)) / R  # Correlation with reference for symbol 0
            r1 = np.sum(y_k * np.conj(s1)) / R  # Correlation with reference for symbol 1
        
            # Non-coherent detection: Compare magnitudes of r0 and r1
            if np.abs(r1) > np.abs(r0):
                bits_hat[k] = 1  # Symbol 1 detected
            else:
                bits_hat[k] = 0  # Symbol 0 detected

        return bits_hat

