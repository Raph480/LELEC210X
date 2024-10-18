#!/usr/bin/env python
#
# Copyright 2021 UCLouvain.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

import numpy as np
from gnuradio import gr


def demodulate(y, B, R, Fdev):
    """
    Non-coherent demodulator.
    #FIRST DUMMY VERSION
    nb_syms = int(len(y) / R)
    bits_hat = np.zeros(nb_syms, dtype=int)
    return bits_hat  # TODO
    """
    """
    Vectorized non-coherent demodulator for Continuous-Phase FSK.
    
    :param y: The received signal, (N * R,).
    :param B: Bit rate (B = 1/T).
    :param R: Oversampling factor.
    :param Fdev: Frequency deviation (Delta_f).
    :return: The demodulated bit sequence, (N,).
    """
    nb_syms = len(y) // R  # Number of CPFSK symbols in y

    # Reshape received signal to separate each symbol's samples
    y = np.resize(y, (nb_syms, R))

    # Generate reference waveforms (s0 for symbol 0, s1 for symbol 1)
    t = np.arange(R) / (R * B)  # Time vector for one symbol period (normalized)
    s0 = np.exp(1j * 2 * np.pi * (-Fdev) * t)  # Reference waveform for symbol 0
    s1 = np.exp(1j * 2 * np.pi * Fdev * t)    # Reference waveform for symbol 1

    # Compute the correlations for all symbols at once
    r0 = np.sum(y * np.conj(s0), axis=1) / R  # Correlation with reference for symbol 0
    r1 = np.sum(y * np.conj(s1), axis=1) / R  # Correlation with reference for symbol 1

    # Non-coherent detection: Compare magnitudes of r0 and r1 for all symbols
    bits_hat = (np.abs(r1) > np.abs(r0)).astype(int)  # Symbol detection in one step

    return bits_hat


class demodulation(gr.basic_block):
    """
    docstring for block demodulation
    """

    def __init__(self, drate, fdev, fsamp, payload_len, crc_len):
        self.drate = drate
        self.fdev = fdev
        self.fsamp = fsamp
        self.frame_len = payload_len + crc_len
        self.osr = int(fsamp / drate)

        gr.basic_block.__init__(
            self, name="Demodulation", in_sig=[np.complex64], out_sig=[np.uint8]
        )

    def symbols_to_bytes(self, symbols):
        """
        Converts symbols (bits here) to bytes
        """
        if len(symbols) == 0:
            return []

        n_bytes = int(len(symbols) / 8)
        bitlists = np.array_split(symbols, n_bytes)
        out = np.zeros(n_bytes).astype(np.uint8)

        for i, l in enumerate(bitlists):
            for bit in l:
                out[i] = (out[i] << 1) | bit

        return out

    def forecast(self, noutput_items, ninput_items_required):
        """
        input items are samples (with oversampling factor)
        output items are bytes
        """
        ninput_items_required[0] = noutput_items * self.osr * 8

    def general_work(self, input_items, output_items):
        n_syms = len(output_items[0]) * 8
        buf_len = n_syms * self.osr

        y = input_items[0][:buf_len]
        self.consume_each(buf_len)

        s = demodulate(y, self.drate, self.osr, self.fdev)
        b = self.symbols_to_bytes(s)
        output_items[0][: len(b)] = b

        return len(b)
