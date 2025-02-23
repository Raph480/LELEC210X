"""
Read a measurements file generated by eval_limesdr_fpga.py
and plots the PER/SNR curve, plus CFO values.
"""

import sys
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

if __name__ == "__main__":
    expected_payload = np.arange(100, dtype=np.uint8)
    num_bits = len(expected_payload) * 8

    def propagation_model(r, r0, L0, n):
        return L0 + 10 * n * np.log10(r / r0)

    def get_pd_from_file(file_name):
        data = defaultdict(list)
        with open(file_name) as f:
            for line in f.read().splitlines():
                if line.startswith("CFO"):
                    cfo, sto = line.split(",")
                    data["cfo"].append(float(cfo.split("=")[1]))
                    data["sto"].append(int(sto.split("=")[1]))
                elif line.startswith("SNR"):
                    snr, txp = line.split(",")
                    data["snr"].append(float(snr.split("=")[1]))
                    data["txp"].append(int(txp.split("=")[1]))
                elif line.startswith("packet"):
                    *_, payload = line.split(",", maxsplit=2)
                    payload = list(map(int, payload.split("=")[1][1:-1].split(",")))
                    ber = (
                        np.unpackbits(
                            expected_payload ^ np.array(payload, dtype=np.uint8)
                        ).sum()
                        / num_bits
                    )
                    invalid = 1 if ber > 0 else 0
                    data["ber"].append(ber)
                    data["invalid"].append(invalid)

            df = pd.DataFrame.from_dict(data)
            return df
    
    # list files in the directory
    file_list = os.listdir()
    list_of_df = []

    for file_name in file_list:
        if file_name.startswith("measurements_"):
            parts = file_name.split("_")
            tx_power = int(parts[1])
            distance = int(parts[2].split(".")[0])
            df = get_pd_from_file(file_name)
            df["distance"] = distance/10
            df["txp"] = tx_power
            list_of_df.append(df) 

    # concatenate all dataframes
    df = pd.concat(list_of_df)   


    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    for txp, group in df.groupby("txp"):
        mean_snr = group.groupby("distance")["snr"].mean()

        # discard value if txp == 0 and distance == 8: continue
        if txp == 0 and 8 in mean_snr.index:
            mean_snr.drop(8, inplace=True)

        print(mean_snr)
        mean_snr.plot(ax=ax, marker='o', label=f'TXP {txp} [dBm]')
        # fit propagation model
        r0 = 0.5
        L0 = max(mean_snr)
        n = -1
        # curve fit
        popt, _ = curve_fit(propagation_model, mean_snr.index, mean_snr, p0=[r0, L0, n])
        print(popt)
        # plot fitted curve
        #r = np.linspace(mean_snr.index.min(), mean_snr.index.max(), 100)
        r = np.linspace(0.5, 16, 100)
        ax2.plot(r, propagation_model(r, *popt), label=f'TXP {txp} [dBm]')
        


    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Mean SNR [dB]")
    ax.set_title("Mean SNR vs Distance for each TXP")

    """
    # change curve colors
    colors = ['r', 'r', 'g', 'g', 'b', 'b', 'orange', 'orange']
    for i, line in enumerate(ax.get_lines()):
        line.set_color(colors[i])
    """

    # change legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    ax2.set_xlabel("Distance [m]")
    ax2.set_ylabel("Mean SNR [dB]")
    ax2.set_title("Mean SNR vs Distance for each TXP (fit)")
    ax2.legend(handles[::-1], labels[::-1])

    plt.show()
