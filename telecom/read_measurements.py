"""
Read a measurements file generated by eval_limesdr_fpga.py
and plots the PER/SNR curve, plus CFO values.
"""

import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    expected_payload = np.arange(100, dtype=np.uint8)
    num_bits = len(expected_payload) * 8

    data = defaultdict(list)
    with open(sys.argv[1]) as f:
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

        print(df)

    #fig = df.groupby("txp").hist(column="cfo")
    #plt.figure()
    fig, ax = plt.subplots()
    df.groupby("txp")["ber"].mean().plot(ax=ax, marker='o')
    ax.set_xlabel("Transmission Power (txp)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("BER vs TXP")
    ax.set_yscale('log')
    
    fig2, ax2 = plt.subplots()
    df.groupby("snr")["ber"].mean().plot(ax=ax2, marker='o')
    ax2.set_xlabel("Signal to noise ratio (SNR)")
    ax2.set_ylabel("Bit Error Rate (BER)")
    ax2.set_title("BER vs SNR")
    # y scale log
    ax2.set_yscale('log')


    # fig 3: incorrect vs SNR
    fig3, ax3 = plt.subplots()
    df.groupby("snr")["invalid"].mean().plot(ax=ax3, marker='o')
    ax3.set_xlabel("Signal to noise ratio (SNR)")
    ax3.set_ylabel("Incorrect packets")
    ax3.set_title("Incorrect packets vs SNR")
    # y scale log
    ax3.set_yscale('log')


    # fig 4: barplot of BER vs SNR with intervals of 5
    fig4, ax4 = plt.subplots()
    snr_bins = np.arange(df["snr"].min(), df["snr"].max() + 1, 5)
    df["snr_bin"] = pd.cut(df["snr"], bins=snr_bins, right=False)
    mean_ber = df.groupby("snr_bin")["ber"].mean()
    mean_ber.plot(ax=ax4, marker='o')
    ax4.set_xlabel("Signal to noise ratio (SNR)")
    ax4.set_ylabel("Bit Error Rate (BER)")
    ax4.set_title("BER vs SNR (binned)")
    ax4.set_yscale('log')
    # set mean xticks values
    ax4.set_xticks(np.arange(len(snr_bins))+(0.44-2)/5)
    ax4.set_xticklabels((snr_bins+0.44-2).round().astype(int))


    # fig 5: incorrect vs SNR with intervals of 5
    fig5, ax5 = plt.subplots()
    mean_invalid = df.groupby("snr_bin")["invalid"].mean()
    mean_invalid.plot(ax=ax5, marker='o', label='Measured')
    ax5.set_xlabel("Signal to noise ratio (SNR)")
    ax5.set_ylabel("Incorrect packets")
    ax5.set_title("Incorrect packets vs SNR (binned)")
    ax5.set_yscale('log')
    # set mean xticks values
    ax5.set_xticks(np.arange(len(snr_bins))+(0.44-2)/5)
    ax5.set_xticklabels((snr_bins+0.44-2).round().astype(int))

    
    data = np.loadtxt('simu_bypass.csv')
    SNRs_dB = data[:,0]
    PERs = data[:,3]
    shift_SNR_out = 9.106613348384755
    shift_SNR_filter = 1.535766260189646
    ax5.plot((SNRs_dB + 2*(shift_SNR_out+shift_SNR_filter)+(0.44-2))/5, PERs, marker='', label='Simulated with bypass')
    

    data = np.loadtxt('simu.csv')
    SNRs_dB = data[:,0]
    PERs = data[:,3]
    shift_SNR_out = 9.106613348384755
    shift_SNR_filter = 1.535766260189646
    ax5.plot((SNRs_dB  + 2*(shift_SNR_out+shift_SNR_filter)+(0.44-2))/5, PERs, marker='', label='Simulated without bypass')

    ax5.legend()
    

    # histogram: for SNR values between 17 and 20, plot histogram of CFO values
    fig6, ax6 = plt.subplots()
    df_snr_17_20 = df[(df["snr"] >= 17) & (df["snr"] <= 20)]
    df_snr_17_20.hist(column="cfo", ax=ax6, bins=20, rwidth=0.9, color='slateblue')
    # print mean
    print(f"Mean CFO for SNR between 17 and 20: {df_snr_17_20['cfo'].mean()}")
    ax6.set_xlabel("CFO [Hz]")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Histogram of CFO values for SNR between 17[dB] and 20[dB]")
    # no grid
    ax6.grid(False)

    plt.show()

