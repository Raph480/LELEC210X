import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def csv_to_arrays(fichier,sep=','):
    datas=pd.read_csv(fichier,sep=sep,header=None)
    datas=datas.values[1:]
    datas=datas.astype(float)
    return datas.T

plt.figure(figsize=(9,5), dpi=100)

#filename = "power subtasks v1"
#filename = "classic_usage_buffer10k"
filename = "classic_usage_buffer5k"
with open(filename+".csv", 'r') as f:
    first_char = f.read(1)
    if (first_char == ';'):
        lines = f.readlines()
        with open(filename+".csv", 'w') as f:
            f.writelines(lines[7:])
datas=csv_to_arrays(filename+".csv")
datas[1] = datas[1] + 8 # shift by 8 seconds to begin at 0
R = 50 # shunt resistance
I = datas[2]/R # current
P_mcu = 3.3*I # power of the mcu
plt.plot(datas[1],P_mcu*1000,label="MCU")

plt.title('Instant power consumption of the MCU')
plt.xlabel('Time [s]')
plt.xticks
plt.ylabel('Power [mW]')
#plt.xlim(xmin=0,xmax=7.15)
plt.ylim(ymin=0, ymax=20)
#plt.legend(loc="upper right")
plt.savefig(filename+".svg")
#plt.show()




# plot a piechart of the energy consumption with data[2] and data[3]
plt.figure(figsize=(9,5), dpi=100)
P_mcu = 3.3*I # power of the mcu
#E_mcu = np.trapz(P_mcu,datas[1]) # energy of the mcu
#print(E_mcu)

#### PIE 1 ####
## Pie of the energy consumption of the MCU during the three subtasks

# masks for different tasks
mask_1 = P_mcu < 8e-3
mask_2 = (P_mcu >= 8e-3) & (P_mcu < 12e-3)
mask_3 = P_mcu >= 12e-3

# energies for different tasks
E_1 = np.trapz(np.where(P_mcu < 8e-3, P_mcu, 0),datas[1])
E_2 = np.trapz(np.where((P_mcu >= 8e-3) & (P_mcu < 12e-3),P_mcu,0),datas[1])
E_3 = np.trapz(np.where(P_mcu >= 12e-3, P_mcu, 0),datas[1])

E_tot = np.trapz(P_mcu,datas[1])


labels = '(1) WFI state', '(2) Audio sampling', '(3) UART transmission'
values = [E_1, E_2, E_3]
colors = ['gold', 'yellowgreen', 'lightcoral']
wedgeprops = {"edgecolor": "white", 'linewidth': 3, 'antialiased': True}
textprops = {'fontsize': 12, 'color': 'white'}

# Format labels
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = pct/100*total
        return f'{val:.3f} [J] \n({pct:.1f}%)'
    return my_format

# Tracer le graphique en secteurs
plt.pie(values, labels=labels, colors=colors, autopct=autopct_format(values), shadow=False, startangle=80, wedgeprops=wedgeprops, textprops=textprops)
plt.axis('equal')
plt.title('Energy consumption of the MCU during the three subtasks')

# change label text color to black, as it is invisible if kept white :)
for text in plt.gca().texts:
    if text.get_text() in labels:
        text.set_color('black')

plt.savefig(filename+"_piechart_1.svg")


# plot mask_1, mask_2, mask_3 spans
plt.figure(figsize=(9,5), dpi=100)
plt.plot(datas[1],P_mcu*1000,label="MCU")
plt.fill_between(datas[1],P_mcu*1000, where=mask_1, color='gold', alpha=0.5, label='WFI state')
plt.fill_between(datas[1],P_mcu*1000, where=mask_2, color='green', alpha=0.5, label='Audio sampling')
plt.fill_between(datas[1],P_mcu*1000, where=mask_3, color='red', alpha=0.5, label='UART transmission')
#plt.title('Energy consumption of the MCU during the three subtasks \n(Piechart 1 equivalent)')
plt.xlabel('Time [s]')
plt.ylabel('Power [mW]')
plt.legend(loc="upper right")
plt.savefig(filename+"_mask1.svg")



#### PIE 2 ####
## Pie of the energy consumption of the MCU during subtasks (2) and (3)
plt.figure(figsize=(9,5), dpi=100)
labels = '(2) Audio sampling', '(3) UART transmission'
values = [E_2, E_3]
colors = ['yellowgreen', 'lightcoral']
wedgeprops = {"edgecolor": "white", 'linewidth': 3, 'antialiased': True}
textprops = {'fontsize': 12, 'color': 'white'}

# Format labels
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = pct/100*total
        return f'{val:.3f} [J] \n({pct:.1f}%)'
    return my_format

# Tracer le graphique en secteurs
plt.pie(values, labels=labels, colors=colors, autopct=autopct_format(values), shadow=False, startangle=0, wedgeprops=wedgeprops, textprops=textprops)
plt.axis('equal')
#plt.title('Energy consumption of the MCU during subtasks (2) and (3)')

# change label text color to black, as it is invisible if kept white :)
for text in plt.gca().texts:
    if text.get_text() in labels:
        text.set_color('black')

plt.savefig(filename+"_piechart_2.svg")


# plot mask_2, mask_3 spans
plt.figure(figsize=(9,5), dpi=100)
plt.plot(datas[1],P_mcu*1000,label="MCU")
plt.fill_between(datas[1],P_mcu*1000, where=mask_2, color='green', alpha=0.5, label='Audio sampling')
plt.fill_between(datas[1],P_mcu*1000, where=mask_3, color='red', alpha=0.5, label='UART transmission')
plt.title('Energy consumption of the MCU during subtasks (2) and (3) \n(Piechart 2 equivalent)')
plt.xlabel('Time [s]')
plt.ylabel('Power [mW]')
plt.legend(loc="upper right")
plt.savefig(filename+"_mask2.svg")


#### PIE 3 ####
## Pie of the energy consumption of the MCU during subtasks (2) and (3), WFI state energy deducted

# mean P_1 power during WFI state
P_1 = np.mean(P_mcu[mask_1])
#deduce energy of P_1 during mask_2 and mask_3
E_2_rel = E_2 - P_1* np.trapz(np.where((P_mcu >= 8e-3) & (P_mcu < 12e-3),1,0),datas[1])
E_3_rel = E_3 - P_1* np.trapz(np.where(P_mcu < 12e-3,1,0),datas[1])

plt.figure(figsize=(9,5), dpi=100)
labels = '(2) Audio sampling', '(3) UART transmission'
values = [E_2_rel, E_3_rel]
colors = ['yellowgreen', 'lightcoral']
wedgeprops = {"edgecolor": "white", 'linewidth': 3, 'antialiased': True}
textprops = {'fontsize': 12, 'color': 'white'}

# Format labels
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = pct/100*total
        return f'{val:.3f} [J] \n({pct:.1f}%)'
    return my_format

# Tracer le graphique en secteurs
plt.pie(values, labels=labels, colors=colors, autopct=autopct_format(values), shadow=False, startangle=0, wedgeprops=wedgeprops, textprops=textprops)
plt.axis('equal')
#plt.title('Energy consumption of the MCU during subtasks (2) and (3), WFI state energy deducted')

# change label text color to black, as it is invisible if kept white :)
for text in plt.gca().texts:
    if text.get_text() in labels:
        text.set_color('black')

plt.savefig(filename+"_piechart_3.svg")


# plot mask_2, mask_3 spans
plt.figure(figsize=(9,5), dpi=100)
plt.plot(datas[1],P_mcu*1000,label="MCU")
plt.fill_between(datas[1],P_mcu*1000, where=mask_2, color='green', alpha=0.5, label='Audio sampling')
plt.fill_between(datas[1],P_mcu*1000, where=mask_3, color='red', alpha=0.5, label='UART transmission')
plt.fill_between(datas[1],P_1*1000, color='white', alpha=1)
plt.title('Energy consumption of the MCU during subtasks (2) and (3), WFI state energy deducted \n(Piechart 3 equivalent)')
plt.xlabel('Time [s]')
plt.ylabel('Power [mW]')
plt.legend(loc="upper right")
plt.savefig(filename+"_mask3.svg")


#### PIE 4 ####
## Pie of the relative energy consumption of the MCU during sequence
plt.figure(figsize=(9,5), dpi=100)
labels = 'Idle (WFI state only)','Audio sampling only', 'UART transmission only'
E_1_rel = P_1 * (datas[1][-1] - datas[1][0])
values = [E_1,E_2-E_1, E_3-E_1]
colors = ['gold','yellowgreen', 'lightcoral']
wedgeprops = {"edgecolor": "white", 'linewidth': 3, 'antialiased': True}
textprops = {'fontsize': 12, 'color': 'white'}

# Format labels
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = pct/100*total
        return f'{val:.3f} [J] \n({pct:.1f}%)'
    return my_format

# Tracer le graphique en secteurs
plt.pie(values, labels=labels, colors=colors, autopct=autopct_format(values), shadow=False, startangle=0, wedgeprops=wedgeprops, textprops=textprops)
plt.axis('equal')
plt.title('Relative energy consumption of the MCU during sequence')

# change label text color to black, as it is invisible if kept white :)
for text in plt.gca().texts:
    if text.get_text() in labels:
        text.set_color('black')

plt.savefig(filename+"_piechart_4.svg")


# plot mask_2, mask_3 spans
plt.figure(figsize=(9,5), dpi=100)
plt.plot(datas[1],P_mcu*1000,label="MCU")
plt.fill_between(datas[1],P_mcu*1000, where=mask_2, color='green', alpha=0.5, label='Audio sampling only')
plt.fill_between(datas[1],P_mcu*1000, where=mask_3, color='red', alpha=0.5, label='UART transmission only')
plt.fill_between(datas[1],P_1*1000, color='white', alpha=1)
plt.fill_between(datas[1],P_1*1000, color='gold', alpha=0.5, label='Idle (WFI state only)')
#plt.title('Relative energy consumption of the MCU during sequence \n(Piechart 4 equivalent)')
plt.xlabel('Time [s]')
plt.ylabel('Power [mW]')
plt.legend(loc="upper right")
plt.savefig(filename+"_mask4.svg")


plt.show()
