import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import os
import sys
import glob
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

massPi = 0.13957018
alf = 1./137.035999139

htc2 = 0.389379338
nanoToMili = 0.000001


def beta(x):
    return np.sqrt(1. - 4.*massPi*massPi/x)


# Filenames
dir_name = './data'
dev_name = './dev'
if not os.path.exists(dev_name):
    os.makedirs(dev_name)

filename = 'dataCSbare'
filename_suffix = 'csv'
out_filename = os.path.join(dev_name, 'outFile.csv')

# Read data to dataframe
data_file = os.path.join(dir_name, filename + '.' + filename_suffix)
print(data_file)

df = pd.read_csv(data_file, sep=',', header=0, names=['S','CS','deltaCS'])
#df.reset_index()
# BaBar: 337
# KLOE: 75
dataX1 = np.sqrt(df.loc[0:336, 'S'])
dataY1 = df.loc[0:336, 'CS']
dataEY1 = df.loc[0:336, 'deltaCS']
#print df.loc[0:336, ['S', 'CS', 'deltaCS']]

dataX2 = np.sqrt(df.loc[337:411, 'S'])
dataY2 = df.loc[337:411, 'CS']
dataEY2 = df.loc[337:411, 'deltaCS']
print(df.loc[337:411, ['S', 'CS', 'deltaCS']])

"""
df['beta'] = beta(df['S'])
df['FSR'] = np.pi * alf / 2. / df['beta']
df['FF'] = (3. * df['S'] * df['CS'] /
            np.pi / alf / alf / np.power(df['beta'], 3.) * nanoToMili / htc2 *
            (1. + df['FSR']))
df['deltaFF'] = (3. * df['S'] * df['deltaCS'] /
            np.pi / alf / alf / np.power(df['beta'], 3.) * nanoToMili / htc2)
print df[['S','CS','FF','deltaFF','beta','FSR']]
# df.to_csv(out_filename)
"""

fig = plt.figure(figsize=(10, 7))
fig.canvas.manager.set_window_title('Figure name')

ax1 = plt.subplot(111)
ax1.set_title('BaBar + KLOE')
ax1.set_xlabel('$\\sqrt{s}$  [GeV]')
ax1.set_ylabel('$\\sigma^{bare}_{\\pi\\pi}$  [nb]')
ax1.errorbar(dataX1, dataY1, yerr=dataEY1, marker='o', linestyle='', markersize=2, color='red', capsize=2, label='BaBar09')
ax1.set_xlim([0.3, 1.2])

# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
yfmt = ScalarFormatter(useOffset=False)
yfmt.set_powerlimits((-4,4))
ax1.yaxis.set_major_formatter(yfmt)
ax1.yaxis.grid()

ax2 = plt.subplot(111)
ax2.errorbar(dataX2, dataY2, yerr=dataEY2, marker='o', linestyle='', markersize=2, color='blue', capsize=2, label='KLOE10')

plt.legend()
fig.tight_layout()
save_file = os.path.join(dev_name, 'plot_CS_bare.png')
fig.savefig(save_file, dpi=150)


plt.show()
