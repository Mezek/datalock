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

filename = 'dataCMD-2'
filename_suffix = 'csv.gz'
out_filename = os.path.join(dev_name, 'outFile.csv')

# Read data to dataframe
data_file = os.path.join(dir_name, filename + '.' + filename_suffix)
print(data_file)

df = pd.read_csv(data_file, sep=',', header=0, names=['sqrtS','CS','deltaCS','pionFF','deltaPionFF'])
# df.reset_index()
df['S'] = np.power(df['sqrtS']/1000., 2.)
dataX = df['sqrtS']/1000.
dataY = df['CS']
dataEY = df['deltaCS']

df['beta'] = beta(df['S'])
df['check'] = 1. / (3. * df['S'] /
            np.pi / alf / alf / np.power(df['beta'], 3.) * nanoToMili / htc2) * df['pionFF']
df['deltaCheck'] = 1. / (3. * df['S'] /
            np.pi / alf / alf / np.power(df['beta'], 3.) * nanoToMili / htc2) * df['deltaPionFF']
print(df[['sqrtS','S','CS','check','deltaCheck']])
# df.to_csv(out_filename)

fig = plt.figure(figsize=(10, 7))
fig.canvas.manager.set_window_title('Figure name')

ax1 = plt.subplot(111)
ax1.set_title('CMD-2')
ax1.set_xlabel('$\\sqrt{s}$  [GeV]')
ax1.set_ylabel('$\\sigma$  [nb]')
ax1.errorbar(dataX, dataY, yerr=dataEY, marker='o', linestyle='', markersize=6)

# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
yfmt = ScalarFormatter(useOffset=False)
yfmt.set_powerlimits((-4, 4))
ax1.yaxis.set_major_formatter(yfmt)
ax1.yaxis.grid()

# fig = plt.figure(figsize=(10, 7))
# ax2 = plt.subplot(111)
# ax2.errorbar(dataX, df['FF'], yerr=df['deltaFF'], marker='o', linestyle='', markersize=6)

fig.tight_layout()
save_file = os.path.join(dev_name, 'plot_CMD-2_CS.png')
fig.savefig(save_file, dpi=150)

plt.show()
