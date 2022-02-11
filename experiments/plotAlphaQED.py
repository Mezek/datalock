import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import os
import sys
import glob
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from scipy import special

massPi = 0.13957018
alf = 1./137.035999139

htc2 = 0.389379338
nanoToMili = 0.000001


def beta(x):
    return np.sqrt(1. - 4.*massPi*massPi/x)


def alphas(x):
    return alf/(1. - alf/3./np.pi*np.log(x/massPi/massPi))


# Filenames
dir_name = './data'
dev_name = './dev'
if not os.path.exists(dev_name):
    os.makedirs(dev_name)

filename = 'dataAlphaQED'
filename_suffix = 'csv'
out_filename = os.path.join(dev_name, 'outFile.csv')

# Read data to dataframe
data_file = os.path.join(dir_name, filename + '.' + filename_suffix)
print(data_file)

df = pd.read_csv(data_file, sep=',', header=0, names=['sqrtS', 'A', 'AL', 'AH'])
# df.reset_index()
dataX = df['sqrtS']
df['S'] = df['sqrtS']*df['sqrtS']
df['beta'] = beta(df['S'])
print(df[['sqrtS','S']])
# df.to_csv(out_filename)

fig = plt.figure(figsize=(10, 7))
fig.canvas.manager.set_window_title('Figure name')

dataCX = []
dataCY = []
for i in range(0, 1000):
    nx = 5. + 9./1000*i
    dataCX.append(nx)
    dataCY.append(alphas(nx*nx))

ax1 = plt.subplot(111)
ax1.set_title('Plot title')
ax1.set_xlabel('$\\sqrt{s}$  [GeV]')
ax1.set_ylabel('$\\alpha$')
ax1.plot(dataCX, dataCY, marker='o', linestyle='', markersize=3)
# ax1.set_yscale('log')

ax2 = plt.subplot(111)
# ax2.plot(df['sqrtS'], alf/(1. - df['A']), marker='.', linestyle='', markersize=3, color='red')
ax2.plot(df['sqrtS'], df['A'], marker='.', linestyle='', markersize=3, color='red')
# ax2.set_yscale('log')

# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
# yfmt = ScalarFormatter(useOffset=False)
# yfmt.set_powerlimits((-4,4))
# ax1.yaxis.set_major_formatter(yfmt)
ax1.yaxis.grid()

fig.tight_layout()
save_file = os.path.join(dev_name, 'plot_alphaQED.png')
fig.savefig(save_file, dpi=150)

plt.show()
