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


def Li2(x):
    return special.spence(1-x)


def etaFSR(x):
    delta = ((1. + x*x)/x *
             (4.*Li2((1. - x)/(1. + x)) + 2.*Li2((x - 1.)/(1. + x))
             - 3.*np.log(2./(1. + x))*np.log((1. + x)/(1. - x)) - 2.*np.log(x)*np.log((1. + x)/(1. - x)))
             - 3.*np.log(4./(1. - x*x)) - 4.*np.log(x)
             + 1./x/x/x*(5./4.*(1. + x*x)*(1. + x*x) - 2.)*np.log((1. + x)/(1. - x))
             + 3./2.*(1. + x*x)/x/x
             )
    return delta


def alphas(x):
    return alf/(1. - alf/3./np.pi*np.log(x/massPi/massPi))


# Filenames
dir_name = './data'
dev_name = './dev'
if not os.path.exists(dev_name):
    os.makedirs(dev_name)

filename = 'dataBaBar'
filename_suffix = 'csv'
out_filename  = os.path.join(dev_name, 'outFile.csv')

# Read data to dataframe
data_file = os.path.join(dir_name, filename + '.' + filename_suffix)
print(data_file)

df = pd.read_csv(data_file, sep=',', header=0,
                 names=['E1', 'E2', 'sqrtS', 'S', 'bareCS', 'deltaCS', 'AR', 'ARL', 'ARH'])
#df.reset_index()
dataX = df['sqrtS']

df['beta'] = beta(df['S'])
df['FSR'] = (1. + etaFSR(df['beta'])*alf/np.pi)
df['VP'] = np.power(alphas(df['S'])/alf, 2)
df['ufactor'] = 3. * df['S'] / np.pi / alf / alf / np.power(df['beta'], 3.) * nanoToMili / htc2 / df['FSR']
df['deltaAR'] = np.abs(df['ARH'] - df['ARL'])
df['dressedCS'] = df['bareCS'] * np.power(df['AR'], 2) / df['FSR']
df['d1v'] = np.power(df['AR'], 2) / df['FSR']
df['d2v'] = 2. * df['bareCS'] * df['AR'] / df['FSR']
df['deltaDressedCS'] = (np.sqrt(np.power(df['d1v'], 2) * np.power(df['deltaCS'], 2)
                        + np.power(df['d2v'], 2) * np.power(df['deltaAR'], 2)))

df['FF'] = df['ufactor'] * df['bareCS'] * np.power(df['AR'], 2)
df['deltaFF'] = df['ufactor'] * (np.power(df['AR'], 2) * df['deltaCS'] + 2.*df['bareCS'] * df['AR'] * df['deltaAR'])
df['d1vf'] = df['ufactor'] * np.power(df['AR'], 2)
df['d2vf'] = df['ufactor'] * 2.*df['bareCS'] * df['AR']
df['deltaFFn'] = (np.sqrt(np.power(df['d1vf'], 2) * np.power(df['deltaCS'], 2)
                  + np.power(df['d2vf'], 2) * np.power(df['deltaAR'], 2)))
df['percompCS'] = np.fabs(df['bareCS'] - df['dressedCS'])/df['bareCS']*100

print(df[['sqrtS', 'S', 'bareCS', 'FF', 'deltaFF', 'deltaFFn', 'percompCS']])
df.to_csv(out_filename)

fig = plt.figure(figsize=(10, 7))
fig.canvas.manager.set_window_title('BaBar')

ax1 = plt.subplot(111)
ax1.set_title('BaBar')
ax1.set_xlabel('$\\sqrt{s}$  [GeV]')
ax1.set_ylabel('$\\sigma$  [nb]')
ax1.errorbar(dataX, df['bareCS'], yerr=df['deltaCS'], marker='o', linestyle='', markersize=6,
             label="bareCS")
ax1.set_yscale('log')
ax2 = plt.subplot(111)
ax2.errorbar(dataX, df['dressedCS'], yerr=df['deltaDressedCS'], marker='.', linestyle='', markersize=6,
             color='r', label="dressedCS")

# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
# yfmt = ScalarFormatter(useOffset=False)
# yfmt.set_powerlimits((-4,4))
# ax1.yaxis.set_major_formatter(yfmt)
ax1.yaxis.grid()
plt.legend(loc=1, prop={'size': 12})
fig.tight_layout()
save_file = os.path.join(dev_name, 'plot_BaBar_CS.png')
fig.savefig(save_file, dpi=150)

fig = plt.figure(figsize=(10, 7))
fig.canvas.manager.set_window_title('BaBar')

ax1 = plt.subplot(111)
ax1.set_title('Plot title')
ax1.set_xlabel('$\\sqrt{s}$  [GeV]')
ax1.set_ylabel('$|F(s)|^2$')
ax1.errorbar(dataX, df['FF'], yerr=df['deltaFF'], marker='o', linestyle='', markersize=6,
             label="FF")
ax1.set_yscale('log')
ax1.yaxis.grid()

plt.legend()
fig.tight_layout()
save_file = os.path.join(dev_name, 'plot_BaBar_FF.png')
fig.savefig(save_file, dpi=150)

fig = plt.figure(figsize=(10, 7))
ax3 = plt.subplot(111)
ax3.set_title('Eta')
ax3.set_xlabel('$\\sqrt{s}$  [GeV]')
ax3.set_ylabel('$\\eta(s)$')
ax3.plot(dataX, etaFSR(df['beta']), marker='o', linestyle='', markersize=3)
fig.tight_layout()
save_file = os.path.join(dev_name, 'plot_eta.png')
fig.savefig(save_file, dpi=150)

fig = plt.figure(figsize=(10, 7))
ax4 = plt.subplot(111)
ax4.set_title('Alpha')
ax4.set_xlabel('$\\sqrt{s}$  [GeV]')
ax4.set_ylabel('$\\alpha(s)$')
# ax4.plot(dataX, alphas(df['beta'])/alf, marker='o', linestyle='', markersize=3)
ax4.plot(dataX, df['AR'], marker='o', linestyle='', markersize=3)
fig.tight_layout()
save_file = os.path.join(dev_name, 'plot_alpha.png')
fig.savefig(save_file, dpi=150)

plt.show()
