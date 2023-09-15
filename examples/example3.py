#!/usr/bin/env python

"""Datalock example to convert data

"""
__author__ = "Erik Bartoš"
__copyright__ = "Copyright © 2023 Erik Bartoš"
__email__ = "erik.bartos@gmail.com"

import os
import matplotlib.markers as mrk
import matplotlib.pyplot as plt
import matplotlib.container as mcn
import numpy as np

from src.experimentalData import ExperimentalData


# wayland setting
os.environ['XDG_SESSION_TYPE'] = 'x11'


def main():
    data_file = "dataKaonsNeutral.csv"
    exd = ExperimentalData()
    exd.read_data(data_file, dtypes=[61], sets=["Ablikim-21",
                                                 ],
                  xmin=.0, xmax=10.)
    # exd.show_data()

    # Convert data
    tx = exd.data_x
    ty = exd.data_y
    tdy = exd.data_dy_plus
    ALPHA = 0.007297352569816315
    HBARC2 = 0.38937937217186
    M_K0 = 0.001 * 497.611
    BETA = np.sqrt(1 - 4*M_K0**2/tx)
    cef = np.pi*ALPHA**2*BETA**3*HBARC2*1e6/(3*tx)
    tz = np.sqrt(ty/cef)
    # terr = 0.5*np.sqrt(1/(ty*cef))*tdy
    terr = 0.5 / cef / tz * tdy

    for i, e in enumerate(tx):
        # print(f"{i}, {e}, {ty[i]}, {tz[i]:.3f} +/- {terr[i]:.3f} ")
        # to_csv_file
        print(f"51,{e:.4f},{tz[i]:.4f},{terr[i]:.4f},{terr[i]:.4f}")

    """
    # Plot data
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle('Pion form factor')

    ax1 = plt.subplot(111)
    ax1.set_title("")
    ax1.set_xlabel("$s$  [(GeV/c)$^2$]")
    # ax1.set_ylabel(r"$\sigma_{tot}(e^+e^-\to\pi^+\pi^-)$")
    ax1.set_ylabel(r"$|F_{\pi}|^2$")
    axes = plt.gca()
    axes.set_xlim([0, 2.1])
    axes.set_ylim([1e-1, 1e2])

    # color = ["blue", "red", "green", "gold", "coral", "orange", "brown"]
    valid_markers = mrk.MarkerStyle.filled_markers + ("p", "o")
    for i, t in enumerate(exd.legend):
        dx = []
        dy = []
        dy_err = []
        for j, w in enumerate(exd.abbrev):
            if w == t:
                dx.append(exd.data_x[j])
                dy.append(exd.data_y[j])
                dy_err.append(exd.data_dy_plus[j])
        ax1.errorbar(dx, dy, dy_err, marker=valid_markers[i],
                     linestyle='', markersize=2, label=t, capsize=2)  # , mfc=color[i], mec=color[i])
    ax1.set_yscale("log")

    # remove errorbars in legend
    handles, labels = ax1.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, mcn.ErrorbarContainer) else h for h in handles]
    plt.legend(handles, labels, loc=1, prop={'size': 10}, shadow=True, facecolor="white", framealpha=1,
               title=r'$\bf{Legend}$', markerscale=3.)  # edgecolor="grey"

    fig.tight_layout()
    # fig.savefig("plot_pion_eff.png", format="png", dpi=150)
    """
    plt.show()


if __name__ == "__main__":
    main()
