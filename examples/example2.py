#!/usr/bin/env python

"""Datalock example to read and plot data

"""
__author__ = "Erik Bartoš"
__copyright__ = "Copyright © 2022 Erik Bartoš"
__email__ = "erik.bartos@gmail.com"

import os
import matplotlib.markers as mrk
import matplotlib.pyplot as plt
import matplotlib.container as mcn

from src.experimentalData import ExperimentalData


# wayland setting
os.environ['XDG_SESSION_TYPE'] = 'x11'


def main():
    data_file = "dataPions.csv"
    exd = ExperimentalData()
    exd.read_data(data_file, dtypes=[101], sets=["Akhmetshin-02",  # CMD-2,
                                                 "Achasov-06",  # SND
                                                 "Ambrosino-11", "Babusci-13",  # KLOE
                                                 "Ablikim-21",  # BESIII
                                                 "Aubert-09",  # BABAR
                                                 "Xiao-18",  # CLEO
                                                 ],
                  xmin=.0, xmax=10.)

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
    plt.show()


if __name__ == "__main__":
    main()
