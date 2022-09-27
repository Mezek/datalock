#!/usr/bin/env python

"""Datalock example to read and plot data

"""
__author__ = "Erik Bartoš"
__copyright__ = "Copyright © 2022 Erik Bartoš"
__email__ = "erik.bartos@gmail.com"

import os
import numpy as np
import logging.config
import matplotlib.markers as mrk
import matplotlib.pyplot as plt
import matplotlib.container as mcn
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import scipy.optimize

from iminuit import Minuit
from iminuit.cost import LeastSquares
from src.bvpasion import FFactor, ExperimentalData
from src.bvpasion.models import constants as cms
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# wayland setting
os.environ['XDG_SESSION_TYPE'] = 'x11'


def main():
    data_file = "./dataPions.csv"
    exd = ExperimentalData()
    exd.read_data(data_file, dtypes=[101], sets=["Akhmetshin-02",  # CMD-2,
                                                 "Achasov-06",  # SND
                                                 "Ambrosino-11", "Babusci-13",  # KLOE
                                                 "Ablikim-21",  # BESIII
                                                 "Aubert-09",  # BABAR
                                                 "Xiao-18",  # CLEO
                                                ],
                  xmin=.0, xmax=10.)
    exd.show_data()

    """
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

    # model line
    n_samples = 10000
    t_min = 0.08
    t_max = 9.90
    tx = np.linspace(t_min, t_max, n_samples)
    ty = mff.value_squared(tx)
    ax1.plot(tx, ty, color='black', linewidth=2, label='U&A model')

    # remove errorbars in legend
    handles, labels = ax1.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, mcn.ErrorbarContainer) else h for h in handles]
    plt.legend(handles, labels, loc=1, prop={'size': 10}, shadow=True, facecolor="white", framealpha=1,
               title=r'$\bf{Legend}$', markerscale=3.)  # edgecolor="grey"

    fig.tight_layout()
    save_file = os.path.join(dev_name, "plot_pion_eff.png")
    fig.savefig(save_file, format="png", dpi=150)
    """

    """
    logger.info(f"Plotting other figure...")
    fig2 = plt.figure(figsize=(10, 7))
    fig2.suptitle(r"Pion difference: $F_{th}^{} - F_{exp}^{}$")

    ax1 = plt.subplot(111)
    ax1.set_title("")
    ax1.set_xlabel("$s$  [(GeV/c)$^2$]")
    ax1.set_ylabel(r"$\Delta$")
    axes = plt.gca()
    axes.set_xlim([0., 2.05])
    # axes.set_ylim([1e-1, 1e2])

    for i, t in enumerate(exd.legend):
        dx = []
        dy = []
        dy_err = []
        dy_mod = []
        for j, w in enumerate(exd.abbrev):
            if w == t:
                dx.append(exd.data_x[j])
                dy.append(exd.data_y[j])
                dmth = mff.value_squared(exd.data_x[j])
                dmex = exd.data_y[j]
                # dm = dmth - dmex
                dm = np.sqrt(dmth) - np.sqrt(dmex)
                dy_mod.append(dm)
                # dy_err.append(exd.data_dy_plus[j])
                dy_err.append(exd.data_dy_plus[j]/2/np.sqrt(dmex))
                # print(j, w, exd.data_x[j], dmex, dmth, dm)
        # ax1.errorbar(dx, dy_mod, dy_err, marker='o', markersize=4, linestyle='', label=t)
        ax1.plot(dx, dy_mod, marker='o', markersize=4, linestyle='', label=t)
        print("Data set...", t)

        linear_regressor = LinearRegression()
        lrx = np.array(dx).reshape(-1, 1)
        lry = np.array(dy_mod).reshape(-1, 1)
        linear_regressor.fit(lrx, lry)
        dy_pred = linear_regressor.predict(lrx)
        lintercept = linear_regressor.intercept_
        lcoef = linear_regressor.coef_
        lscore = linear_regressor.score(lrx, lry)
        print(f"Intercept={lintercept}, Coef.={lcoef}, Score={lscore}")
        ax1.plot(dx, dy_pred, color='black', linestyle='-', label=f"score={lscore:.3f}", linewidth=2)
    """
        """
        npoly = 7
        poly = PolynomialFeatures(npoly)
        x_poly = np.array(dx).reshape(-1, 1)
        lry = np.array(dy_mod).reshape(-1, 1)
        poly_features = poly.fit_transform(x_poly)
        poly_regressor = LinearRegression()
        poly_regressor.fit(poly_features, lry)
        poly_pred = poly_regressor.predict(poly_features)
        print(f"Polynomial = {npoly}:")
        # print("Coefficients: \n", poly_regressor.coef_)
        rmse = np.sqrt(mean_squared_error(lry, poly_pred))
        r2 = r2_score(lry, poly_pred)
        print(f"rmse = {rmse:.3}")
        print(f"R2   = {r2:.3}")

        ax1.plot(dx, poly_pred, color='black', linestyle='-', label=rf'poly-{npoly}: $R^2$={r2:.3}', linewidth=2)

        lrx = np.array(dx)
        res = fit_fcn(lrx, np.array(dy_mod))
        print("Cosine fit parameters:")
        print(f"Amplitude={res['amp']:.4}, ExpAmp={res['examp']:.4}, Angular freq.={res['omega']:.4}")
        print(f"Phase={res['phase']:.4}, Offset={res['offset']:.4}, Max. Cov.={res['maxcov']:.4}")

        ax1.plot(lrx, res["fitfunc"](lrx), "r-", label=rf"$\cos$: MaxCov={res['maxcov']:.4}", linewidth=2)
        """
    plt.show()


if __name__ == "__main__":
    main()
