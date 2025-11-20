#!/usr/bin/env python

"""Datalock example to read and plot data

"""
__author__ = "Erik Bartoš"
__copyright__ = "Copyright © 2022 Erik Bartoš"
__email__ = "erik.bartos@gmail.com"

import os
import matplotlib.pyplot as plt
from src.datalock.experimentalData import ExperimentalData

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

    # build-in functions
    exd.show_data()
    print(exd.inner_check())
    plt.show()


if __name__ == "__main__":
    main()
