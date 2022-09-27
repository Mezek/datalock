# ----------------------------------------------------------------------------
# datalock
# Copyright © 2021 Erik Bartoš <erik.bartos@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

"""
Data for elementary particles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Any, Callable, List, Optional, Tuple, AnyStr


class ExperimentalData:
    """
    Class to read and prepare experimental data.
    """

    __slots__ = (
        "_data_files",
        "_data_types",
        "_data_sets",
        "_df"
    )

    @property
    def df(self) -> pd.DataFrame:
        """
        Get list of current data as pandas dataframe.
        """
        return self._df

    @property
    def files(self) -> List[str]:
        """
        Get list of read data files.
        """
        return self._data_files

    @property
    def dtypes(self) -> List[float]:
        """
        Get list of counted data types.
        """
        return self._data_types

    @property
    def sets(self) -> List[str]:
        """
        Get list of counted data sets.
        """
        return self._data_sets

    @property
    def data_x(self) -> List[float]:
        """
        Get list of data x values.
        """
        return self._df["x"].to_numpy()

    @property
    def data_y(self) -> List[float]:
        """
        Get list of data y values.
        """
        return self._df["y"].to_numpy()

    @property
    def data_dx_plus(self) -> List[float]:
        """
        Get list of data x errors.
        """
        return self._df["dx+"].to_numpy()

    @property
    def data_dx_minus(self) -> List[float]:
        """
        Get list of data x errors.
        """
        return self._df["dx-"].to_numpy()

    @property
    def data_dy_plus(self) -> List[float]:
        """
        Get list of data y errors.
        """
        return self._df["dy+"].to_numpy()

    @property
    def data_dy_minus(self) -> List[float]:
        """
        Get list of data y errors.
        """
        return self._df["dy-"].to_numpy()

    @property
    def abbrev(self) -> List[str]:
        """
        Get data abbreviations.
        """
        return self._df["Abbrev"]

    @property
    def legend(self) -> List[str]:
        """
        Get list of data sources.
        """
        return self._df["Abbrev"].unique()

    @property
    def count(self) -> int:
        """
        Count number of current data points.
        """
        return self._df.shape[0]

    def __init__(self):
        """
        Initialize object for experimental data.
        """
        # self._df = pd.DataFrame(columns=["Code", "Abbrev", "x", "y", "dy+", "dy-"])
        self._df = pd.DataFrame()
        self._data_files = []
        self._data_types = []
        self._data_sets = []

    def reset(self):
        self._df = pd.DataFrame(None)
        self._data_files = []
        self._data_types = []
        self._data_sets = []

    def read_data(self, filename: str, dtypes: Optional[List[int]] = None, sets: Optional[List[str]] = None,
                  xmin: Optional[float] = None, xmax: Optional[float] = None):
        """
        Read one data file with user defined data types.

        :param filename: Input data file.
        :param dtypes: Data type which are counted. If None (default), all types are used.
        :param sets: List of data abbreviations. If None (default), all sets are used.
        :param xmin: Minimal x value
        :param xmax: Maximal x value
        :return: nothing
        """
        df = pd.read_csv(filename, sep=",", header=0, comment="#", skipinitialspace=True)
        if dtypes is None:
            dtypes = list(df.Code.unique())
        self._data_files.append(filename)
        self._data_types.append(dtypes)
        df = df.loc[df["Code"].isin(dtypes)]
        if sets is None:
            sets = list(df.Abbrev.unique())
        self._data_sets.append(sets)
        df = df.loc[df["Abbrev"].isin(sets)]
        if xmin:
            df = df.loc[(df["x"] >= xmin)]
        if xmax:
            df = df.loc[(df["x"] <= xmax)]
        # self._df = self._df.append(df)  # depreciated in new pandas
        self._df = pd.concat([self._df, df], axis=0, join="outer")
        # check NaN values
        if df.isnull().values.any():
            raise RuntimeError("Missing data value(s) in data file")

    def show_data(self):
        fig = plt.figure(figsize=(7, 5), )
        fig.canvas.manager.set_window_title("Experimental data with y errors")
        ax = plt.subplot(111)
        ax.set_title("All data")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.errorbar(self.data_x, self.data_y, yerr=[self.data_dy_plus, self.data_dy_minus], fmt="o", linestyle="",
                    markersize=5, color="royalblue", capsize=3)

    def show_data_xy_err(self):
        fig = plt.figure(figsize=(7, 5), )
        fig.canvas.manager.set_window_title("Experimental data with xy errors")
        ax = plt.subplot(111)
        ax.set_title("All data")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.errorbar(self.data_x, self.data_y, yerr=[self.data_dy_plus, self.data_dy_minus],
                    xerr=[self.data_dx_plus, self.data_dx_minus], fmt="o", linestyle="",
                    markersize=5, color="royalblue", capsize=3)

    def inner_check(self):
        # Duplicated values
        duplicates = self._df[self._df.duplicated()]
        log_str = f"{'='*23}\nExperimental data check\n"
        log_str += f"Data count:        {self.count}\n"
        log_str += f"Data sets:         {self._data_sets}\n"
        log_str += f"Duplicated rows:   {len(duplicates)}\n"
        # Asymmetric y errors
        comparison_column = np.where(self._df["dy-"] == self._df["dy+"], 0, 1)
        nasm = np.sum(comparison_column)
        log_str += f"Asymmetric errors: {nasm}  ({nasm/self.count*100:.2f}%)\n{'='*23}"
        return log_str
