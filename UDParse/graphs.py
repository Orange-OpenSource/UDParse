#!/usr/bin/env python3

# Software Name: UDParse
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: Mozilla Public License 2.0
#
# This software is distributed under the MPL-2.0 license.
# the text of which is available at https://www.mozilla.org/en-US/MPL/2.0/
# or see the "LICENSE" file for more details.
#
# Author: Johannes HEINECKE <johannes(dot)heinecke(at)orange(dot)com> et al.


# make a nice graphs from ud_parser.py's log files


import matplotlib.pyplot as plt


class Graph:
    def __init__(self, logfile, keys, title=None):
        self.colors = "bgrcmyk"  # blue, green, red, cyan, magenta, yellow, black
        self.colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
        self.values = {}  # key: [vals]

        if not keys:
            keys = ["UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]

        for k in keys:
            self.values[k] = []

        keyset = set(keys)

        self.logfile = logfile
        ifp = open(self.logfile)
        for line in ifp:
            line = line.strip()
            if line.startswith("Dev"):
                elems = line.split(",")

                for x in elems:
                    x = x.strip()
                    ff = x.split(":")
                    if ff[0] in keyset:
                        self.values[ff[0]].append(float(ff[1].strip()))
        plt.xlabel("epochs")  # , fontweight="medium")
        plt.ylabel("F1")  # , fontweight="medium")
        if title:
            plt.title(title)  # , fontweight="medium")
        else:
            plt.title(logfile)  # , fontweight="medium")
        plt.grid()

    def plot(self, show=True):
        ct = 0
        iitems = sorted(self.values.items())
        for key, vals in iitems:
            if len(vals) > 0:
                plt.plot(range(1, 1 + len(vals)), vals, self.colors[ct % 7])
                plt.annotate(
                    "%s" % key,
                    # xy = (xvals[-1], 0.95),
                    # xy = (0.89, yvals[0]),
                    # xy = (1.0, yvals[0]),
                    xy=(1, vals[0]),
                    # xytext = (xvals[-1] - 0.03, 0.97),
                    # xytext = (0,95, yvals[0]),
                    verticalalignment="center",
                    horizontalalignment="center",
                    size="small",
                    color=self.colors[ct % 7],
                )
                maxy = 0
                maxx = 0
                cx = 1
                for y in vals:
                    if y > maxy:
                        maxy = y
                        maxx = cx
                    cx += 1
                plt.scatter(maxx, maxy, s=20)

                ct += 1

        plt.savefig("%s.pdf" % (self.logfile), format="pdf", bbox_inches="tight")
        if show:
            plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: UDParse/graphs.py logs/cy-20191103-embs-wiki-D/log [UPOS XPOS UAS LAS CLAS Lemmas]")
    else:
        keys = None
        if len(sys.argv) > 2:
            keys = sys.argv[2:]
        gg = Graph(sys.argv[1], keys)
        gg.plot()
