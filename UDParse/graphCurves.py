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


# faire un courbe avec des fichiers eval_out.txt

import matplotlib.pyplot as plt
import collections


class Result:
    def __init__(self, fn):
        self.vals = {"UPOS": 0, "Lemmas": 0, "LAS": 0, "CLAS": 0}

        filename = "logs/%s" % fn
        if fn[0] == "/":
            filename = "%s" % fn
        ifp = open(filename)
        for line in ifp:
            line = line.strip()
            elems = line.split()
            if line.startswith("UPOS"):
                self.vals["UPOS"] = float(elems[6])
            elif line.startswith("Lemmas"):
                self.vals["Lemmas"] = float(elems[6])
            elif line.startswith("LAS"):
                self.vals["LAS"] = float(elems[6])
            elif line.startswith("CLAS"):
                self.vals["CLAS"] = float(elems[6])


class Graphs:
    def __init__(self, dataset, data, outdir, format, dpi=200):
        vals = data["tests"]
        title = data["name"]

        results = collections.OrderedDict()
        for rep, name in vals:
            res = Result(rep)
            results[name] = res

        ymin = 70
        if "min" in data:
            ymin = data["min"]
        plt.rcParams["font.family"] = "Lato"
        plt.rcParams["font.weight"] = "medium"

        plt.rcParams["figure.figsize"] = [2.5, 4]
        # print(plt.rcParams["figure.figsize"])

        if len(vals) < 4:
            plt.rcParams["figure.figsize"] = [1.5, 4]
            #    plt.gcf().subplots_adjust(left=0.2) # 3
        elif len(vals) < 5:
            plt.rcParams["figure.figsize"] = [2.4, 4]
        else:
            plt.rcParams["figure.figsize"] = [4.5, 4]

        # else:
        # ne fonctionne plus
        # plt.rcParams["figure.figsize"] = [6.4, 4.8]
        # plt.gcf().subplots_adjust(left=0)

        plt.rcParams["figure.dpi"] = dpi

        # plt.xlabel("config") #, fontweight="medium")
        plt.ylabel("%", fontweight="medium")

        plt.title(title, fontweight="medium")
        plt.grid(axis="y")
        plt.ylim(ymin, 100)

        self.colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]

        ct = 0

        xvals = range(len(results.items()))

        items = results.items()

        for metric in "LAS", "CLAS", "UPOS", "Lemmas":
            xticks = []
            yvals = []
            for n, r in items:
                xticks.append(n)
                yvals.append(r.vals[metric])
            plt.plot(xvals, yvals, "o", ls="-", color=self.colors[ct % 7], linewidth=3)
            plt.xticks(xvals, xticks)

            # legend on line
            ypos = 10
            if metric in ["CLAS"]:
                ypos = -10
            plt.annotate(
                "%s" % metric,
                xy=(0, yvals[0]),
                # xytext = (0, yvals[0] + 0.5),
                textcoords="offset pixels",  # xytext is in pixel from xy
                xytext=(0, ypos),
                verticalalignment="center",
                horizontalalignment="center",
                # size = "small",
                fontweight="bold",
                color=self.colors[ct % 7],
            )

            maxy = 0
            maxx = 0
            cx = 0
            for y in yvals:
                if y > maxy:
                    maxy = y
                    maxx = cx
                cx += 1
            # best value
            # plt.scatter(maxx, maxy, s=40)
            plt.annotate(
                "%s" % maxy,
                xy=(maxx, maxy),
                # xytext = (maxx, maxy - 1),
                textcoords="offset pixels",  # xytext is in pixel from xy
                xytext=(0, -10),
                verticalalignment="center",
                horizontalalignment="center",
                fontweight="bold",
                # size = "small",
                color=self.colors[ct % 7],
            )

            ct += 1
        # plt.savefig("%s.pdf" % (title.replace(" ", "_")), format='pdf')
        if outdir:
            # plt.savefig("%s/%s.png" % (outdir, title.replace(" ", "_")), format='png')
            print("writing %s/Graph_%s.%s" % (outdir, dataset, format))
            plt.savefig("%s/Graph_%s.%s" % (outdir, dataset, format), format=format, bbox_inches="tight")

            plt.clf()
        else:
            plt.show()
        plt.close()  # to forget current figure


if __name__ == "__main__":
    import yaml, os, sys

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", "-o", default=None, type=str, help="output directory for graphs, without open graph in window")
    parser.add_argument("--lg", default=None, type=str, help="output only graphs for given language (comma separated list)")
    parser.add_argument("--format", "-f", default="pdf", type=str, help="format of graph files (pdf, png, jpg, svg)")
    parser.add_argument("--dpi", default=200, type=int, help="dpi of image (png, jpg)")
    # parser.add_argument("--elmos", default=False, action="store_true", help="calculate contextual embeddings for this file")

    parser.print_help()

    args = parser.parse_args()

    lgs = set()
    if args.lg:
        lgs = set(args.lg.split(","))
    datasets = yaml.safe_load(open(os.path.dirname(__file__) + "/graphsets.yml"))
    for dataset, data in sorted(datasets.items()):
        if lgs and not dataset in lgs:
            continue

        # print(dataset, data["name"])
        # print(data["tests"])
        try:
            gg = Graphs(dataset, data, outdir=args.outdir, format=args.format, dpi=args.dpi)
        except:
            print("problem with", dataset, sys.exc_info()[1])
        # break
    pass
