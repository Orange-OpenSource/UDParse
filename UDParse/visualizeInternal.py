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

# class to create heatmaps from dependency analysis
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io


class VisuDep:
    def __init__(self):
        pass

    ##    def nnheatmap(self, data, labels):
    ##        #print(data)
    ##        #print(data[1:])

    ##        # delete first row (since root is nowhere a dependant
    ##        data = data[1:]
    ##        ylabels = labels[1:]
    ##        xlabels = labels
    ##        plt.imshow(data) #, cmap='hot', interpolation='nearest')

    ##        # ... and label them with the respective list entries
    ##        plt.yticks(range(len(ylabels)), ylabels)

    ##        plt.xlabel("heads")
    ##        plt.ylabel("dependants")

    ##        # Rotate the tick labels and set their alignment.
    ##        #plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
    ##        #         rotation_mode="anchor")
    ##        plt.xticks(rotation=90)

    ##        plt.savefig("t.pdf", bbox_inches='tight')

    def heatmap(self, data, heads, labels):
        # print(data, labels)
        # return

        # delete first row (since root is nowhere a dependant)
        data = data[1:]
        ylabels = labels[1:]
        xlabels = labels

        fig, ax = plt.subplots()
        im = ax.imshow(data)

        plt.ylim(-0.5, len(data) - 0.5)

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_yticks(np.arange(len(ylabels)))
        # ax.set_xticks(np.arange(data.shape[1]+1))
        # ax.set_yticks(np.arange(data.shape[0]+1))
        # ... and label them with the respective list entries
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)

        plt.xlabel("heads")
        plt.ylabel("dependants")

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", va="center", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fontsize = 10
        if len(xlabels) > 6:
            fontsize = int(10 - 1 - len(xlabels) / 4)
        # print("font", fontsize)
        # print("HEADS", heads)
        if fontsize > 1:
            for i in range(len(ylabels)):
                for j in range(len(xlabels)):
                    if j == heads[i + 1]:
                        ff = fontsize  # 10
                        col = "r"
                    else:
                        ff = fontsize
                        col = "w"
                    text = ax.text(j, i, "%.2f" % data[i, j], fontsize=ff, ha="center", va="center", color=col)

        # ax.set_title("Harvest of local farmers (in tons/year)")
        # fig.tight_layout()
        # plt.show()
        plt.savefig("t.pdf", bbox_inches="tight")
        ofp = io.StringIO()
        plt.savefig(ofp, format="svg")
        plt.close()
        # print(ofp.getvalue())
        return ofp.getvalue()
