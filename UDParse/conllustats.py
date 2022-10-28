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


# count sentences and words of a CoNLL-U file


class CountConllu:
    def __init__(self, fn):
        ifp = open(fn)

        self.sct = 0
        self.wct = 0
        for line in ifp:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            self.wct += 1
            if line.startswith("1\t"):
                self.sct += 1


if __name__ == "__main__":
    import sys

    cc = CountConllu(sys.argv[1])
    print(cc.sct, cc.wct)
