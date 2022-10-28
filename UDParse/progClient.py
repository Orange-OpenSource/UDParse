#!/usr/bin/python3
# a client for the progressServer

# Software Name: UDParse
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: Mozilla Public License 2.0
#
# This software is distributed under the MPL-2.0 license.
# the text of which is available at https://www.mozilla.org/en-US/MPL/2.0/
# or see the "LICENSE" file for more details.
#
# Author: Johannes HEINECKE <johannes(dot)heinecke(at)orange(dot)com> et al.


import sys
import collections
import requests


class PS_Client:
    def __init__(self, serie, url):
        self.serie = serie
        self.url = url
        self.index = None
        self.I_know_serverDown = False

    def update(self, index, **kwargs):
        try:
            params = collections.OrderedDict([("serie", self.serie), ("index", index)])
            self.index = index
            for k, v in kwargs.items():
                if k == "listtype":
                    for item in v:
                        params[item[0]] = item[1]
                else:
                    params[k] = v
            # params["logdir"] = '%s:%s' % (socket.gethostname(), args.logdir)

            # print("pppp", params)
            r = requests.get(url="%s/setlog" % (self.url), params=params)
            # print("progressserver %s" % (self.url), file=sys.stderr)
            self.I_know_serverDown = False
        except:  # Exception as e:
            if not self.I_know_serverDown:
                # display error that progress server is down only once
                print(sys.exc_info(), file=sys.stderr)
                print("cannot contact %s" % (self.url), file=sys.stderr)
                self.I_know_serverDown = True

    def setlimits(self, value, minv, maxv, inverse=False):
        # in order to colourize numerical values
        try:
            params = {
                "serie": self.serie,
                "valuename": value,
                # "inverse": inverse,
                "minval": minv,
                "maxval": maxv,
            }
            r = requests.get(url="%s/limits" % (self.url), params=params)
            print("setlimits", params)
        except:
            print(sys.exc_info())
            print("cannot set colors at %s" % (self.url), file=sys.stderr)

    def delete(self, deleteserie=False):
        if deleteserie:
            # delete whole series
            self.index = None
        self.__del__()

    def __del__(self):
        try:
            # log to progressserver, if it is running
            params = {"serie": self.serie, "index": self.index}
            r = requests.get(url="%s/cleanlog" % (self.url), params=params)
            print("progressserver clean %s" % (self.url), file=sys.stderr)
        except:
            print(sys.exc_info(), file=sys.stderr)
            print("cannot contact final %s/cleanlog" % (self.url), file=sys.stderr)


if __name__ == "__main__":
    psc = PS_Client("a", "b")
