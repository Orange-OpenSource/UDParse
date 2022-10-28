#!/usr/bin/env python3


# Software Name: UDParse
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: Mozilla Public License 2.0
#
# This software is distributed under the MPL-2.0 license.
# the text of which is available at https://www.mozilla.org/en-US/MPL/2.0/
# or see the "LICENSE" file for more details.
#
# Author: Johannes HEINECKE <johannes(dot)heinecke(at)orange(dot)com>.


# general script which trains/tests/runs server.
# it's data is defined in the data-dict


import logging
import sys



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("language", nargs="?", type=str, help="name of the configuration to be loaded")
    parser.add_argument("--outfile", "-o", default=None, type=str, help="output file for predict (optionally for test)")
    parser.add_argument("--evalfile", "-e", default=None, type=str, help="where to write the evaluation report")
    parser.add_argument("--infile", "-i", default=None, type=str, help="tokenised conllu/text infile to predict")
    parser.add_argument("--istext", "-I", default=False, action="store_true", help="input (for predict) is text")
    parser.add_argument("--example", default=None, type=str, help="example sentence to quickly test a model (implies --istext)")
    parser.add_argument("--presegmented", "-S", default=False, action="store_true", help="input text (for predict) is a line per sentence")

    parser.add_argument("--pytorch", default=False, action="store_true", help="use pytorch instead of tensorflow for vectorisation")
    parser.add_argument("--forcegpu", default=False, action="store_true", help="force gpu for predict")
    parser.add_argument("--gpu", default=-2, type=int, help="force a gpu card (different from the one in data.yml)")
    parser.add_argument("--port", default=8844, type=int, help="server port (overrides port in config file)")

    parser.add_argument("--progressServer", "-s", default=None)  # "http://localhost:6464", type=str, help="progress server url (http://server:port",

    parser.add_argument("--yml", default=None, type=str, help="yaml-configuration file (use data.yml in same directory as default")
    parser.add_argument("-a", "--action", default="test", type=str, help="show|test|predict|train")
    parser.add_argument("-l", "--list", default=False, action="store_true", help="List defined configurations")
    parser.add_argument("--force", default=False, action="store_true", help="Overwrite an exiting output directory")

    parser.add_argument("--loglevel", help="log level (debug,info,warning,error)", type=str, default="warning")


    # parser.add_argument("--heatmap", default=False, action="store_true", help="Show heatmap (only in servermode)")

    if len(sys.argv) < 2:
        parser.print_help()
    else:
        args = parser.parse_args()

        import UDParse

        level = UDParse.loglevels.get(args.loglevel.lower())
        if level == None:
            print("invalid loglevel, using \"warning\"", file=sys.stderr)
            level = UDParse.loglevels.get("warning")
        
        logger = logging.getLogger("udparse")
        logger.setLevel(level)
        logging.basicConfig(level=level,
                            format="%(levelname)s %(name)s %(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


        act = None

        if args.list:
            act = UDParse.Action.LIST
        else:
            if args.action.startswith("te"):
                act = UDParse.Action.TEST
            elif args.action.startswith("pr"):
                act = UDParse.Action.PREDICT
            elif args.action.startswith("tr"):
                act = UDParse.Action.TRAIN
            elif args.action.startswith("tt"):
                act = UDParse.Action.TRAIN_TEST
            elif args.action.startswith("em"):
                act = UDParse.Action.EMB
            elif args.action.startswith("sh"):
                act = UDParse.Action.SHOW
            elif args.action.startswith("ser"):
                act = UDParse.Action.SERVER
            else:
                print("invalid action")

        if act != None:
            udp = UDParse.UDParse(
                args.language,
                action=act,
                yml=args.yml,
                gpu=args.gpu,
                forcegpu=args.forcegpu,
                usepytorch=args.pytorch,
                ps=args.progressServer,
                forceoutdir=args.force,
            )

            if act == UDParse.Action.SERVER:
                import UDParse.server
                udpserver = UDParse.server.UDarseServer(udp, args.port, udp.args)

                #import api.server_api
                #app = api.server_api.init_flask_app_and_api(None, args.port)
                #api.server_api.add_udparse_to_app(app, udp)

                ##if server_config is not None:
                ##    appconfig.set_config_filename(server_config)
                #app.logger.debug("Launch from Command Line")
                #app.run(host="0.0.0.0", debug=False, port=args.port) # threaded=False, processes=1)

            else:
                udp.run(
                    infile=args.infile,
                    outfile=args.outfile,
                    evalfile=args.evalfile,
                    istext=args.istext,
                    presegmented=args.presegmented,
                    example=args.example,
                )
