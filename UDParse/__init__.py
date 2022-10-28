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



# general script which trains/tests/runs server.
# it's data is defined in the data-dict

import copy
from enum import Enum
import io
import json
import logging
import logging.config
import os
import shutil
import socket
import sys
import yaml
import datetime
import time

import concurrent.futures

# configuration or logger (used in all files)
logging.config.dictConfig(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "default": {
                        "format": "%(levelname)s %(name)s %(asctime)s %(message)s",
                    }
                },
                "handlers": {"default": {"class": "logging.StreamHandler", "formatter": "default", }},
                "loggers": {"udparse": {"handlers": ["default"], "level": "WARN", "propagate": False}},
            }
        )
logger = logging.getLogger("udparse")


# make tensorflow less verbose (ATTENTION: comment in case of searching for problems!!)
# must be set before loading tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import UDParse.graphs as graphs
import UDParse.iwpt20_xud_eval as ud_eval
# loaded when we know whether TF or PT is used
#import UDParse.prepareEmbeddings as prepareEmbeddings
import UDParse.ud_dataset as ud_dataset

# TODO improve
try:
    import UDParse.udpipe as udpipe
except:
    print("UDParse.udpipe not found, cannot predict text")
import UDParse.version as version


class DefaultArgs:
    def __init__(self):
        self.batch_size = 32  # Batch size.
        self.beta_2 = 0.99  # Adam beta 2
        self.char_dropout = 0  # Character dropout
        self.checkpoint = ""  # Checkpoint.
        self.cle_dim = 256  # Character-level embedding dimension.
        self.dropout = 0.5  # Dropout
        self.embeddings_data = None  # External contextualized embeddings to use.
        self.calculate_embeddings = None  # calculate contextual embeddings on the fly using given type (see prepareEmbeddings.py for types)
        self.epochs = "40:1e-3,20:1e-4"  # Epochs and learning rates 
                                         # (40 epochs with LF 1e-3 followed by 20 epochs with LR 1e-4)
        # self.epochs = "1:1e-3,1:1e-4"
        self.exp = None  # Experiment name.
        self.label_smoothing = 0.03  # Label smoothing.
        self.logdir = "" # Model and log directory.
        self.max_sentence_len = 200  # Max sentence length.
        self.min_epoch_batches = 300  # Minimum number of batches per epoch.
        self.parse = True  # Parse.
        self.parser_layers = 1  # Parser layers.
        self.parser_deprel_dim = 128  # Parser deprel dim.
        self.predict = False  # action="store_true",  Only predict.
        self.tokmodel = None  # Use for tokenisation (in the a server).
        self.predict_input = None  # Input to prediction.
        self.predict_output = None  # Output to prediction.
        self.rnn_cell = "LSTM"  # RNN cell type.
        self.rnn_cell_dim = 512  # RNN cell dimension.
        self.rnn_layers = 2  # RNN layers.
        self.rnn_layers_parser = 1  # Parser RNN layers.
        self.rnn_layers_tagger = 1  # Tagger RNN layers.
        self.seed = 42  # Initial random seed.
        self.tags = "UPOS,XPOS,FEATS,LEMMAS"  # Tags.
        self.tag_layers = 1  # Additional tag layers.
        self.threads = 4  # 8 # 4 #   Maximum number of threads to use.
        self.we_dim = 512  # Word embedding dimension.
        self.word_dropout = 0.2  # Word dropout
        self.train_input = None  # train file (list) # (comma separated list of train files)
        self.dev_input = None  # dev file list #(comma separated list, used during training to stop early)
        self.test_input = None  # test file list #(comma separated list, used during training/testing to show performance)
        self.server = False
        self.fixedchunks = None  # file with a list of fixed chunks
        self.progressServer = None
        self.heatmap = False  # only for server
        self.originalinfo = None # to make the server tell the client where the model comes from

    def __repr__(self):
        res = []
        for x in dir(self):
            if x[0] != "_":
                val = eval("self.%s" % x)
                res.append("%-20s\t%s" % (x, val))
            # res.append("%s" % eval("self.%s" % x))
        return "\n".join(res)


class Action(Enum):
    TEST = 0
    SERVER = 1
    PREDICT = 2
    TRAIN = 4
    TRAIN_TEST = 8
    EMB = 0x10
    SHOW = 0x100
    LIST = 0x1000


loglevels = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG
}



class UDParse:
    def __init__(
        self,
        lg: str,  # id of the paragraph in the config file
        action: Action = Action.TEST,  #
        yml: str = None,  # config directory (where to find a data.yml file) or config file name
        gpu: int = -2,
        forcegpu: bool = False,
        usepytorch: bool = False,
        ps=None,  # TODO add comment
        forceoutdir=False,
    ):
        if not yml:
            self.alldata = yaml.safe_load(open(os.path.dirname(__file__) + "/data.yml"))
            self.ymlbasedir = os.path.abspath(os.path.dirname(__file__) + "/..")
        else:
            # if yml is a directory, add "data.yml"
            if os.path.isdir(yml):
                yml += "/data.yml"
            self.alldata = yaml.safe_load(open(yml))
            self.ymlbasedir = os.path.abspath(os.path.dirname(yml))

        self.lg = lg  # config name
        self.cc = None  # calculate contextual embeddings
        self.progServer = ps
        self.usepytorch = usepytorch

        self.action = action

        print("UDParse v%s" % version.getVersion())

        if self.action == Action.LIST:
            for k in self.alldata["configs"]:
                if k != "TEMPLATE":
                    print("  %-20s %s" % (k, self.alldata["configs"][k]["out"]))
        else:
            if not yml:
                if lg not in self.alldata["configs"]:
                    self.errmsg("language '%s' not defined" % lg)
            else:
                # if we use another yml-file the default language is the first in the yml file
                # usually there is just one language in the logs/*/data.yml
                if lg not in self.alldata["configs"]:
                    lg = list(self.alldata["configs"].keys())[0]

            self.data = self.alldata["configs"][lg]
            #print(self.data)

            if "framework" in self.data:
                if (self.usepytorch and self.data["framework"] != "PT") \
                   or (not self.usepytorch and self.data["framework"] != "TF"):
                    if self.usepytorch:
                        logger.warning("*** model was trained with %s vectorisation, do not use --pytorch" % (self.data["framework"]))
                    else:
                        logger.warning("*** model was trained with %s vectorisation, use --pytorch" % (self.data["framework"]))
            else:
                logger.warning("*** model does not indicate which framework (PT or TF) was used for vectorisation")

            # replace variables with values
            vars = None
            if "vars" in self.alldata:
                vars = self.alldata["vars"]
            self.resolvevars(vars, self.data)

            # show loaded configuration
            for k, vv in self.data.items():
                if Action.SERVER: # and k in ["train", "dev", "test"]:
                    break
                # if type(vv) == list:
                if isinstance(vv, list):
                    print("%-20s" % (k))
                    for v in vv:
                        msg = ""
                        if k in ["train", "dev", "test"]:
                            if not os.path.isfile(v):
                                msg = "DOES NOT EXIST: "

                        print("    %s%s" % (msg, v))
                else:
                    msg = ""
                    if k in ["dev", "test", "tokmodel"]:
                        if not os.path.isfile(vv):
                            msg = "DOES NOT EXIST: "

                    print("%-20s %s%s" % (k, msg, vv))

            if self.action in [Action.TRAIN, Action.TRAIN_TEST, Action.EMB] or forcegpu:
                if gpu > -2:
                    # force GPU-ID with option --gpu
                    self.data["gpu"] = gpu
                if self.data["gpu"] > -1:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % self.data["gpu"]
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
            else:
                # do not use GPU if we are not training
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

            # print("zzzzz", os.environ["CUDA_VISIBLE_DEVICES"])
            args = DefaultArgs()
            args.forceframework = "PT" if self.usepytorch else "TF"
            args.gpu = self.data.get("gpu", -1)
            if "rnn_layers" in self.data:
                args.rnn_layers = int(self.data["rnn_layers"])
            if "rnn_layers_parser" in self.data:
                args.rnn_layers_parser = int(self.data["rnn_layers_parser"])
            if "rnn_layers_tagger" in self.data:
                args.rnn_layers_tagger = int(self.data["rnn_layers_tagger"])
            if "rnn_cell_dim" in self.data:
                args.rnn_cell_dim = int(self.data["rnn_cell_dim"])

            args.progressServer = ps
            args.forceoutdir = forceoutdir
            self.args = args

            # we need a tokenizer for predict
            if self.action in (Action.PREDICT, Action.SERVER):
                args.tokmodel = self.data["tokmodel"]
                self.validkeys(["tokmodel"])
                self.udpipe = udpipe.Udpipe(tokmodel=args.tokmodel, lexmodel=None, parsemodel=None, presegmented=False)
            if self.action == Action.SERVER:
                # finalise initialisation of ud_parser
                # load only here to respect CUDA_VISIBLE_DEVICE
                import UDParse.ud_parser_tf2 as ud_parser

                if True: #try:
                    # read original filenames form config file
                    ifp = open("%s/config" % self.data["out"])
                    args.originalinfo = {}
                    for line in ifp:
                        line = line.strip()
                        elems = line.split(None, 1)
                        if line.startswith("train_input"):
                            # if elems[1].startswith('['):
                            exec("args.originalinfo['train'] = %s" % elems[1])
                            # else:
                            #     exec("args.originalinfo['train'] = '%s'" % elems[1])
                        elif line.startswith("dev_input"):
                            # if elems[1].startswith('['):
                            exec("args.originalinfo['dev'] = %s" % elems[1])
                            # else:
                            #    exec("args.originalinfo['dev'] = '%s'" % elems[1])
                        elif line.startswith("logdir"):
                            args.originalinfo['logdir'] = elems[1]
                        elif line.startswith("trained"):
                            args.originalinfo['trained'] = elems[1]


                args.server = True
                args.predict = True
                args.logdir = self.data["out"]
                if "calculate_embeddings" in self.data:
                    args.calculate_embeddings = self.data["calculate_embeddings"]
                print("configuration:")
                print(args)
                #self.root_factors, self.trainobj, self.network, self.calc = ud_parser.predict(args)
                def loadudparse(args, ud_parser):
                    self.root_factors, self.trainobj, self.network, self.calc = ud_parser.predict(args)

                pool = concurrent.futures.ThreadPoolExecutor()
                self.submitresult = pool.submit(loadudparse, args, ud_parser)
    



    def run(self, infile=None, outfile=None, evalfile=None, istext=False, presegmented=False, example=None):
        if example:
            istext = True
            infile = io.StringIO(example)

        if self.action == Action.TEST:
            self.test(self.args, outfile, evalfile)
        elif self.action == Action.PREDICT:
            self.predict(infile, outfile, istext=istext, presegmented=presegmented)
        elif self.action == Action.TRAIN:
            self.train(self.args, gpu=self.args.gpu)
        elif self.action == Action.TRAIN_TEST:
            # args will be modified by train we have to keep the original for the test
            argsorig = copy.copy(self.args)
            self.train(self.args, gpu=self.args.gpu)

            self.args = argsorig
            self.test(self.args, outfile, evalfile)
        elif self.action == Action.EMB:
            self.train(self.args, gpu=self.args.gpu, stopafterEmbeddings=True)
        elif self.action == Action.SHOW:
            pass

    def resolvevars(self, vars, data):
        # replace variables (filenames) with values
        for k, v in data.items():
            if v == "":
                data[k] = None

        if vars:
            for k, v in vars.items():
                for dk, dv in data.items():
                    if type(dv) == str:
                        data[dk] = dv.replace("${%s}" % k, v)
                    elif type(dv) == list:
                        orig = dv
                        modif = []
                        for f in orig:
                            modif.append(f.replace("${%s}" % k, v))
                        data[dk] = modif

        # make relative filenames absolute from data.yml's point of view
        for dk, dv in data.items():
            if dk in ["calculate_embeddings", "framework"]:
                continue  #

            if type(dv) == str:
                if data[dk][0] != "/":
                    data[dk] = self.ymlbasedir + "/" + data[dk]
            elif type(dv) == list:
                orig = dv
                modif = []
                for f in orig:
                    if f[0] != "/":
                        f = self.ymlbasedir + "/" + f
                    modif.append(f)
                data[dk] = modif

    def errmsg(self, msg):
        #print("*** %s" % msg, file=sys.stderr)
        logger.error("*** %s" % msg)
        sys.exit(2)

    def validkeys(self, keylist):
        # check whether files exist (keylist must only contain keys which have file names as values)
        for key in keylist:
            if type(self.data[key]) == str:
                if not os.path.exists(self.data[key]):
                    self.errmsg("%s file '%s' does not exist " % (key, self.data[key]))
            elif type(self.data[key]) == list:
                for f in self.data[key]:
                    if not os.path.exists(f):
                        self.errmsg("%s file '%s' does not exist " % (key, f))

    def train(self, args, gpu=-1, stopafterEmbeddings=False):
        args.predict = False

        checklist = ["train"]
        files_to_save = []

        if "embeddings" in self.data:
            args.embeddings = self.data["embeddings"]
            checklist.append("embeddings")
            # do not save, too big
            # files_to_save.append(args.embeddings)
        args.logdir = self.data["out"]
        # args.train_input = ",".join(self.data["train"])
        args.train_input = [f for f in self.data["train"]]
        files_to_save.extend(self.data["train"])

        now = datetime.datetime.now()
        args.trained = now.strftime("%Y.%m.%d %H:%M:%S")

        if "dev" in self.data:
            dev = self.data["dev"]
            if isinstance(dev, list):
                # args.dev_input = ",".join(self.data["dev"])
                args.dev_input = [f for f in self.data["dev"]]
                files_to_save.extend(self.data["dev"])
            else:
                args.dev_input = [self.data["dev"]]
                files_to_save.append(self.data["dev"])
                self.data["dev"] = [dev]
            checklist.append("dev")

        # if os.path.exists(self.data["test"]):
        # files_to_save.append(self.data["test"])
        tt = self.data["test"]
        if isinstance(tt, list):
            for t in tt:
                files_to_save.append(t)
        else:
            files_to_save.append(tt)
            self.data["test"] = [tt]

        if not args.forceoutdir and os.path.exists(args.logdir):
            self.errmsg("output directory '%s' exists already. Use --force to overwrite" % args.logdir)

        if "calculate_embeddings" in self.data:
            args.calculate_embeddings = self.data["calculate_embeddings"]

        #print()

        args.emb_suffix = "-TF"
        if self.usepytorch:
            args.emb_suffix = "-PT"
        if "embeddings_data" in self.data:
            if not os.path.exists(self.data["embeddings_data"] + "-train%s.npz" % args.emb_suffix):
                # self.errmsg("%s file '%s-train.npz' does not exist " % ("embeddings_data", self.data["embeddings_data"]))
                #print("creating %s-train.npz from %s" % (self.data["embeddings_data"], self.data["train"]))
                logger.info("creating %s-train%s.npz from %s" % (self.data["embeddings_data"], args.emb_suffix, self.data["train"]))
                self.createContextual(
                    lg=args.calculate_embeddings,
                    infiles=self.data["train"],
                    outfile="%s-train%s.npz" % (self.data["embeddings_data"], args.emb_suffix),
                    gpu=gpu,
                )

            if args.dev_input and not os.path.exists(self.data["embeddings_data"] + "-dev%s.npz" % args.emb_suffix):
                # self.errmsg("%s file '%s-dev.npz' does not exist " % ("embeddings_data", self.data["embeddings_data"]))
                #print("creating %s-dev.npz from %s" % (self.data["embeddings_data"], self.data["dev"]))
                logger.info("creating %s-dev%s.npz from %s" % (self.data["embeddings_data"], args.emb_suffix, self.data["dev"]))
                self.createContextual(
                    lg=args.calculate_embeddings,
                    infiles=self.data["dev"],  # args.dev_input,
                    outfile="%s-dev%s.npz" % (self.data["embeddings_data"], args.emb_suffix),
                    gpu=gpu,
                )

            args.embeddings_data = self.data["embeddings_data"]
            files_to_save.append(self.data["embeddings_data"] + "-train%s.npz" % args.emb_suffix)
            if args.dev_input:
                files_to_save.append(self.data["embeddings_data"] + "-dev%s.npz" % args.emb_suffix)

            if not os.path.exists(self.data["embeddings_data"] + "-test%s.npz" % args.emb_suffix):
                #print("creating %s-test.npz from %s" % (self.data["embeddings_data"], self.data["test"]))
                logger.info("creating %s-test%s.npz from %s" % (self.data["embeddings_data"], args.emb_suffix, self.data["test"]))
                self.createContextual(
                    lg=args.calculate_embeddings,
                    infiles=self.data["test"],
                    outfile="%s-test%s.npz" % (self.data["embeddings_data"], args.emb_suffix),
                    gpu=gpu,
                )
            files_to_save.append(self.data["embeddings_data"] + "-test%s.npz" % args.emb_suffix)

        if stopafterEmbeddings:
            return

        os.makedirs(args.logdir, exist_ok=True)
        self.validkeys(checklist)

        # deep copy data (to create data.yml in logdir)
        newdata = {}
        for k, v in self.data.items():
            newdata[k] = v

        if os.path.exists(self.data["tokmodel"]):
            args.tokmodel = self.data["tokmodel"]
            files_to_save.append(self.data["tokmodel"])
            # self.data["tokmodel"] = self.data["tokmodel"].split("/")[-1]
            if "tokmodel" in newdata:
                newdata["tokmodel"] = newdata["tokmodel"].split("/")[-1]

        # store all configuration (args) data in outdir
        with open("{}/config".format(args.logdir), "w") as ofp:
            print(args, file=ofp)
            ofp.close()

        # copy important data to the output directory
        # and change self.data to be sure the data.yml is correct
        for f in files_to_save:
            logger.debug("copy %s to %s" % (f, args.logdir))
            shutil.copy2(f, args.logdir)

        def deletepath(obj):
            if isinstance(obj, list):
                t = []
                for f in obj:
                    t.append(f.split("/")[-1])
                return t
            else:
                return obj.split("/")[-1]

        if "embeddings_data" in newdata:
            newdata["embeddings_data"] = newdata["embeddings_data"].split("/")[-1]

        if "dev" in newdata:
            # newdata["dev"] = newdata["dev"].split("/")[-1]
            newdata["dev"] = deletepath(newdata["dev"])

        # newdata["test"] = newdata["test"].split("/")[-1]
        newdata["test"] = deletepath(newdata["test"])

        # t = []
        # for f in newdata["train"]:
        #    t.append(f.split("/")[-1])
        # newdata["train"] = t
        newdata["train"] = deletepath(newdata["train"])

        # store yaml configuration data in outdir
        with open("{}/data.yml".format(args.logdir), "w") as ofp:
            newdata["out"] = "."
            newdata["framework"] = "PT" if self.usepytorch else "TF"
            tmp = {"configs": {self.lg: newdata}}  # "vars" : self.alldata["vars"],
            yaml.safe_dump(tmp, ofp)
            ofp.close()

        #print()

        # load only here to respect CUDA_VISIBLE_DEVICE
        import UDParse.ud_parser_tf2 as ud_parser

        ud_parser.train(args)

    def createContextual(self, lg, infiles, outfile, gpu=-1):
        # used for training
        if not self.cc:
            if self.usepytorch:
                import UDParse.prepareEmbeddingsPT as prepareEmbeddings
            else:
                import UDParse.prepareEmbeddings as prepareEmbeddings
            self.cc = prepareEmbeddings.Embeddings(lg=lg, progServer=self.progServer, gpu=gpu)
        self.cc.process(outfile, infiles)

    # def server(self, args, port=0): #, heatmap=False):
    #     # start the server
    #     args.port= self.data["port"]
    #     if port != 0:
    #         # override port from data.yml
    #         args.port = port
    #     args.predict = True
    #     args.logdir = self.data["out"]
    #     #args.heatmap = heatmap

    #     if "calculate_embeddings" in self.data:
    #         args.calculate_embeddings = self.data["calculate_embeddings"]

    #     checklist = []
    #     if "tokmodel" in self.data:
    #         args.tokmodel = self.data["tokmodel"]
    #         checklist.append("tokmodel")

    #     if "embeddings" in self.data:
    #         args.embeddings = self.data["embeddings"]
    #         checklist.append("embeddings")

    #     if "fixedchunks" in self.data:
    #         args.fixedchunks = self.data["fixedchunks"]
    #         checklist.append("fixedchunks")

    #     self.validkeys(checklist)

    #     print("using Tensorflow %d" % self.tfversion)
    #     if self.tfversion == 1:
    #         # TF1.14
    #         # import bin.ud_parser as ud_parser
    #         import ud_parser as ud_parser
    #     else:
    #         # TF 2.5.0
    #         # import bin.ud_parser_tf2 as ud_parser
    #         import ud_parser_tf2 as ud_parser

    #     ud_parser.predict(args)


    #   datatypye: pure text or tokenized conllu
    #   presegmented
    def predict(self, infile, outfile, istext=False, presegmented=False):
        # process a tokenized CoNLL-U file
        if not infile:
            self.errmsg("option --infile needed")

        self.args.predict_output = outfile
        checklist = ["out"]

        if istext:
            if isinstance(infile, io.StringIO):
                # infile is not a filename but a String.IO instance, ready to be read
                text_input = infile.read()
            else:
                ifp = open(infile)
                text_input = ifp.read()
                ifp.close()
                if not outfile:
                    self.errmsg("option --outfile needed")

            tok = self.udpipe.process_line(text_input, presegmented=presegmented)
            self.args.predict_input = [io.StringIO(tok)]
        else:
            self.args.predict_input = [infile]

        self.args.predict = True
        self.args.logdir = self.data["out"]

        self.args.emb_suffix = "-TF"
        if self.usepytorch:
            self.args.emb_suffix = "-PT"

        # only needed for W2V ....
        if "embeddings" in self.data:
            self.args.embeddings = self.data["embeddings"]
            checklist.append("embeddings")

        if "embeddings_data" in self.data:
            # if not os.path.exists(self.data["embeddings_data"] + "-test.npz"):
            #    self.errmsg("%s file '%s-test.npz' does not exist " % ("embeddings_data", self.data["embeddings_data"]))
            self.args.embeddings_data = self.data["embeddings_data"]

        if "calculate_embeddings" in self.data:
            self.args.calculate_embeddings = self.data["calculate_embeddings"]

        self.validkeys(checklist)

        # print("using Tensorflow %d" % self.tfversion)
        # load only here to respect CUDA_VISIBLE_DEVICE
        import UDParse.ud_parser_tf2 as ud_parser

        ud_parser.predict(self.args)

    def api_process(self, in_text: str, is_text: bool = False, is_pre_segmented: bool = False,
                    tok_only: bool = False,
                    do_parse: bool = True):
        """ process content submitted through the web API

        Args:
            in_text (str): [raw text or CoNLL-U text to be parsed]
            is_text (bool, optional): [if true in_text is in CoNLL-U format (segmented and tokenised)]. Defaults to False.
            is_pre_segmented (bool, optional): [description]. Defaults to False.
            do_parse (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """

        # only called by server

        if is_text:
            tok = self.udpipe.process_line(in_text, presegmented=is_pre_segmented)
            # replace QQQQ again by spaces (produced for French in process.py:process_inputs()
            tok = tok.replace("QQQQ", " ")
            fp = io.StringIO(tok)
        else:
            fp = io.StringIO(in_text.rstrip() + "\n\n")
            tok = in_text

        if tok_only:
            return tok

        # start time of vectorisation
        aa = time.time()
        text = ud_dataset.UDDataset([fp],
                                    self.root_factors,
                                    train=self.trainobj,
                                    shuffle_batches=False,
                                    emb_calculator=self.calc
                                    )
        # end time of vectorisation, start time of tagging/parsing
        bb = time.time()
        self.args.parse = True
        if not do_parse:
            self.args.parse = False

        result = self.network.predict(text, False, self.args, self.trainobj)
        # end time of sentence processing
        cc = time.time()

        # show time of vectorisation and tagging/parsing to be used with getVectorisationAndParsingTime.py
        #print("VECTORISATION_PARSING:vectime:parsetime", bb-aa, cc-bb)
        #sys.stdout.flush()
        return result

    def api_config(self):
        # sent configuration data
        #return "%s\n" % self.args
        return json.dumps(self.args.__dict__)

    def api_info(self):
        cwd = os.path.abspath(os.path.curdir)
        hostname = socket.gethostname()
        #return "%s:%s\n%s\n" % (hostname, cwd, " ".join(sys.argv))
        dico = {
            "hostname": hostname,
            "cwd": cwd,
            "commandline":  " ".join(sys.argv),
            "version": version.getVersion()
        }
        conda_env = os.getenv("CONDA_DEFAULT_ENV")
        if conda_env:
            dico["conda_default_dev"] = conda_env

        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            dico["conda_prefix"] = conda_prefix
        return json.dumps(dico)


    def api_infoh(self):
        return " ".join(sys.argv) + "\n"

    def test(self, args, outfile, evalfile):
        # run test and evaluation
        args.predict = True

        tt = self.data["test"]
        if not isinstance(tt, list):
            self.data["test"] = [tt]

        args.predict_input = self.data["test"]

        if outfile:
            args.predict_output = outfile
        else:
            args.predict_output = self.data["out"] + "/test_out.conllu"
        args.logdir = self.data["out"]

        checklist = ["out", "test"]


        self.validkeys(["out", "test"])

        args.emb_suffix = "-TF"
        if self.usepytorch:
            args.emb_suffix = "-PT"

        if "embeddings_data" in self.data:
            if not os.path.exists(self.data["embeddings_data"] + "-test%s.npz" % args.emb_suffix):
                self.errmsg("%s file '%s-test%s.npz' does not exist " % ("embeddings_data", self.data["embeddings_data"], args.emb_suffix))

            args.embeddings_data = self.data["embeddings_data"]

        # load only here to respect CUDA_VISIBLE_DEVICE
        import UDParse.ud_parser_tf2 as ud_parser

        ud_parser.predict(args)
        self.evaluate(args, evalfile)

        # plot evolution of dev-results
        try:
            gg = graphs.Graph(self.data["out"] + "/log", ["Lemmas", "UPOS", "XPOS", "LAS", "CLAS", "MLAS"], title=self.data["out"].split("/")[-1],)
            gg.plot(show=False)
        except Exception as e:
            logger.error("cannot create log graph: " + str(e))

    def evaluate(self, args, evalfile):
        def show(ofp=sys.stdout):
            print("Metric     | Precision |    Recall |  F1 Score | AligndAcc", file=ofp)
            print("-----------+-----------+-----------+-----------+-----------", file=ofp)
            for metric, val in metrics.items():
                print(
                    "{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
                        metric,
                        100 * val.precision,
                        100 * val.recall,
                        100 * val.f1,
                        "{:10.2f}".format(100 * val.aligned_accuracy) if val.aligned_accuracy is not None else "",
                    ),
                    file=ofp,
                )

        # TODO if more than one test file concatenate
        infile = "%s/test_in.conllu" % self.data["out"]
        ofp = open(infile, "w")
        for testfile in args.predict_input:
            ifp = open(testfile)
            for line in ifp:
                ofp.write(line)
            ifp.close()
        ofp.close()

        evalo = io.StringIO()
        metrics = ud_eval.evaluate(
            ud_eval.load_conllu(open(infile), {}), ud_eval.load_conllu(open(args.predict_output), {}), ofp=evalo,  # open(args.predict_input)),
        )

        show(ofp=evalo)

        logger.warning(evalo.getvalue())

        if evalfile:
            evalf = open(evalfile, "w")
        else:
            evalf = open(self.data["out"] + "/eval_out.txt", "w")

        print(evalo.getvalue(), file=evalf)
        evalo.close()
        evalf.close()
