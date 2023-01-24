#!/usr/bin/env python3
# coding: utf-8


# Software Name: UDParse
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: Mozilla Public License 2.0
#
# This software is distributed under the MPL-2.0 license.
# the text of which is available at https://www.mozilla.org/en-US/MPL/2.0/
# or see the "LICENSE" file for more details.
#
# Author: Johannes HEINECKE <johannes(dot)heinecke(at)orange(dot)com> et al.


import collections
import logging
import os
import psutil
import regex
import socket
import sys
import time
import requests

# do not show FutureWarnings from numpy
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import tensorflow.compat.v1 as tf

import UDParse.progClient as progClient

# from transformers import XLMModel, XLMTokenizer
from transformers import TFGPT2Model, GPT2Tokenizer
from transformers import TFOpenAIGPTModel, OpenAIGPTTokenizer
from transformers import TFBertModel, BertTokenizer
from transformers import TFRobertaModel, RobertaTokenizer
from transformers import TFXLMRobertaModel, XLMRobertaTokenizer
from transformers import TFCamembertModel, CamembertTokenizer
from transformers import TFDistilBertModel, DistilBertTokenizer
from transformers import TFFlaubertModel, FlaubertTokenizer
from transformers import TFMT5Model, MT5Tokenizer
from transformers import TFT5Model, T5Tokenizer



import UDParse.conllustats

# ideas from  https://gitlab.tech.orange/nlp/onnx
#from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

logger = logging.getLogger("udparse")
tf.logging.set_verbosity(tf.logging.ERROR)

#from tensorflow.keras import mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)

class Embeddings:
    def __init__(self, lg, progServer=None, gpu=-1):
        logger.warning("Using Tensorflow for vectorisation " + __name__)
        self.lg = lg
        self.gpu = gpu  # GPU device used
        self.progressServer = progServer
        self.netlength = 512
        self.re_extras = regex.compile(r"^\d+-|^\d+\.")
        if self.progressServer:
            logger.info("logging progress on %s" % self.progressServer)

        logger.warning("Numpy version: %s" % np.version.full_version)

        self.display_one_of_X = 1
        if len(tf.config.experimental.list_physical_devices("GPU")) > 0:
            self.cuda_available = True
            self.display_one_of_X = 7
            #logger.warning("%d CUDA device(s) available" % len(tf.config.experimental.list_physical_devices("GPU")))
            logger.warning("using GPU device '%s'" % os.environ.get("CUDA_VISIBLE_DEVICES"))
        else:
            self.cuda_available = False

        logger.info("using '%s' to create embeddings" % lg)
        self.use_onnx = False
        with tf.device("/gpu:0"):
            if lg == "multi" or lg == "bert":
                #if self.use_onnx:
                #    # ONNX
                #    from transformers import AutoTokenizer
                #    print("============================== onnx")
                #    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)
                #    options = SessionOptions()
                #    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
                #    self.model = InferenceSession("/home/xxxx/tools/onnx/bert-base-multilingual-cased.onnx", options, providers=["CPUExecutionProvider"])
                #else:
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
                self.model = TFBertModel.from_pretrained("bert-base-multilingual-cased")

                # else:
                #    # https://github.com/huggingface/transformers/issues/677
                #    pathname = MODELDIR + "/bert-base-multilingual-cased"
                #    self.tokenizer = BertTokenizer.from_pretrained(pathname + "/bert-base-multilingual-cased-vocab.txt", do_lower_case=False)
                #    self.model = BertModel.from_pretrained(pathname)

                if self.cuda_available:
                    self.model = tf.function(self.model)
                    # print(dir(self.tokenizer))

                self.getVectors = self.getVectorsBert
            elif lg == "bert-uncased":
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
                self.model = TFBertModel.from_pretrained("bert-base-multilingual-uncased")
                if self.cuda_available:
                    self.model = tf.function(self.model)
                    # print(dir(self.tokenizer))

                self.getVectors = self.getVectorsBert
            elif lg == "electra":
                from transformers import TFElectraModel, ElectraTokenizer
                self.tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
                self.model = TFElectraModel.from_pretrained("google/electra-base-discriminator")
                if self.cuda_available:
                    self.model = tf.function(self.model)
                self.getVectors = self.getVectorsBert

            elif lg == "electra_fr":
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/electra-base-french-europeana-cased-discriminator")
                self.model = TFAutoModel.from_pretrained("dbmdz/electra-base-french-europeana-cased-discriminator")
                if self.cuda_available:
                    self.model = tf.function(self.model)
                self.getVectors = self.getVectorsBert


            elif lg == "distilbert":
                self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
                self.model = TFDistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
                if self.cuda_available:
                    self.model = tf.function(self.model)
                self.getVectors = self.getVectorsBert
            elif lg == "extremdistilbert":
                self.tokenizer = BertTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
                self.model = TFBertModel.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
                if self.cuda_available:
                    self.model = tf.function(self.model)
                self.getVectors = self.getVectorsBert

            elif lg == "itBERT":
                from transformers import TFAutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
                self.model = TFAutoModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "gaBERT":
                from transformers import TFAutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained("DCU-NLP/bert-base-irish-cased-v1")
                self.model = TFAutoModel.from_pretrained("DCU-NLP/bert-base-irish-cased-v1") #, from_pt=True) # needs torch installed
                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "arBERT":
                from transformers import TFAutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
                self.model = TFAutoModel.from_pretrained("asafaya/bert-base-arabic")

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "fiBERT":
                from transformers import AutoTokenizer, TFAutoModel  # WithLMHead

                self.tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
                self.model = TFAutoModel.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "slavicBERT":
                logging.warning("only available for pytorch, install pytorch in your environment")
                from transformers import AutoTokenizer, TFAutoModel  # WithLMHead

                self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-bg-cs-pl-ru-cased")
                self.model = TFAutoModel.from_pretrained("DeepPavlov/bert-base-bg-cs-pl-ru-cased", from_pt=True)

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "plBERT":
                logging.warning("only available for pytorch, install pytorch in your environment")
                from transformers import AutoTokenizer, TFAutoModel  # WithLMHead

                self.tokenizer = AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
                self.model = TFAutoModel.from_pretrained("dkleczek/bert-base-polish-uncased-v1", from_pt=True)

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "svBERT":
                from transformers import AutoTokenizer, TFAutoModel  # WithLMHead

                self.tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
                self.model = TFAutoModel.from_pretrained("KB/bert-base-swedish-cased")

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "nlBERT":
                from transformers import AutoTokenizer, TFAutoModel  # WithLMHead

                self.tokenizer = AutoTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased")
                self.model = TFAutoModel.from_pretrained("wietsedv/bert-base-dutch-cased")

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "flaubert":
                # https://github.com/getalp/Flaubert huggingface: flaubert/flaubert_base_cased
                # wget https://zenodo.org/record/3567594/files/xlm_bert_fra_base_lower.tar
                # useold = False
                # if useold:
                #    modelname = MODELDIR + "/xlm_bert_fra_base_lower"
                #    #print("zzzz", modelname)
                #    self.model, log = XLMModel.from_pretrained(modelname, output_loading_info=True)
                #    ##print(log)
                #
                #    ## Load tokenizer
                #    self.tokenizer = XLMTokenizer.from_pretrained(modelname, do_lowercase_and_remove_accent=False)
                # else:
                logging.warning("only available for pytorch")
                self.tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
                self.model = TFFlaubertModel.from_pretrained("flaubert/flaubert_base_cased", from_pt=True)

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "camembert" or lg == "fr":
                self.model = TFCamembertModel.from_pretrained("camembert-base")
                self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base", do_lower_case=False)

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "roberta" or lg == "en":
                self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large", do_lower_case=False)
                self.model = TFRobertaModel.from_pretrained("roberta-large")

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "mt5":
                self.tokenizer = MT5Tokenizer.from_pretrained("google/mt5-large", do_lower_case=False)
                self.model = TFMT5Model.from_pretrained("google/mt5-large")

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "t5":
                self.tokenizer = T5Tokenizer.from_pretrained("t5-large", do_lower_case=False)
                self.model = TFT5Model.from_pretrained("t5-large")

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg == "xml-roberta" or lg == "xlmr":
                #print("SSSSSSS xlmr")
                #from transformers import TFXLMRobertaModel, XLMRobertaTokenizer
                self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large", do_lower_case=False)
                if True:
                    # classic
                    self.model = TFXLMRobertaModel.from_pretrained("jplu/tf-xlm-roberta-large")
                
                    if self.cuda_available:
                        self.model = tf.function(self.model) #, jit_compile=True)

                else:
                    # slower !!
                    self.model = TFXLMRobertaModel.from_pretrained("jplu/tf-xlm-roberta-large")

                    #def model2(a, attention_mask):
                    #    return self.model1(a, attention_mask=attention_mask, training=False)
                
                
                    if self.cuda_available:
                        self.model = tf.function(input_signature=(tf.TensorSpec(shape=[None,self.netlength], dtype=tf.int32),
                                                                  tf.TensorSpec(shape=[None,self.netlength], dtype=tf.int32),
                                                                  )
                                                 )(self.model)



                self.getVectors = self.getVectorsBert

            elif lg == "gpt":
                self.tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt", do_lower_case=False)
                # DOES NOT WORK
                # print("CLS", self.tokenizer.cls_token)
                # if not self.tokenizer.cls_token:
                #    self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
                #    self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})

                self.model = TFOpenAIGPTModel.from_pretrained("openai-gpt")

                if self.cuda_available:
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

            elif lg in ["gpt2", "gpt2-medium", "gpt2-large"]:
                self.tokenizer = GPT2Tokenizer.from_pretrained(lg, do_lower_case=False)
                # DOES NOT WORK
                # if not self.tokenizer.cls_token:
                #    self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
                #    self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})

                self.model = TFGPT2Model.from_pretrained(lg)

                if self.cuda_available:
                    # print(dir(self.tokenizer))
                    self.model = tf.function(self.model)

                self.getVectors = self.getVectorsBert

        self.ctlong = 0
        logging.info("embeddings '%s' read" % lg)

    def filterOS(self, t):  # t is a 1-dim torch tensor of sentencepiece indexes
        condition = t != self.bos
        t = t[condition]
        condition = t != self.eos
        t = t[condition]
        return t

    def getVectorsBert(self, sentence):  # [[cols]]
        #aa = time.time()
        # concatenate tokens to normal sentence
        # text_with_mwt = ""
        #start = time.time()
        words = []
        for cols in sentence:
            # if "-" in cols[0]:
            if self.re_extras.match(cols[0]):
                continue
            # text_with_mwt += cols[1]
            words.append(cols[1])

        # print("=========cls:", self.tokenizer.cls_token)
        # print("=========sep:", self.tokenizer.sep_token)
        # add CLS and SEP tokens
        if self.tokenizer.cls_token:
            words = [self.tokenizer.cls_token, *words, self.tokenizer.sep_token]  # add [CLS] and [SEP] around sentence
        else:
            # GPT2 has no CLS/SEP tokens
            words = [*words]  # add nothing

        # test: use just random vectors to measure speed
        # v = []
        # for x in range(len(words)):
        #     v.append(tf.random.uniform(shape=[1024], dtype=tf.float32))
        # print("CCCC", len(v), v[0].shape, v[1].shape)
        # return v[1:-1]


        # print("AAA", len(words))
        # print("BBB", self.tokenizer.encode(words[0], None, add_special_tokens=False))
        # TODO: optimiser
        tokens = [self.tokenizer.encode(t, None, add_special_tokens=False) for t in words]

        #print("words", words)
        #print("token", tokens)
        #tokens2 = self.tokenizer.encode(words, None, add_special_tokens=False, is_split_into_words=True)

        # there are at times "words" which only consists of strange codepoints
        # the tokenizer creates an empty list. So we put the code for "." token instead
        for tt in tokens:
            if len(tt) == 0:
                tt.append(119)  # "."

        #if True: #False:  # debug:
        #     # print("TOKEN", tokens)
        #     for tt in tokens:
        #         # print("TOKS", len(tt), end=" ")
        #         for t in tt:
        #             print("%d %d/<%s>" % (len(self.tokenizer.decode([t])), t, self.tokenizer.decode([t])), end=" ")
        #         print()
        #     print()
        #     for tt in tokens2:
        #         # print("TOKS", len(tt), end=" ")
        #         for t in [tt]:
        #             print("%d %d/<%s>" % (len(self.tokenizer.decode([t])), t, self.tokenizer.decode([t])), end=" ")
        #         print()

        #bb = time.time()
        lengths = [len(w) for w in tokens]

        tokens = [t for s in tokens for t in s]  # flatten tokens list
        #print("tokensflat", tokens)

        tlen = len(tokens)
        number_of_parts = ((tlen - 1) // self.netlength) + 1
        if number_of_parts > 1:
            logging.warning("long sentence %d          " % tlen)

        # unpadded = tokens[:]
        attention = [1] * len(tokens)
        # Padding (TF very slow without padding on GPU)
        #if tlen < self.netlength:
        if self.cuda_available and tlen < self.netlength:
            # padding
            padlength = (number_of_parts * self.netlength) - tlen
            if self.tokenizer.pad_token_id == None:
                tokens += [0] * padlength
                attention += [0] * padlength
            else:
                # tokens += [self.tokenizer.pad_token_id] * ((number_of_parts * self.netlength)-tlen)
                tokens += [self.tokenizer.pad_token_id] * padlength
                attention += [0] * padlength
        else:
            attention = [1] * tlen

        #if self.use_onnx:
        #    token_tensors = tokens
        #    attention_tensor = attention
        #else:
        token_tensors = tf.convert_to_tensor([tokens])
        attention_tensor = tf.convert_to_tensor([attention])
        # unpadded_tensors = tf.convert_to_tensor([unpadded])




        llfeats = []
        #print("number", number_of_parts)
        for p in range(number_of_parts):
            #if self.use_onnx:
            #    currenttokens = token_tensors[p * self.netlength : (p + 1) * self.netlength]
            #    currentattention = attention_tensor[p * self.netlength : (p + 1) * self.netlength]
            #else:
            currenttokens = token_tensors[:, p * self.netlength : (p + 1) * self.netlength]
            currentattention = attention_tensor[:, p * self.netlength : (p + 1) * self.netlength]

            #print("ooooo", currenttokens.shape)
            #print("OOOOO", currentattention.shape)
            #if self.use_onnx:
            #    #print("ffff", currenttokens)
            #    last_layer_features = self.model.run(None, {"input_ids": [currenttokens], "attention_mask": [currentattention]})[0]
            #else:
            # T5, mT5
            if "t5" in self.lg:
                last_layer_features = self.model.encoder(currenttokens, attention_mask=currentattention, training=False)[0]
            else:
                last_layer_features = self.model(currenttokens, attention_mask=currentattention, training=False)[0]
                #last_layer_features = self.model(currenttokens, training=False)[0]

            llfeats.append(last_layer_features)

        last_layer_features = tf.concat(llfeats, 1)
        #cc = time.time()

        ci = 0  # sentencepiece index to extract
        vectors = []  # list of mean vectors for words

        for wl in range(len(lengths)):
            cw = []  # current word vectors
            # print("----", wl)
            for _ in range(lengths[wl]):
                # print(ci, len(cw), last_layer_features.shape)
                cw.append(last_layer_features[0, ci, :])
                ci += 1
            # mean = torch.mean(torch.stack(cw), dim=0)
            mean = tf.reduce_mean(tf.stack(cw), axis=0)
            # print("ZZZZ", mean.shape)
            vectors.append(mean)  # .numpy())

        #dd = time.time()
        #print("INTERNAL_VECTORISATION_PARSING:toktime:enctime:jointime", bb-aa, cc-bb, dd-cc)
        #for vct in vectors:
        #    vv = "%s" % vct
        #    print(vv.replace("\n", " "))

        if self.tokenizer.cls_token:
            return vectors[1 : len(vectors) - 1]  # for the server
        else:
            return vectors


    def process(self, outfile, fns):
        self.vectors = collections.OrderedDict()  # sentencenum: [vectors]

        block = []  # list of conllu-lines

        # senttext = None
        aa = time.time()
        for fn in fns:
            progclient = None
            if self.progressServer:
                progclient = progClient.PS_Client("UDParse Embeddings", self.progressServer)
            logging.info("reading from %s" % fn)
            cc = UDParse.conllustats.CountConllu(fn)
            logging.info("  %d sentences, %d words" % (cc.sct, cc.wct))

            ifp = open(fn)

            # self.progress(fn, 0, cc.sct)
            if progclient:
                progclient.update(
                    index=psutil.Process().pid,
                    type=self.lg,
                    gpu="%s:%s" % (socket.gethostname(), self.gpu),
                    filename=fn,
                    sentences="%d/%d" % (0, cc.sct),
                )
            for line in ifp:
                line = line.strip()
                if not line:
                    # process sentence
                    sid = len(self.vectors)  # count sentences by counting vectors (there is one vector per sentence)

                    # print(block)
                    if sid % self.display_one_of_X == 0:
                        print("sentence %6d/%d" % (sid, cc.sct), end="\r")
                        if progclient and sid % 28 == 1:
                            progclient.update(
                                index=psutil.Process().pid,
                                type=self.lg,
                                gpu="%s:%s" % (socket.gethostname(), self.gpu),
                                filename=fn,
                                sentences="%d/%d" % (sid, cc.sct),
                            )

                    self.vectors["%s" % sid] = self.getVectors(block)

                    block = []

                else:
                    # if line.startswith("# text ="):
                    #    senttext = line[9:].strip()
                    if line[0] == "#":
                        continue
                    else:
                        elems = line.split("\t")
                        block.append(elems)
            if block:
                sid = len(self.vectors)
                self.vectors["%s" % sid] = self.getVectors(block)

            # print(len(self.vectors), "                  ")
            progclient = None

        # print(len(self.vectors), "                  ")
        bb = time.time()
        print("%d sentences processed in %d secs" % (cc.sct, bb-aa))

        logging.warning("number of long sentences: %d" % self.ctlong)
        # print(self.vectors)

        # create npz
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.savez(outfile, **self.vectors)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lg", default="multi", type=str, help="Language (fr/en)")
    parser.add_argument("--outfile", "-o", required=True, help="output filename")
    parser.add_argument("--infiles", "-i", required=True, nargs="+", help="output filename")

    if len(sys.argv) < 1:
        parser.print_help()
    else:
        args = parser.parse_args()

    cc = Embeddings(lg=args.lg)

    cc.process(args.outfile, args.infiles)
