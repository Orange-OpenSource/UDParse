#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# Software Name: UDParse
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: Mozilla Public License 2.0
#
# This software is distributed under the MPL-2.0 license.
# the text of which is available at https://www.mozilla.org/en-US/MPL/2.0/
# or see the "LICENSE" file for more details.
#
# Author: Johannes HEINECKE <johannes(dot)heinecke(at)orange(dot)com> et al.

# Wrapper for udpipe in order to process paragraphes (list of sentences) without having
# udpipe resegments the sentences

from __future__ import absolute_import, division, print_function

import codecs
import logging
import sys

logger = logging.getLogger("udparse")

import ufal.udpipe


class UdpTok2:
    # utiliser le tokenizer d'UDPipe avec presegmented
    def __init__(self, modelname, presegmented=False):
        #print("Loading tokenizer model: %s (presegmented: %s)" % (modelname, presegmented), file=sys.stderr)
        logger.info(f"Loading tokenizer model: {modelname} (presegmented: {presegmented})")
        self.udpipemodel = ufal.udpipe.Model.load(modelname, None)  # None: no lexicon.model
        self.errorTokenizer = ufal.udpipe.ProcessingError()
        if not presegmented:
            self.tokenizer = self.udpipemodel.newTokenizer(self.udpipemodel.DEFAULT)  # TOKENIZER_PRESEGMENTED)
        else:
            self.tokenizer = self.udpipemodel.newTokenizer(self.udpipemodel.TOKENIZER_PRESEGMENTED)
        # self.udpipeoutput = ufal.udpipe.OutputFormat.newOutputFormat("conllu")

    def tokenize(self, text):
        self.tokenizer.setText(text.strip())
        sentence = ufal.udpipe.Sentence()
        sentences = []
        while self.tokenizer.nextSentence(sentence, self.errorTokenizer):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if self.errorTokenizer.occurred():
            raise Exception(self.errorTokenizer.message)
            # print >> sys.stderr, "zzz", self.udpipeoutput.writeSentence(sentences[0])
        # return self.udpipeoutput.writeSentence(sentences[0])
        return sentences


class UdpTagDep:
    # utiliser UDPipe avec une pipeline
    def __init__(self, modelname, lexmodelname=None, parsing=True):
        logger.info("Loading parse model: %s" % modelname)
        self.udpipemodel = ufal.udpipe.Model.load(modelname, lexmodelname)
        #                                     model, inputformat, tagger, parser, outputformat
        if parsing:
            p = ufal.udpipe.Pipeline.DEFAULT
        else:
            p = ufal.udpipe.Pipeline.NONE
        self.pipeline = ufal.udpipe.Pipeline(self.udpipemodel, "conllu", ufal.udpipe.Pipeline.DEFAULT, p, "conllu")
        self.error = ufal.udpipe.ProcessingError()

    def tag_parse(self, conllu):
        processed = self.pipeline.process(conllu, self.error)
        if self.error.occurred():
            # raise Exception("Error: " + self.errorTokenizer.message)
            raise Exception("Error: %s" % self.error.occurred())
        return processed


class Udpipe:
    def __init__(self, tokmodel, parsemodel, lexmodel=None, presegmented=False, parsing=True):
        logger.info("Using tokmodel: %s, parsemodel: %s" % (tokmodel, parsemodel))
        self.tokmodel = tokmodel
        self.parsemodel = parsemodel
        self.lexmodel = lexmodel
        if tokmodel:
            # self.tokenizer = UdpTok2(tokmodel, presegmented=presegmented)
            self.tokenizer = UdpTok2(tokmodel, presegmented=False)
            self.tokenizerPreSeg = UdpTok2(tokmodel, presegmented=True)
        else:
            self.tokenizer = None

        if parsemodel:
            self.tagparser = UdpTagDep(parsemodel, lexmodel, parsing=parsing)
        else:
            self.tagparser = None
        self.udpipeoutput = ufal.udpipe.OutputFormat.newOutputFormat("conllu")

    def process_line(self, text, presegmented=False):
        if self.tokenizer:
            if presegmented:
                sentences = self.tokenizerPreSeg.tokenize(text)
            else:
                sentences = self.tokenizer.tokenize(text)

            result = []
            for s in sentences:
                result.append(self.udpipeoutput.writeSentence(s))
            tok = "".join(result)

            if self.tagparser:
                return self.tagparser.tag_parse(tok)
            else:
                return tok
        else:
            # no tagger
            return self.tagparser.tag_parse(text)
            # result = []
            # for s in sentences:
            #    result.append(self.udpipeoutput.writeSentence(s))
            # return '\n'.join(result)

    def processfile(self, fn):
        ifp = open(fn)
        paragraph = []
        for line in ifp:
            line = line.strip()
            res = self.process_line(line)
            print(res)


if __name__ == "__main__":

    from optparse import OptionParser

    parser = OptionParser(usage="usage: %prog [options]", description="call udpipe")
    parser.add_option("--tok", dest="tokmodel", default=None, help="tokenizer model")
    parser.add_option("--tagparse", dest="tagparsemodel", default=None, help="model for tagging and parsing")
    parser.add_option("--tag", dest="tagmodel", default=None, help="model for tagging (no parsing, ignored if --tagparse is used)")
    parser.add_option("--lex", dest="lexmodel", default=None, help="model for lexicon")
    parser.add_option("--file", dest="file", default=None, help="file to process (one sentence per line, empty line between paragraphs)")

    (options, comargs) = parser.parse_args()
    if options.tokmodel == None and options.tagmodel == None:
        parser.print_help()

    else:
        if options.tagmodel:
            tm = options.tagmodel
            p = False
        else:
            tm = options.tagparsemodel
            p = True
        ud = Udpipe(options.tokmodel, tm, options.lexmodel, parsing=p)

        if options.file:
            ud.processfile(options.file)
        else:
            # print ud.process_sentences(["la souris à mangé le très grand fromage.", "Maintenant elle dort.",  "Ensuite elle va de nouveau manger."])
            while True:
                sent = input(">> ")
                if not sent:
                    break
                res = ud.process_line(sent)
                print(res)
