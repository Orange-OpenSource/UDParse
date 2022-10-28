# Software Name: UDParse
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: Mozilla Public License 2.0
#
# This software is distributed under the MPL-2.0 license.
# the text of which is available at https://www.mozilla.org/en-US/MPL/2.0/
# or see the "LICENSE" file for more details.
# 
# Author: Johannes HEINECKE <johannes(dot)heinecke(at)orange(dot)com> et al.
#
# Code based on UDPipe Future <http://github.com/CoNLL-UD-2018/UDPipe-Future>.
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import io
import logging
import re

import numpy as np
import psutil
import tensorflow.compat.v1 as tf

import UDParse.progClient as progClient

logger = logging.getLogger("udparse")

def _min_edit_script(source, target, allow_copy):
    a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i in range(0, len(source) + 1):
        for j in range(0, len(target) + 1):
            if i == 0 and j == 0:
                a[i][j] = (0, "")
            else:
                if allow_copy and i and j and source[i - 1] == target[j - 1] and a[i - 1][j - 1][0] < a[i][j][0]:
                    a[i][j] = (a[i - 1][j - 1][0], a[i - 1][j - 1][1] + "→")
                if i and a[i - 1][j][0] < a[i][j][0]:
                    a[i][j] = (a[i - 1][j][0] + 1, a[i - 1][j][1] + "-")
                if j and a[i][j - 1][0] < a[i][j][0]:
                    a[i][j] = (a[i][j - 1][0] + 1, a[i][j - 1][1] + "+" + target[j - 1])
    return a[-1][-1][1]


def _gen_lemma_rule(form, lemma, allow_copy):
    form = form.lower()

    previous_case = -1
    lemma_casing = ""
    for i, c in enumerate(lemma):
        case = "↑" if c.lower() != c else "↓"
        if case != previous_case:
            lemma_casing += "{}{}{}".format("¦" if lemma_casing else "", case, i if i <= len(lemma) // 2 else i - len(lemma))
        previous_case = case
    lemma = lemma.lower()

    best, best_form, best_lemma = 0, 0, 0
    for l in range(len(lemma)):
        for f in range(len(form)):
            cpl = 0
            while f + cpl < len(form) and l + cpl < len(lemma) and form[f + cpl] == lemma[l + cpl]:
                cpl += 1
            if cpl > best:
                best = cpl
                best_form = f
                best_lemma = l

    rule = lemma_casing + ";"
    if not best:
        rule += "a" + lemma
    else:
        rule += "d{}¦{}".format(
            _min_edit_script(form[:best_form], lemma[:best_lemma], allow_copy),
            _min_edit_script(form[best_form + best :], lemma[best_lemma + best :], allow_copy),
        )
    return rule


def _apply_lemma_rule(form, lemma_rule):
    if not ";" in lemma_rule:
        logger.error("ATTENTION, invalid rule %s %s" % (form, lemma_rule))

    casing, rule = lemma_rule.split(";", 1)
    if rule.startswith("a"):
        lemma = rule[1:]
    else:
        form = form.lower()
        rules, rule_sources = rule[1:].split("¦"), []
        assert len(rules) == 2
        for rule in rules:
            source, i = 0, 0
            while i < len(rule):
                if rule[i] == "→" or rule[i] == "-":
                    source += 1
                else:
                    assert rule[i] == "+"
                    i += 1
                i += 1
            rule_sources.append(source)

        try:
            lemma, form_offset = "", 0
            for i in range(2):
                j, offset = 0, (0 if i == 0 else len(form) - rule_sources[1])
                while j < len(rules[i]):
                    if rules[i][j] == "→":
                        lemma += form[offset]
                        offset += 1
                    elif rules[i][j] == "-":
                        offset += 1
                    else:
                        assert rules[i][j] == "+"
                        lemma += rules[i][j + 1]
                        j += 1
                    j += 1
                if i == 0:
                    lemma += form[rule_sources[0] : len(form) - rule_sources[1]]
        except:
            lemma = form

    for rule in casing.split("¦"):
        if rule == "↓0":
            continue  # The lemma is lowercased initially
        if not rule:
            continue  # Empty lemma might generate empty casing rule
        case, offset = rule[0], int(rule[1:])
        lemma = lemma[:offset] + (lemma[offset:].upper() if case == "↑" else lemma[offset:].lower())

    if not lemma:
        return form  # JHE
    return lemma


class UDDataset:
    FORMS = 0
    LEMMAS = 1
    UPOS = 2
    XPOS = 3
    FEATS = 4
    HEAD = 5
    DEPREL = 6
    DEPS = 7
    MISC = 8
    FACTORS = 9
    EMBEDDINGS = 9
    ELMO = 10

    FACTORS_MAP = {
        "FORMS": FORMS,
        "LEMMAS": LEMMAS,
        "UPOS": UPOS,
        "XPOS": XPOS,
        "FEATS": FEATS,
        "HEAD": HEAD,
        "DEPREL": DEPREL,
        "DEPS": DEPS,
        "MISC": MISC,
    }

    re_extras = re.compile(r"^#|^\d+-|^\d+\.")

    class _Factor:
        ROOT = 2

        def __init__(self, with_root, characters, train=None):
            self.words_map = train.words_map if train else {"<pad>": 0, "<unk>": 1, "<root>": 2}
            self.words = train.words if train else ["<pad>", "<unk>", "<root>"]
            self.word_ids = []
            self.strings = []
            self.with_root = with_root
            self.characters = characters
            if characters:
                self.alphabet_map = train.alphabet_map if train else {"<pad>": 0, "<unk>": 1, "<root>": 2}
                self.alphabet = train.alphabet if train else ["<pad>", "<unk>", "<root>"]
                self.charseqs_map = {"<pad>": 0, "<unk>": 1, "<root>": 2}
                self.charseqs = [[0], [1], [2]]
                self.charseq_ids = []

        def __repr__(self):
            res = []
            res.append("words_map: %s" % self.words_map)
            # res.append("words: %s" % self.words)
            res.append("word_ids: %s" % self.word_ids)
            return "\n".join(res)

    def __init__(
        self,
        filenames,
        root_factors=[],
        embeddings_data=None,
        train=None,
        shuffle_batches=True,
        max_sentence_len=None,
        max_sentences=None,
        progserver=None,
        emb_calculator=None,
    ):
        # Create factors
        self._factors = []
        for f in range(self.FACTORS):
            self._factors.append(self._Factor(f in root_factors, f == self.FORMS, train._factors[f] if train else None))
        self._extras = []
        self._lr_allow_copy = train._lr_allow_copy if train else None
        lemma_dict_with_copy, lemma_dict_no_copy = {}, {}

        # Prepare embeddings
        self._embeddings = {}
        if train is not None:
            self._embeddings = train._embeddings


        # Load contextualized embeddings
        self._embeddings_data = []

        if embeddings_data and not emb_calculator:
            # we read contextual embeddings from file
            for embeddings_data_path in embeddings_data.split(","):
                with np.load(embeddings_data_path) as embeddings_data_file:
                    print("loading prepared vectors '%s'" % embeddings_data_path)
                    psc = None
                    if progserver:
                        psc = progClient.PS_Client("Contextual Embeddings", progserver)

                    for i, (zzz, value) in enumerate(embeddings_data_file.items()):
                        # print("iii", i, zzz, value.shape, max_sentence_len)
                        if i % 70 == 0:
                            print("  %d vectors read" % (i), end="\r")
                        if psc and i % 70 == 0:
                            psc.update(index=psutil.Process().pid, type="", gpu="", filename=embeddings_data_path, sentences="%d" % (i))

                        if max_sentence_len:
                            value = value[:max_sentence_len]
                        if i >= len(self._embeddings_data):
                            self._embeddings_data.append(value)
                        else:
                            self._embeddings_data[i] = np.concatenate([self._embeddings_data[i], value], axis=1)
                    assert i + 1 == len(self._embeddings_data)
                    print("  %d vectors read  " % i)
                    psc = None
        self._embeddings_data_size = self._embeddings_data[0].shape[1] if self._embeddings_data else 0

        # Load the sentences
        text = []  # list of tokenised sentences (to get contextual embeddings on the fly)
        for filename in filenames:
            if type(filename) == str:
                logger.info("read from '%s'" % filename)
                infile = open(filename, "r", encoding="utf-8")
            else:
                # string.io object
                infile = filename

            in_sentence = False
            conlluline = []  # needed to create contextual embeddings on the fly
            lct = 0
            sct = 0
            for line in infile:
                lct += 1
                if lct % 700 == 0:
                    print("%d CoNLL-U lines read..." % (lct), end="\r")
                line = line.rstrip("\r\n")
                # print("LINE:", line)
                if line:
                    if self.re_extras.match(line):
                        # comment, n-m or n.1 CoNLL-U line
                        if in_sentence:
                            while len(self._extras) < len(self._factors[0].word_ids):
                                self._extras.append([])
                            while len(self._extras[-1]) <= len(self._factors[0].word_ids[-1]) - self._factors[0].with_root:
                                self._extras[-1].append("")
                        else:
                            while len(self._extras) <= len(self._factors[0].word_ids):
                                self._extras.append([])
                            if not len(self._extras[-1]):
                                self._extras[-1].append("")
                        self._extras[-1][-1] += ("\n" if self._extras[-1][-1] else "") + line
                        continue

                    if max_sentence_len and in_sentence and len(self._factors[0].strings[-1]) - self._factors[0].with_root >= max_sentence_len:
                        continue

                    columns = line.split("\t")
                    conlluline.append(columns)
                    columns = columns[1:]

                    for f in range(self.FACTORS):
                        factor = self._factors[f]
                        if not in_sentence:
                            if len(factor.word_ids):
                                factor.word_ids[-1] = np.array(factor.word_ids[-1], np.int32)
                            factor.word_ids.append([])
                            factor.strings.append([])
                            if factor.characters:
                                factor.charseq_ids.append([])
                            if factor.with_root:
                                factor.word_ids[-1].append(factor.ROOT)
                                factor.strings[-1].append(factor.words[factor.ROOT])
                                if factor.characters:
                                    factor.charseq_ids[-1].append(factor.ROOT)

                        word = columns[f]
                        factor.strings[-1].append(word)

                        # Preprocess word
                        if f == self.LEMMAS and self._lr_allow_copy is not None:
                            word = _gen_lemma_rule(columns[self.FORMS], columns[self.LEMMAS], self._lr_allow_copy)

                        # Character-level information
                        if factor.characters:
                            if word not in factor.charseqs_map:
                                factor.charseqs_map[word] = len(factor.charseqs)
                                factor.charseqs.append([])
                                for c in word:
                                    if c not in factor.alphabet_map:
                                        if train:
                                            c = "<unk>"
                                        else:
                                            factor.alphabet_map[c] = len(factor.alphabet)
                                            factor.alphabet.append(c)
                                    factor.charseqs[-1].append(factor.alphabet_map[c])
                            factor.charseq_ids[-1].append(factor.charseqs_map[word])

                        # Word-level information
                        if f == self.HEAD:
                            factor.word_ids[-1].append(int(word) if word != "_" else -1)
                        elif f == self.LEMMAS and self._lr_allow_copy is None:
                            factor.word_ids[-1].append(0)
                            lemma_dict_with_copy[_gen_lemma_rule(columns[self.FORMS], word, True)] = 1
                            lemma_dict_no_copy[_gen_lemma_rule(columns[self.FORMS], word, False)] = 1
                        else:
                            if word not in factor.words_map:
                                if train:
                                    word = "<unk>"
                                else:
                                    factor.words_map[word] = len(factor.words)
                                    factor.words.append(word)
                            factor.word_ids[-1].append(factor.words_map[word])
                    in_sentence = True
                else:
                    in_sentence = False
                    text.append(conlluline)
                    conlluline = []
                    sct += 1
                    if max_sentences is not None and len(self._factors[self.FORMS].word_ids) >= max_sentences:
                        break
            print("%d CoNLL-U lines read..." % (lct))
            logger.info("all %d CoNLL-U lines read. %d sentences" % (lct, sct))

        # Finalize lemmas if needed
        if self._lr_allow_copy is None:
            self._lr_allow_copy = True if len(lemma_dict_with_copy) < len(lemma_dict_no_copy) else False
            lemmas = self._factors[self.LEMMAS]
            for i in range(len(lemmas.word_ids)):
                for j in range(lemmas.with_root, len(lemmas.word_ids[i])):
                    word = _gen_lemma_rule(
                        self._factors[self.FORMS].strings[i][j - lemmas.with_root + self._factors[self.FORMS].with_root],
                        lemmas.strings[i][j],
                        self._lr_allow_copy,
                    )
                    if word not in lemmas.words_map:
                        lemmas.words_map[word] = len(lemmas.words)
                        lemmas.words.append(word)
                    lemmas.word_ids[i][j] = lemmas.words_map[word]

        # Compute sentence lengths
        sentences = len(self._factors[self.FORMS].word_ids)
        self._sentence_lens = np.zeros([sentences], np.int32)
        for i in range(len(self._factors[self.FORMS].word_ids)):
            self._sentence_lens[i] = len(self._factors[self.FORMS].word_ids[i]) - self._factors[self.FORMS].with_root

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._sentence_lens)) if self._shuffle_batches else np.arange(len(self._sentence_lens))

        if emb_calculator:
            psc = None
            if emb_calculator.progressServer:
                psc = progClient.PS_Client("Contextual Embeddings", emb_calculator.progressServer)
            self._embeddings_data = []
            # create contextual embeddings on the fly (specially needed for server mode)
            if isinstance(filename, io.StringIO):
                f = "(StringIO object)"  # filename.getvalue()
            else:
                f = filename

            logger.info("%d sentence(s), calculating embeddings data for '%s'" % (sentences, f))
            ct = 0
            # print(text)
            for conlluline in text:
                ct += 1
                if ct % 7 == 0:
                    print("%3d sentences vectorised" % ct, end="\r")
                    if psc and ct % 28 == 0:
                        psc.update(
                            index=psutil.Process().pid,
                            type=emb_calculator.lg,
                            gpu=emb_calculator.gpu,
                            filename=filename,
                            sentences="%d/%d" % (ct, sentences),
                        )
                emb = emb_calculator.getVectors(conlluline)
                # print(emb, max_sentence_len, len(emb))
                if max_sentence_len:
                    emb = emb[:max_sentence_len]
                # print(np.array(emb).shape)
                # print("=========================")
                #self._embeddings_data.append(np.array(emb)) # expensive !!!!!!
                self._embeddings_data.append(emb) # keep original tf.Tensor
            print("all %d sentences vectorised" % ct)
            #self._embeddings_data_size = self._embeddings_data[0].shape[1] if self._embeddings_data else 0 # if numpy
            self._embeddings_data_size = len(self._embeddings_data[0][0]) if self._embeddings_data else 0 # Tensor from prepareEmbeddings
            if isinstance(filename, io.StringIO):
                f = "(StringIO object)"  # filename.getvalue()
            else:
                f = filename

            logger.info("embeddings data for %3d sentences OK '%s'" % (ct, f))

        if self._embeddings_data:
            logger.info("%d sentences, %d embeddings" % (sentences, len(self._embeddings_data)))
            assert sentences == len(self._embeddings_data)
            # print("check embeddings for %d sentences" % sentences)
            for i in range(sentences):
                assert self._sentence_lens[i] == len(self._embeddings_data[i]), "{} {} {}".format(
                    i, self._sentence_lens[i], len(self._embeddings_data[i])
                )

    @property
    def sentence_lens(self):
        return self._sentence_lens

    @property
    def factors(self):
        return self._factors

    @property
    def embeddings_data_size(self):
        return self._embeddings_data_size

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentence_lens)) if self._shuffle_batches else np.arange(len(self._sentence_lens))
            return True
        return False

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]

        # General data
        batch_sentence_lens = self._sentence_lens[batch_perm]
        max_sentence_len = np.max(batch_sentence_lens)

        # Word-level data
        batch_word_ids = []
        for factor in self._factors:
            #batch_word_ids.append(np.zeros([batch_size, max_sentence_len + factor.with_root], np.int32))
            #for i in range(batch_size):
            #    batch_word_ids[-1][i, 0 : batch_sentence_lens[i] + factor.with_root] = factor.word_ids[batch_perm[i]]

            nouveau = np.zeros([batch_size, max_sentence_len + factor.with_root], np.int32)
            for i in range(batch_size):
                nouveau[i, 0 : batch_sentence_lens[i] + factor.with_root] = factor.word_ids[batch_perm[i]]
            batch_word_ids.append(nouveau)

        # Embeddings
        forms = self._factors[self.FORMS]
        #batch_word_ids.append(np.zeros([batch_size, max_sentence_len + forms.with_root], np.int32))
        nouveau2 = np.zeros([batch_size, max_sentence_len + forms.with_root], np.int32)
        if len(self._embeddings):
            for i in range(batch_size):
                for j, string in enumerate(forms.strings[batch_perm[i]]):
                    nouveau2[i, j] = self._embeddings.get(string, 0) or self._embeddings.get(string.lower(), 0)
        batch_word_ids.append(nouveau2)

        # Contextualized embeddings
        if self._embeddings_data:
            #batch_word_ids.append(np.zeros([batch_size, max_sentence_len + forms.with_root, self.embeddings_data_size], np.float32))
            nouveau3 = np.zeros([batch_size, max_sentence_len + forms.with_root, self.embeddings_data_size], np.float32)
            for i in range(batch_size):
                nouveau3[i, forms.with_root : forms.with_root + len(self._embeddings_data[batch_perm[i]])] = self._embeddings_data[
                    batch_perm[i]
                ]
            batch_word_ids.append(nouveau3)

        # Character-level data
        batch_charseq_ids, batch_charseqs, batch_charseq_lens = [], [], []
        for factor in self._factors:
            if not factor.characters:
                batch_charseq_ids.append([])
                batch_charseqs.append([])
                batch_charseq_lens.append([])
                continue

            batch_charseq_ids.append(np.zeros([batch_size, max_sentence_len + factor.with_root], np.int32))
            charseqs_map = {}
            charseqs = []
            charseq_lens = []
            for i in range(batch_size):
                for j, charseq_id in enumerate(factor.charseq_ids[batch_perm[i]]):
                    if charseq_id not in charseqs_map:
                        charseqs_map[charseq_id] = len(charseqs)
                        charseqs.append(factor.charseqs[charseq_id])
                    batch_charseq_ids[-1][i, j] = charseqs_map[charseq_id]

            batch_charseq_lens.append(np.array([len(charseq) for charseq in charseqs], np.int32))
            batch_charseqs.append(np.zeros([len(charseqs), np.max(batch_charseq_lens[-1])], np.int32))
            for i in range(len(charseqs)):
                batch_charseqs[-1][i, 0 : len(charseqs[i])] = charseqs[i]

        return self._sentence_lens[batch_perm], batch_word_ids, batch_charseq_ids, batch_charseqs, batch_charseq_lens

    def write_sentence(self, output, index, overrides):  # , weights=None, deltas=None, svg=None):
        # print("OVERRIDES", overrides)
        # these weights are useless :-(
        # if weights:
        #    print("# weights = %s" % (weights),  file=output)
        #    print("# weight = %f" % (sum(weights)/len(weights)),  file=output)
        # if deltas:
        #    print("# deltas = %s" % (deltas),  file=output)
        # if svg:
        #    print("# svg = %s" % svg.replace("\n", ""),  file=output)

        for i in range(self._sentence_lens[index] + 1):

            # Start by writing extras
            if index < len(self._extras) and i < len(self._extras[index]) and self._extras[index][i]:
                print(self._extras[index][i], file=output)

            if i == self._sentence_lens[index]:
                break

            fields = []
            fields.append(str(i + 1))
            for f in range(self.FACTORS):
                factor = self._factors[f]
                offset = i + factor.with_root

                field = factor.strings[index][offset]
                # if f == self.DEPREL:
                #    print("qzzz",i,f,factor.words)
                #    print("qaaa",index,field, overrides)

                # Overrides
                if overrides is not None and f < len(overrides) and overrides[f] is not None:
                    # print("rrr",f, self.HEAD, self.LEMMAS, self.DEPREL)
                    if f == self.HEAD:
                        field = str(overrides[f][offset]) if overrides[f][offset] >= 0 else "_"
                    else:
                        field = factor.words[overrides[f][offset]]
                    if f == self.LEMMAS:
                        # print(len(factor.words), overrides[self.LEMMAS], factor.words)
                        if ";" in field:
                            # print("RULE", fields, field)
                            #orig = field
                            field = _apply_lemma_rule(fields[-1], field)
                            #print("correct rule:", fields, orig, field, offset, overrides[f])
                        else:
                            print("missing ; in rule:", fields, field, offset, overrides[f])

                fields.append(field)

            print("\t".join(fields), file=output)
        print(file=output)
