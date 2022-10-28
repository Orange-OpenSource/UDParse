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
#
# Code based on UDPipe Future <http://github.com/CoNLL-UD-2018/UDPipe-Future>.
#
# Copyright 2019 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


from collections import OrderedDict
import datetime
import time
import gzip
import logging
import os
import re
import socket
import sys
import io
import pickle
import warnings

logger = logging.getLogger("udparse")
if logger.getEffectiveLevel() > logging.DEBUG:
    # do not show FutureWarnings from numpy (must be before import numpy)
    warnings.simplefilter(action="ignore", category=FutureWarning)
    # show some deprecation warnings from Tensorflow
    warnings.simplefilter(action="ignore", category=UserWarning)

    # make tensorflow less verbose (ATTENTION: comment in case of searching for problems!!)
    # must be set before loading tensorflow
    # see also below
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np


#  https://www.tensorflow.org/guide/migrate
# must use compat.v1 as the API of tf2 is too different to work here
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa

import dependency_decoding
import UDParse.iwpt20_xud_eval as ud_eval
import UDParse.progClient
import UDParse.ud_dataset

# load further down when we know whether Pytorch or Tensorflow is used for vectorisation
#import UDParse.prepareEmbeddings

# import visualizeInternal


tf.enable_eager_execution()
# tf.disable_v2_behavior() # creates SystemError: <built-in method TFE_Py_TapeWatch of PyCapsule object at 0x7efeb43d7f00> returned a result with an error set


logger.warning("TF Version: %s" % tf.version.VERSION)

# needed to dump validation info
np.set_printoptions(threshold=np.inf)

# make tensorflow less verbose (ATTENTION: comment in case of searching for problems!!)
tf.logging.set_verbosity(tf.logging.ERROR)


logger.warning("TF: Number of available GPUs: %d" % len(tf.config.experimental.list_physical_devices("GPU")))

#from tensorflow.keras import mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)


class Network:
    METRICS = ["UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed

        config = tf.ConfigProto(inter_op_parallelism_threads=threads, intra_op_parallelism_threads=threads, allow_soft_placement=True)
        config.gpu_options.allow_growth = True  # do not reserver the entire GPU memory
        self.session = tf.Session(graph=graph, config=config)


    def construct(self, args, num_words, num_chars, num_tags, num_deprels, predict_only):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.word_ids = tf.placeholder(tf.int32, [None, None])
            self.charseqs = tf.placeholder(tf.int32, [None, None])
            self.charseq_lens = tf.placeholder(tf.int32, [None])
            self.charseq_ids = tf.placeholder(tf.int32, [None, None])

            if args.embeddings_data_size:
                self.embeddings_data = tf.placeholder(tf.float32, [None, None, args.embeddings_data_size])

            self.tags = dict((tag, tf.placeholder(tf.int32, [None, None])) for tag in args.tags)
            self.heads = tf.placeholder(tf.int32, [None, None])
            self.deprels = tf.placeholder(tf.int32, [None, None])
            self.is_training = tf.placeholder(tf.bool, [])
            self.learning_rate = tf.placeholder(tf.float32, [])

            # RNN Cell
            # TODO: use tf.keras.layers instead of tf.nn.rnn_cell (does not work yet with TF 2.4.1/2.5.0 and numpy 1.19/1.20)
            if args.rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell
                # rnn_cell = tf.keras.layers.LSTMCell
            elif args.rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell
                # rnn_cell = tf.keras.layers.GRUCell
            else:
                raise ValueError("Unknown rnn_cell {}".format(args.rnn_cell))

            # Word embeddings
            inputs = []
            if args.we_dim:
                word_embeddings = tf.get_variable("word_embeddings", shape=[num_words, args.we_dim], dtype=tf.float32)
                inputs.append(tf.nn.embedding_lookup(params=word_embeddings, ids=self.word_ids))

            # Character-level embeddings
            character_embeddings = tf.get_variable("character_embeddings", shape=[num_chars, args.cle_dim], dtype=tf.float32)
            characters_embedded = tf.nn.embedding_lookup(params=character_embeddings, ids=self.charseqs)
            # characters_embedded = tf.layers.dropout(characters_embedded, rate=args.dropout, training=self.is_training)
            characters_embedded = tf.keras.layers.Dropout(rate=args.dropout)(characters_embedded, training=self.is_training)
            _, (state_fwd, state_bwd) = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(args.cle_dim),
                # tf.keras.layers.GRUCell(args.cle_dim),
                tf.nn.rnn_cell.GRUCell(args.cle_dim),
                # tf.keras.layers.GRUCell(args.cle_dim),
                characters_embedded,
                sequence_length=self.charseq_lens,
                dtype=tf.float32,
            )
            cle = tf.concat([state_fwd, state_bwd], axis=1)
            cle_inputs = tf.nn.embedding_lookup(params=cle, ids=self.charseq_ids)
            # If CLE dim is half WE dim, we add them together, which gives
            # better results; otherwise we concatenate CLE and WE.
            if 2 * args.cle_dim == args.we_dim:
                inputs[-1] += cle_inputs
            else:
                inputs.append(cle_inputs)

            # Contextualized embeddings
            if args.embeddings_data_size:
                inputs.append(self.embeddings_data)

            # All inputs done
            inputs = tf.concat(inputs, axis=2)

            # Shared RNN layers
            # hidden_layer = tf.layers.dropout(inputs, rate=args.dropout, training=self.is_training)
            hidden_layer = tf.keras.layers.Dropout(rate=args.dropout)(inputs, training=self.is_training)
            for i in range(args.rnn_layers):
                (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                    rnn_cell(args.rnn_cell_dim),
                    rnn_cell(args.rnn_cell_dim),
                    hidden_layer,
                    sequence_length=self.sentence_lens + 1,
                    dtype=tf.float32,
                    scope="word-level-rnn-{}".format(i),
                )
                previous = hidden_layer
                # hidden_layer = tf.layers.dropout(hidden_layer_fwd + hidden_layer_bwd, rate=args.dropout, training=self.is_training)
                hidden_layer = tf.keras.layers.Dropout(rate=args.dropout)(hidden_layer_fwd + hidden_layer_bwd, training=self.is_training)
                if i:
                    hidden_layer += previous

            # Tagger
            # loss = 0
            loss = tf.Variable(0, dtype=tf.float32)
            with tf.GradientTape() as g:
                weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
                weights_sum = tf.reduce_sum(input_tensor=weights)
                self.predictions = {}
                self.prediction_probs = {}
                tag_hidden_layer = hidden_layer[:, 1:]
                for i in range(args.rnn_layers_tagger):
                    (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                        rnn_cell(args.rnn_cell_dim),
                        rnn_cell(args.rnn_cell_dim),
                        tag_hidden_layer,
                        sequence_length=self.sentence_lens,
                        dtype=tf.float32,
                        scope="word-level-rnn-tag-{}".format(i),
                    )
                    previous = tag_hidden_layer
                    # tag_hidden_layer = tf.layers.dropout(hidden_layer_fwd + hidden_layer_bwd, rate=args.dropout, training=self.is_training)
                    tag_hidden_layer = tf.keras.layers.Dropout(rate=args.dropout)(hidden_layer_fwd + hidden_layer_bwd, training=self.is_training)
                    if i:
                        tag_hidden_layer += previous
                for tag in args.tags:
                    tag_layer = tag_hidden_layer
                    for _ in range(args.tag_layers):
                        # tag_layer += tf.layers.dropout(tf.layers.dense(tag_layer, args.rnn_cell_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                        tag_layer += tf.keras.layers.Dropout(rate=args.dropout)(
                            tf.layers.dense(tag_layer, args.rnn_cell_dim, activation=tf.nn.tanh),
                            #tf.keras.layers.Dense(units=args.rnn_cell_dim, activation=tf.nn.tanh)(tag_layer),
                            training=self.is_training
                        )
                    if tag == "LEMMAS":
                        tag_layer = tf.concat([tag_layer, cle_inputs[:, 1:]], axis=2)
                    output_layer = tf.layers.dense(tag_layer, num_tags[tag])
                    #output_layer = tf.keras.layers.Dense(units=num_tags[tag])(tag_layer)
                    self.predictions[tag] = tf.argmax(input=output_layer, axis=2, output_type=tf.int32)
                    self.prediction_probs[tag] = tf.nn.softmax(output_layer, axis=2)

                    if args.label_smoothing:
                        gold_labels = tf.one_hot(self.tags[tag], num_tags[tag]) * (1 - args.label_smoothing) + args.label_smoothing / num_tags[tag]
                        loss = loss + tf.losses.softmax_cross_entropy(gold_labels, output_layer, weights=weights)
                        # loss.assign_add(tf.losses.softmax_cross_entropy(gold_labels, output_layer, weights=weights))
                    else:
                        loss = loss + tf.losses.sparse_softmax_cross_entropy(self.tags[tag], output_layer, weights=weights)
                        # loss.assign_add(tf.losses.sparse_softmax_cross_entropy(self.tags[tag], output_layer, weights=weights))

                # Parsing
                if args.parse:
                    max_words = tf.reduce_max(input_tensor=self.sentence_lens)

                    if args.rnn_layers == 0:
                        parser_inputs = [inputs]
                        for tag in ["UPOS", "XPOS", "FEATS"]:
                            parser_inputs.append(
                                tf.nn.embedding_lookup(
                                    params=tf.get_variable(tag + "_embeddings", shape=[num_tags[tag], 128], dtype=tf.float32),
                                    ids=tf.pad(tensor=self.predictions[tag], paddings=((0, 0), (1, 0)), constant_values=2),
                                )
                            )
                        parser_inputs = tf.concat(parser_inputs, axis=2)
                        # hidden_layer = tf.layers.dropout(parser_inputs, rate=args.dropout, training=self.is_training)
                        hidden_layer = tf.keras.layers.Dropout(rate=args.dropout)(parser_inputs, training=self.is_training)

                    for i in range(args.rnn_layers_parser):
                        (hidden_layer_fwd, hidden_layer_bwd), _ = tf.nn.bidirectional_dynamic_rnn(
                            rnn_cell(args.rnn_cell_dim),
                            rnn_cell(args.rnn_cell_dim),
                            hidden_layer,
                            sequence_length=self.sentence_lens + 1,
                            dtype=tf.float32,
                            scope="word-level-rnn-parser-{}".format(i),
                        )
                        previous = hidden_layer
                        # hidden_layer = tf.layers.dropout(hidden_layer_fwd + hidden_layer_bwd, rate=args.dropout, training=self.is_training)
                        hidden_layer = tf.keras.layers.Dropout(rate=args.dropout)(hidden_layer_fwd + hidden_layer_bwd, training=self.is_training)
                        if i:
                            hidden_layer += previous

                    # Heads
                    head_deps = hidden_layer[:, 1:]
                    for _ in range(args.parser_layers):
                        # head_deps += tf.layers.dropout(tf.layers.dense(head_deps, args.rnn_cell_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                        head_deps += tf.keras.layers.Dropout(rate=args.dropout)(
                            tf.layers.dense(head_deps, args.rnn_cell_dim, activation=tf.nn.tanh),
                            #tf.keras.layers.Dense(units=args.rnn_cell_dim, activation=tf.nn.tanh)(head_deps),
                            training=self.is_training
                        )
                    head_roots = hidden_layer
                    for _ in range(args.parser_layers):
                        # head_roots += tf.layers.dropout(tf.layers.dense(head_roots, args.rnn_cell_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                        head_roots += tf.keras.layers.Dropout(rate=args.dropout)(
                            tf.layers.dense(head_roots, args.rnn_cell_dim, activation=tf.nn.tanh),
                            #tf.keras.layers.Dense(units=args.rnn_cell_dim, activation=tf.nn.tanh)(head_roots),
                            training=self.is_training
                        )

                    head_deps_bias = tf.get_variable("head_deps_bias", [args.rnn_cell_dim], dtype=tf.float32, initializer=tf.zeros_initializer)
                    head_roots_bias = tf.get_variable("head_roots_bias", [args.rnn_cell_dim], dtype=tf.float32, initializer=tf.zeros_initializer)
                    head_biaffine = tf.get_variable(
                        "head_biaffine", [args.rnn_cell_dim, args.rnn_cell_dim], dtype=tf.float32, initializer=tf.zeros_initializer
                    )

                    heads = tf.reshape(
                        tf.matmul(tf.reshape(head_deps, [-1, args.rnn_cell_dim]) + head_deps_bias, head_biaffine),
                        [tf.shape(input=hidden_layer)[0], -1, args.rnn_cell_dim],
                    )
                    heads = tf.matmul(heads, head_roots + head_roots_bias, transpose_b=True)
                    self.heads_logs = tf.nn.log_softmax(heads)
                    self.heads_probs = tf.nn.softmax(heads)
                    if args.label_smoothing:
                        gold_labels = tf.one_hot(self.heads, max_words + 1) * (1 - args.label_smoothing)
                        gold_labels += args.label_smoothing / tf.to_float(max_words + 1)
                        loss = loss + tf.losses.softmax_cross_entropy(gold_labels, heads, weights=weights)
                        # loss.assign_add(tf.losses.softmax_cross_entropy(gold_labels, heads, weights=weights))
                    else:
                        loss = loss + tf.losses.sparse_softmax_cross_entropy(self.heads, heads, weights=weights)
                        # loss.assign_add(tf.losses.sparse_softmax_cross_entropy(self.heads, heads, weights=weights))

                    # Deprels
                    self.deprel_hidden_layer = tf.identity(hidden_layer)
                    self.deprel_heads = tf.identity(self.heads)

                    # deprel_deps = tf.layers.dropout(tf.layers.dense(self.deprel_hidden_layer[:, 1:], args.parser_deprel_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                    deprel_deps = tf.keras.layers.Dropout(rate=args.dropout)(
                        tf.layers.dense(self.deprel_hidden_layer[:, 1:], args.parser_deprel_dim, activation=tf.nn.tanh),
                        #tf.keras.layers.Dense(units=args.parser_deprel_dim, activation=tf.nn.tanh)(self.deprel_hidden_layer[:, 1:]),
                        training=self.is_training
                    )
                    for _ in range(args.parser_layers - 1):
                        # deprel_deps += tf.layers.dropout(tf.layers.dense(deprel_deps, args.parser_deprel_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                        deprel_deps += tf.keras.layers.Dropout(rate=args.dropout)(
                            tf.layers.dense(deprel_deps, args.parser_deprel_dim, activation=tf.nn.tanh),
                            #tf.keras.layers.Dense(units=args.parser_deprel_dim, activation=tf.nn.tanh)(deprel_deps),
                            training=self.is_training
                        )

                    deprel_indices = tf.stack(
                        [
                            tf.tile(
                                tf.expand_dims(tf.range(tf.shape(input=self.deprel_heads)[0]), axis=1),
                                multiples=[1, tf.shape(input=self.deprel_heads)[1]],
                            ),
                            self.deprel_heads,
                        ],
                        axis=2,
                    )
                    deprel_roots = tf.gather_nd(
                        self.deprel_hidden_layer,
                        deprel_indices,
                    )
                    # deprel_roots = tf.layers.dropout(tf.layers.dense(deprel_roots, args.parser_deprel_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                    deprel_roots = tf.keras.layers.Dropout(rate=args.dropout)(
                        tf.layers.dense(deprel_roots, args.parser_deprel_dim, activation=tf.nn.tanh),
                        #tf.keras.layers.Dense(units=args.parser_deprel_dim, activation=tf.nn.tanh)(deprel_roots),
                        training=self.is_training
                    )
                    for _ in range(args.parser_layers - 1):
                        # deprel_roots += tf.layers.dropout(tf.layers.dense(deprel_roots, args.parser_deprel_dim, activation=tf.nn.tanh), rate=args.dropout, training=self.is_training)
                        deprel_roots += tf.keras.layers.Dropout(rate=args.dropout)(
                            tf.layers.dense(deprel_roots, args.parser_deprel_dim, activation=tf.nn.tanh),
                            #tf.keras.layers.Dense(units=args.parser_deprel_dim, activation=tf.nn.tanh)(deprel_roots),
                            training=self.is_training
                        )

                    deprel_deps_bias = tf.get_variable(
                        "deprel_deps_bias", [args.parser_deprel_dim], dtype=tf.float32, initializer=tf.zeros_initializer
                    )
                    deprel_roots_bias = tf.get_variable(
                        "deprel_roots_bias", [args.parser_deprel_dim], dtype=tf.float32, initializer=tf.zeros_initializer
                    )
                    deprel_biaffine = tf.get_variable(
                        "deprel_biaffine",
                        [args.parser_deprel_dim, num_deprels * args.parser_deprel_dim],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer,
                    )

                    deprels = tf.reshape(
                        tf.matmul(tf.reshape(deprel_deps, [-1, args.parser_deprel_dim]) + deprel_deps_bias, deprel_biaffine),
                        [tf.shape(input=self.deprel_hidden_layer)[0], -1, num_deprels, args.parser_deprel_dim],
                    )
                    deprels = tf.squeeze(tf.matmul(deprels, tf.expand_dims(deprel_roots + deprel_roots_bias, axis=3)), axis=3)
                    self.predictions_deprel = tf.argmax(input=deprels, axis=2, output_type=tf.int32)
                    self.predictions_deprel_probs = tf.nn.softmax(deprels, axis=2)
                    if args.label_smoothing:
                        gold_labels = tf.one_hot(self.deprels, num_deprels) * (1 - args.label_smoothing)
                        gold_labels += args.label_smoothing / num_deprels
                        loss = loss + tf.losses.softmax_cross_entropy(gold_labels, deprels, weights=weights)
                        # loss.assign_add(tf.losses.softmax_cross_entropy(gold_labels, deprels, weights=weights))
                    else:
                        loss = loss + tf.losses.sparse_softmax_cross_entropy(self.deprels, deprels, weights=weights)
                        # loss.assign_add(tf.losses.sparse_softmax_cross_entropy(self.deprels, deprels, weights=weights))

                # Pretrain saver
                self.saver_inference = tf.train.Saver(max_to_keep=1)
                if predict_only:
                    return

                # Training
                self.global_step = tf.train.create_global_step()
                # self.training = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate, beta2=args.beta_2).minimize(loss, global_step=self.global_step)

                self.training = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta2=args.beta_2).minimize(
                    loss, var_list=tf.trainable_variables(), global_step=self.global_step
                )  # , global_step=self.global_step)
                # self.training = tfa.optimizers.LazyAdam(learning_rate=self.learning_rate, beta_2=args.beta_2
            #                                        ).minimize(getLoss,
            #                                                   var_list=tf.trainable_variables()) #, global_step=self.global_step)

            # Train saver
            self.saver_train = tf.train.Saver(max_to_keep=2)

            # Summaries
            # ud_parser.py:275:46: ERROR: tf.contrib.summary.record_summaries_every_n_global_steps requires manual check. (Manual edit required)
            # tf.contrib.summary.record_summaries_every_n_global_steps(n, step) should be replaced by a call to
            # --> tf.compat.v2.summary.record_if() with the argument `lambda: tf.math.equal(0, global_step % n)` (or in graph mode, the lambda body can be used directly).
            # If no global step was passed, instead use tf.train.get_or_create_global_step().
            #            summary_writer = tf.compat.v2.summary.create_file_writer(logdir=args.logdir, flush_millis=10 * 1000)
            #            self.summaries = {}
            #            #with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):"
            #            with summary_writer.as_default(), tf.compat.v2.summary.record_if(lambda: tf.math.equal(0, tf.train.get_or_create_global_step())):
            #                self.summaries["train"] = [
            #                    tf.compat.v2.summary.scalar(name="train/loss", data=loss, step=tf.train.get_or_create_global_step()),
            #                    tf.compat.v2.summary.scalar(name="train/lr", data=self.learning_rate, step=tf.train.get_or_create_global_step())]
            #                for tag in args.tags:
            #                    self.summaries["train"].append(tf.compat.v2.summary.scalar(
            #                        name="train/{}".format(tag),
            #                        data=tf.reduce_sum(input_tensor=tf.cast(tf.equal(self.tags[tag], self.predictions[tag]), tf.float32) * weights) /
            #                        weights_sum, step=tf.train.get_or_create_global_step()))
            #                if args.parse:
            #                    heads_acc = tf.reduce_sum(input_tensor=tf.cast(tf.equal(self.heads, tf.argmax(input=heads, axis=-1, output_type=tf.int32)),
            #                                                      tf.float32) * weights) / weights_sum
            #                    self.summaries["train"].extend([tf.compat.v2.summary.scalar(name="train/heads_acc", data=heads_acc, step=tf.train.get_or_create_global_step())])
            #                    deprels_acc = tf.reduce_sum(input_tensor=tf.cast(tf.equal(self.deprels, tf.argmax(input=deprels, axis=-1, output_type=tf.int32)),
            #                                                        tf.float32) * weights) / weights_sum
            #                    self.summaries["train"].extend([tf.compat.v2.summary.scalar(name="train/deprels_acc", data=deprels_acc, step=tf.train.get_or_create_global_step())])

            #            with summary_writer.as_default(), tf.compat.v2.summary.record_if(True):
            if True:
                self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=weights_sum)
                self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
                self.metrics = dict((metric, tf.placeholder(tf.float32, [])) for metric in self.METRICS)
            #                for dataset in ["dev", "test"]:
            #                    self.summaries[dataset] = [tf.compat.v2.summary.scalar(name=dataset + "/loss", data=self.current_loss, step=tf.train.get_or_create_global_step())]
            #                    for metric in self.METRICS:
            #                        self.summaries[dataset].append(tf.compat.v2.summary.scalar(name="{}/{}".format(dataset, metric),
            #                                                  data=self.metrics[metric], step=tf.train.get_or_create_global_step()))

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

    #            with summary_writer.as_default():
    #                # creates args.logdir
    #                tf.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, learning_rate, args):
        batches, at_least_one_epoch = 0, False
        while batches < args.min_epoch_batches:
            while not train.epoch_finished():
                sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = train.next_batch(args.batch_size)
                if args.word_dropout:
                    mask = np.random.binomial(n=1, p=args.word_dropout, size=word_ids[train.FORMS].shape)
                    word_ids[train.FORMS] = (1 - mask) * word_ids[train.FORMS] + mask * train.factors[train.FORMS].words_map["<unk>"]
                if args.char_dropout:
                    mask = np.random.binomial(n=1, p=args.char_dropout, size=charseqs[train.FORMS].shape)
                    charseqs[train.FORMS] = (1 - mask) * charseqs[train.FORMS] + mask * train.factors[train.FORMS].alphabet_map["<unk>"]

                feeds = {
                    self.is_training: True,
                    self.learning_rate: learning_rate,
                    self.sentence_lens: sentence_lens,
                    self.charseqs: charseqs[train.FORMS],
                    self.charseq_lens: charseq_lens[train.FORMS],
                    self.word_ids: word_ids[train.FORMS],
                    self.charseq_ids: charseq_ids[train.FORMS],
                }
                # if args.embeddings:
                #     if args.word_dropout:
                #         mask = np.random.binomial(n=1, p=args.word_dropout, size=word_ids[train.EMBEDDINGS].shape)
                #         word_ids[train.EMBEDDINGS] = (1 - mask) * word_ids[train.EMBEDDINGS]
                #     embeddings = np.zeros([word_ids[train.EMBEDDINGS].shape[0], word_ids[train.EMBEDDINGS].shape[1], args.embeddings_size])
                #     for i in range(embeddings.shape[0]):
                #         for j in range(embeddings.shape[1]):
                #             if word_ids[train.EMBEDDINGS][i, j]:
                #                 embeddings[i, j] = args.embeddings_data[word_ids[train.EMBEDDINGS][i, j] - 1]
                #     feeds[self.embeddings] = embeddings
                if args.embeddings_data_size:
                    feeds[self.embeddings_data] = word_ids[train.ELMO]
                for tag in args.tags:
                    feeds[self.tags[tag]] = word_ids[train.FACTORS_MAP[tag]]
                if args.parse:
                    feeds[self.heads] = word_ids[train.HEAD]
                    feeds[self.deprels] = word_ids[train.DEPREL]
                self.session.run([self.training], feeds)  # , self.summaries["train"]
                batches += 1
                if at_least_one_epoch:
                    break
            at_least_one_epoch = True

    def predict(self, dataset, evaluating, args, train, attachments=None, epoch="?"):
        debug = False
        if epoch != "?":
            # only in train epoch has a value >= 0. In this case we log some stuff
            debug = True
            ofpfeeds = gzip.open("%s/epoch-%02d-feeds.txt.gz" % (args.logdir, epoch), "wt")
            ofppred = gzip.open("%s/epoch-%02d-predictions.txt.gz" % (args.logdir, epoch), "wt")
            ofpdevout = gzip.open("%s/epoch-%02d-dev.txt.gz" % (args.logdir, epoch), "wt")

            if epoch == 0:
                ofpf = gzip.open("%s/epoch-%02d-factors.txt.gz" % (args.logdir, epoch), "wt")
                inv = {}
                for k in UDParse.ud_dataset.UDDataset.FACTORS_MAP:
                    inv[UDParse.ud_dataset.UDDataset.FACTORS_MAP[k]] = k
                for i, f in enumerate(dataset._factors):
                    print("Factors:", inv.get(i), file=ofpf)
                    print(f, file=ofpf)
                ofpf.close()

        conllu = io.StringIO()
        sentences = 0

        if evaluating:
            self.session.run(self.reset_metrics)
        ct = -1
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(args.batch_size)
            ct += 1
            feeds = {
                self.is_training: False,
                self.sentence_lens: sentence_lens,
                self.charseqs: charseqs[train.FORMS],
                self.charseq_lens: charseq_lens[train.FORMS],
                self.word_ids: word_ids[train.FORMS],
                self.charseq_ids: charseq_ids[train.FORMS],
            }
            #if args.embeddings:
            #    embeddings = np.zeros([word_ids[train.EMBEDDINGS].shape[0], word_ids[train.EMBEDDINGS].shape[1], args.embeddings_size])
            #    for i in range(embeddings.shape[0]):
            #        for j in range(embeddings.shape[1]):
            #            if word_ids[train.EMBEDDINGS][i, j]:
            #                embeddings[i, j] = args.embeddings_data[word_ids[train.EMBEDDINGS][i, j] - 1]
            #    feeds[self.embeddings] = embeddings
            if args.embeddings_data_size:
                feeds[self.embeddings_data] = word_ids[train.ELMO]
            if evaluating:
                for tag in args.tags:
                    feeds[self.tags[tag]] = word_ids[train.FACTORS_MAP[tag]]
                if args.parse:
                    feeds[self.heads] = word_ids[train.HEAD]
                    feeds[self.deprels] = word_ids[train.DEPREL]

            targets = [self.predictions]
            # print("AAA", targets)

            if args.parse:
                targets.extend([self.heads_logs, self.deprel_hidden_layer])
            if evaluating:
                targets.append(self.update_loss)
            # print("BBB", targets)
            if debug:
                for t in "LEMMAS", "UPOS":
                    print(t, ct, feeds[self.tags[t]].shape, file=ofpfeeds)
                    print(feeds[self.tags[t]], file=ofpfeeds)

            predictions, *other_values = self.session.run(targets, feeds)

            # print("YYY", predictions) # contains predicted TAG, Lemma, Feats
            if args.parse:
                prior_heads, deprel_hidden_layer, *_ = other_values
                # print("XXX2", deprel_hidden_layer)
                # print("ZZZZ", other_values)

                heads = np.zeros(prior_heads.shape[:2], dtype=np.int32)
                # these weights are useless
                # allweights = []
                # alldeltas = []
                for i in range(len(sentence_lens)):
                    padded_heads = np.pad(
                        prior_heads[i][: sentence_lens[i], : sentence_lens[i] + 1].astype(np.float), ((1, 0), (0, 0)), mode="constant"
                    )
                    # print("=====================================", padded_heads, sep="\n")
                    padded_heads[:, 0] = np.nan
                    # we put 0 to the root (highest weight)
                    padded_heads[1 + np.argmax(prior_heads[i][: sentence_lens[i], 0]), 0] = 0
                    # padded_heads[-1][-2] = 1.0e-4
                    # padded_heads[3][5] = 0.0
                    # padded_heads[1][3] = 0.0
                    # print(" ====================================", padded_heads, sep="\n")
                    if attachments:
                        # increase weights of attachments we want to keep
                        for word, head in attachments.items():
                            padded_heads[word][head] = 0.0
                        # print(" ====================================", padded_heads, sep="\n")
                    # JHE here the graph is analysed (using the Chu-Liu-Edmonds algo) to get a tree

                    chosen_heads, _ = dependency_decoding.chu_liu_edmonds(padded_heads)
                    # print(">>===================================", chosen_heads, sep="\n")

                    # get weights for each attachment

                    word = -1
                    # weights = [] # weight for each word
                    # deltas = [] # delta to second best weight
                    # for h in chosen_heads:
                    #    word += 1
                    #    if h == -1:
                    #        # ignore "root" which has no head
                    #        continue
                    #    else:
                    #        #print(padded_heads[word][h])
                    #        best = padded_heads[word][h]
                    #        weights.append(best)

                    #        # find second best head
                    #        ordered = sorted(padded_heads[word])
                    #        if ordered[-1] == best:
                    #            second = ordered[-2]
                    #        else:
                    #            second = ordered[-1]
                    #        delta = best - second
                    #        deltas.append(delta)
                    #        #print("rrrr", delta, best, second)

                    # allweights.append(weights)
                    # alldeltas.append(deltas)
                    # svg = None
                    # if args.heatmap:
                    #    #or d in dataset._factors:
                    #    #   print(d.strings)
                    #    vd = visualizeInternal.VisuDep()
                    #    svg = vd.heatmap(padded_heads,
                    #                     chosen_heads,
                    #                     dataset._factors[ud_dataset.UDDataset.FORMS].strings[i])

                    heads[i, : sentence_lens[i]] = chosen_heads[1:]

                # predict deprels from unlabelled tree
                deprels = self.session.run(
                    self.predictions_deprel, {self.is_training: False, self.deprel_hidden_layer: deprel_hidden_layer, self.deprel_heads: heads}
                )


            for i in range(len(sentence_lens)):
                overrides = [None] * dataset.FACTORS
                for tag in args.tags:
                    overrides[dataset.FACTORS_MAP[tag]] = predictions[tag][i]

                    if debug and tag == "LEMMAS":
                        print(ct, i, (ct * args.batch_size) + i, predictions[tag][i], file=ofppred)

                if args.parse:
                    overrides[dataset.HEAD] = heads[i]
                    overrides[dataset.DEPREL] = deprels[i]
                dataset.write_sentence(conllu, sentences, overrides)  # , allweights[i], alldeltas[i], svg)
                sentences += 1

        if debug:
            print(conllu.getvalue(), file=ofpdevout)
            ofpdevout.close()
            ofppred.close()
            ofpfeeds.close()
        return conllu.getvalue()

    # write debug info if epoch != "?"
    def evaluate(self, dataset_name, dataset, dataset_conllu, args, train, epoch="?"):
        import io

        conllu = self.predict(dataset, True, args, train, epoch=epoch)
        # conll18_eval.py
        metrics = ud_eval.evaluate(dataset_conllu, ud_eval.load_conllu(io.StringIO(conllu)))

        # iwpt20_eval.py
        # metrics = ud_eval.evaluate(dataset_conllu,
        #                           ud_eval.load_conllu(io.StringIO(conllu), {}))

        # self.session.runself.summaries[dataset_name],
        #                 dict((self.metrics[metric], tf.Variable(metrics[metric].f1, dtype=tf.float32)) for metric in self.METRICS))

        if args.parse:
            return (metrics["LAS"].f1 + metrics["MLAS"].f1 + metrics["BLEX"].f1) / 3.0, metrics
        else:
            return metrics["AllTags"].f1, metrics


def train(args):
    command_line = " ".join(sys.argv)

    # Fix random seed
    np.random.seed(args.seed)

    if args.exp is None:
        args.exp = "{}-{}".format(os.path.basename(__file__), datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

    if args.logdir is None:
        # Create logdir name
        do_not_log = {"exp", "max_sentence_len", "predict", "predict_input", "predict_output", "tags", "threads"}
        args.logdir = "logs/{}-{}".format(
            args.exp,
            ",".join(
                (
                    "{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), re.sub("^.*/", "", value) if type(value) == str else value)
                    for key, value in sorted(vars(args).items())
                    if key not in do_not_log
                )
            ),
        )
        if not args.predict and not os.path.exists("logs"):
            os.mkdir("logs")  # TF 1.6 will do this by itself

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    # Postprocess args
    args.tags = args.tags.split(",")
    args.epochs = [(int(epochs), float(lr)) for epochs, lr in (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]


    root_factors = [UDParse.ud_dataset.UDDataset.FORMS]

    calc = None
    if args.calculate_embeddings:
        if args.forceframework == "PT":
            import UDParse.prepareEmbeddingsPT as prepareEmbeddings
        else:
            import UDParse.prepareEmbeddings as prepareEmbeddings
        calc = prepareEmbeddings.Embeddings(
            lg=args.calculate_embeddings,
            progServer=args.progressServer,
            gpu=args.gpu,
        )



    train = UDParse.ud_dataset.UDDataset(
        args.train_input,  # .split(","), #"{}-ud-train.conllu".format(args.basename),
        root_factors,
        max_sentence_len=args.max_sentence_len,
        embeddings_data=re.sub("(?=,|$)", "-train%s.npz" % args.emb_suffix, args.embeddings_data) if args.embeddings_data else None,
        progserver=args.progressServer,
    )

    # save train data, needed for prediction
    logger.info("save training data references '%s/train_pickle.gz'" % args.logdir)
    with gzip.GzipFile("%s/train_pickle.gz" % (args.logdir), "w") as f:
        # train._embeddings_data contains the embeddings loaded for the train/dev set
        # we do not need them after training, so we do not write them to disk
        # but keep them for afterwards
        keep_embeddings_datas = train._embeddings_data
        del train._embeddings_data
        pickle.dump(train, f)
        # put them agin into _embeddings_data
        train._embeddings_data = keep_embeddings_datas
        #sys.exit(0) # stop here only to rewrite train_pickle when the name of ud_dataset has changed

    # if os.path.exists("{}-ud-dev.conllu".format(args.basename)):
    if args.dev_input:  # and os.path.exists(args.dev_input):
        for d in args.dev_input:
            if not os.path.exists(d):
                logger.error("dev file %s does not exist!" % d)
                break
        else:
            dev = UDParse.ud_dataset.UDDataset(
                args.dev_input,  # "{}-ud-dev.conllu".format(args.basename),
                root_factors,
                train=train,
                shuffle_batches=False,
                embeddings_data=re.sub("(?=,|$)", "-dev%s.npz" % args.emb_suffix, args.embeddings_data) if args.embeddings_data else None,
            )
    else:
        dev = None

    if args.test_input:  # and os.path.exists(args.test_input):
        for d in args.test_input:
            if not os.path.exists(d):
                logger.error("test file %s does not exist!" % d)
                break
        else:
            test = UDParse.ud_dataset.UDDataset(
                args.test_input,  # "{}-ud-test.conllu".format(args.basename),
                root_factors,
                train=train,
                shuffle_batches=False,
                embeddings_data=re.sub("(?=,|$)", "-test%s.npz" % args.emb_suffix, args.embeddings_data) if args.embeddings_data else None,
            )
    else:
        test = None

    if test:
        args.embeddings_data_size = test.embeddings_data_size
    elif dev:
        args.embeddings_data_size = train.embeddings_data_size
    elif train:
        args.embeddings_data_size = train.embeddings_data_size


    # Construct the network
    network = Network(threads=args.threads, seed=args.seed)
    network.construct(
        args,
        len(train.factors[train.FORMS].words),
        len(train.factors[train.FORMS].alphabet),
        dict((tag, len(train.factors[train.FACTORS_MAP[tag]].words)) for tag in args.tags),
        len(train.factors[train.DEPREL].words),
        predict_only=args.predict,
    )


    if args.calculate_embeddings:
        if args.forceframework == "PT":
            import UDParse.prepareEmbeddingsPT as prepareEmbeddings
        else:
            import UDParse.prepareEmbeddings as prepareEmbeddings

        calc = prepareEmbeddings.Embeddings(
            lg=args.calculate_embeddings,
            progServer=args.progressServer,
            gpu=args.gpu,
        )

    if args.checkpoint:
        with network.session.graph.as_default():
            saver = network.saver_train if args.predict is None else network.saver_inference
            saver.restore(network.session, args.checkpoint)


  

    # we pass here only in training
    with open("{}/cmd".format(args.logdir), "w") as cmd_file:
        cmd_file.write(command_line)

    log_file = open("{}/log".format(args.logdir), "w")
    for tag in args.tags + ["DEPREL"]:
        print("{}: {}".format(tag, len(train.factors[train.FACTORS_MAP[tag]].words)), file=log_file, flush=True)

    logger.warning(
        "Parsing with args:\n\t" + "\n\t".join(("{}: {}".format(key, value) for key, value in sorted(vars(args).items()) if key not in ["embeddings_data", "embeddings_words"]))
    )

    if dev:
        dev_conllu = ud_eval.load_conllu(args.dev_input)  # "{}-ud-dev.conllu".format(args.basename))

    if test:
        test_conllu = ud_eval.load_conllu(args.test_input)  # "{}-ud-test.conllu".format(args.basename))

    starttime = last = datetime.datetime.now()
    print("Started training: {}".format(last.strftime("%d/%m/%Y %H:%M:%S")), file=log_file, flush=True)

    progclient = None
    if args.progressServer != None:
        progclient = UDParse.progClient.PS_Client("UdpipeFuture", args.progressServer)
        plist = []
        for metric in Network.METRICS:
            if metric not in ["qqMLAS", "BLEX"]:
                plist.append((metric, ""))

        progclient.update(
            index="%s:%s" % (args.logdir.split("/")[-1], args.gpu),
            started=starttime.strftime("%d/%m/%Y %H:%M:%S"),
            ETA="",
            gpu="%s:%s" % (socket.gethostname(), args.gpu),
            epoch=0,
            last_epoch="",
            listtype=plist,
            logdir="%s:%s" % (socket.gethostname(), args.logdir),
        )
        progclient.setlimits("epoch", 0, 60)
        progclient.setlimits("LAS", 0, 100, inverse=True)

    all_epochs = args.epochs[0][0] + args.epochs[1][0]
    ctepoch = -1
    for i, (epochs, learning_rate) in enumerate(args.epochs):
        for epoch in range(epochs):
            network.train_epoch(train, learning_rate, args)

            now = datetime.datetime.now()
            delta = str(now - last).split(".")[0]  # split decimals
            last = now
            ctepoch += 1
            reste_epochs = all_epochs - (ctepoch + 1)
            total_time = now - starttime
            reste_time = (total_time / (ctepoch + 1)) * reste_epochs
            reste_time = str(reste_time).split(".")[0]  # split decimals
            # print(reste_epochs, total_time, epoch+1, reste_time)
            # print("now =", now)
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            if not dev and not test:
                print("{} {} ETA: {} epoch {:2d}, lr {}".format(dt_string, delta, reste_time, ctepoch + 1, learning_rate), file=log_file, flush=True)

            if dev:
                # DEBUG epoch=epoch, else epoch ="?"
                dev_accuracy, metrics = network.evaluate("dev", dev, dev_conllu, args, train, epoch=ctepoch)
                usemetrics = []
                for x in Network.METRICS:
                    if x != "qqMLAS" and x != "BLEX":
                        usemetrics.append(x)

                metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric].f1) for metric in usemetrics))
                print(
                    "Dev,  {} {} ETA: {} epoch {:2d}, lr {}, {}".format(dt_string, delta, reste_time, ctepoch + 1, learning_rate, metrics_log),
                    file=log_file,
                    flush=True,
                )

                if progclient:
                    plist = []
                    for metric in Network.METRICS:
                        if metric not in ["qqMLAS", "BLEX"]:
                            plist.append((metric, "%.2f" % (100 * metrics[metric].f1)))

                    progclient.update(
                        index="%s:%s" % (args.logdir.split("/")[-1], args.gpu),
                        started=starttime.strftime("%d/%m/%Y %H:%M:%S"),
                        ETA=reste_time,
                        gpu="%s:%s" % (socket.gethostname(), args.gpu),
                        epoch=ctepoch + 1,
                        last_epoch=delta,
                        listtype=plist,
                        logdir="%s:%s" % (socket.gethostname(), args.logdir),
                    )

            else:
                if progclient:
                    progclient.update(
                        index="%s:%s" % (args.logdir.split("/")[-1], args.gpu),
                        started=starttime.strftime("%d/%m/%Y %H:%M:%S"),
                        ETA=reste_time,
                        gpu="%s:%s" % (socket.gethostname(), args.gpu),
                        epoch=ctepoch + 1,
                        last_epoch=delta,
                        logdir="%s:%s" % (socket.gethostname(), args.logdir),
                    )

            if test:
                test_accuracy, metrics = network.evaluate("test", test, test_conllu, args, train)
                metrics_log = ", ".join(("{}: {:.2f}".format(metric, 100 * metrics[metric].f1) for metric in usemetrics))
                print(
                    "Test, {} {} ETA: {} epoch {:2d}, lr {}, {}".format(dt_string, delta, reste_time, ctepoch + 1, learning_rate, metrics_log),
                    file=log_file,
                    flush=True,
                )

            # save checkpoints after each epoch
            with network.session.graph.as_default():
                network.saver_train.save(
                    network.session, "{}/checkpoint".format(args.logdir), global_step=network.global_step, write_meta_graph=False
                )

    if progclient:
        progclient.delete()

    with network.session.graph.as_default():
        network.saver_inference.save(network.session, "{}/checkpoint-inference-last".format(args.logdir), write_meta_graph=False)


def predict(args):
    command_line = " ".join(sys.argv)

    # Fix random seed
    np.random.seed(args.seed)

    if args.exp is None:
        args.exp = "{}-{}".format(os.path.basename(__file__), datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

    # Postprocess args
    args.tags = args.tags.split(",")
    #args.epochs = [(int(epochs), float(lr)) for epochs, lr in (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]


    root_factors = [UDParse.ud_dataset.UDDataset.FORMS]

    logger.info("Reading %s/train_pickle.gz" % (args.logdir) )
    with gzip.GzipFile("%s/train_pickle.gz" % (args.logdir)) as f:
        train = pickle.load(f)
        args.embeddings_data_size = train.embeddings_data_size


    # Construct the network
    network = Network(threads=args.threads, seed=args.seed)
    network.construct(
        args, # including args.embeddings_data_size
        len(train.factors[train.FORMS].words),
        len(train.factors[train.FORMS].alphabet),
        dict((tag, len(train.factors[train.FACTORS_MAP[tag]].words)) for tag in args.tags),
        len(train.factors[train.DEPREL].words),
        predict_only=args.predict,
    )

    calc = None
    if args.calculate_embeddings:
        if args.forceframework == "PT":
            import UDParse.prepareEmbeddingsPT as prepareEmbeddings
        else:
            import UDParse.prepareEmbeddings as prepareEmbeddings

        calc = prepareEmbeddings.Embeddings(
            lg=args.calculate_embeddings,
            progServer=args.progressServer,
            gpu=args.gpu,
        )

    if not args.server:
        # just one file to predict
        #aa = time.time()
        #print("START vec", aa)
        test = UDParse.ud_dataset.UDDataset(
            args.predict_input,
            root_factors,
            train=train,
            shuffle_batches=False,
            embeddings_data=re.sub("(?=,|$)", "-test%s.npz" % args.emb_suffix, args.embeddings_data) if args.embeddings_data else None,
            progserver=args.progressServer,
            emb_calculator=calc,
        )
        #bb = time.time()
        #print("END vec", bb)
    if not args.checkpoint:
        # default checkpoint in data.yml out-directory
        args.checkpoint = "%s/checkpoint-inference-last" % args.logdir



    if args.checkpoint:
        with network.session.graph.as_default():
            saver = network.saver_train if args.predict is None else network.saver_inference
            saver.restore(network.session, args.checkpoint)


    if args.server:
        # server mode: return stuff needed later
        return root_factors, train, network, calc


    if args.predict:
        # just predicting a file (test-file or another file)
        #cc = time.time()
        #print("START predict", cc)
        conllu = network.predict(test, False, args, train)
        print(conllu, end="", file=open(args.predict_output, "w", encoding="utf-8") if args.predict_output else sys.stdout)
        #dd = time.time()
        #print("END predict", dd)

        #print("VECTORISATION_PARSING:vectime:parsetime", bb-aa, dd-cc)
        return




