# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains code to read tokenized data from file,
truncate, pad and process it into batches ready for training"""

from __future__ import absolute_import
from __future__ import division

import random
import time
import re

import numpy as np
from six.moves import xrange
from vocab import PAD_ID, UNK_ID


class Batch(object):
    """A class to hold the information needed for a training batch"""

    def __init__(self, context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens, ans_span, ans_tokens, char_context_ids, char_qn_ids, char_context_mask, char_qn_mask, uuids=None):
        """
        Inputs:
          {context/qn}_ids: Numpy arrays.
            Shape (batch_size, {context_len/question_len}). Contains padding.
          {context/qn}_mask: Numpy arrays, same shape as _ids.
            Contains 1s where there is real data, 0s where there is padding.
          {context/qn/ans}_tokens: Lists length batch_size, containing lists (unpadded) of tokens (strings)
          ans_span: numpy array, shape (batch_size, 2)
          uuid: a list (length batch_size) of strings.
            Not needed for training. Used by official_eval mode.
        """
        self.context_ids = context_ids
        self.context_mask = context_mask
        self.context_tokens = context_tokens

        self.qn_ids = qn_ids
        self.qn_mask = qn_mask
        self.qn_tokens = qn_tokens

        self.ans_span = ans_span
        self.ans_tokens = ans_tokens

        self.uuids = uuids

        self.batch_size = len(self.context_tokens)

        # added for character embeddings
        self.char_context_ids = char_context_ids
        self.char_qn_ids = char_qn_ids
        self.char_context_mask = char_context_mask
        self.char_qn_mask = char_qn_mask


def split_by_whitespace(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def intstr_to_intlist(string):
    """Given a string e.g. '311 9 1334 635 6192 56 639', returns as a list of integers"""
    return [int(s) for s in string.split()]


def sentence_to_token_ids(sentence, word2id):
    """Turns an already-tokenized sentence string into word indices
    e.g. "i do n't know" -> [9, 32, 16, 96]
    Note any token that isn't in the word2id mapping gets mapped to the id for UNK
    """
    tokens = split_by_whitespace(sentence) # list of strings
    ids = [word2id.get(w, UNK_ID) for w in tokens]
    return tokens, ids

def words_to_char_ids(words, char2id):
    """Returns a list of lists of character IDs for a given list of words."""
    ids = [[char2id.get(c, UNK_ID) for c in word] for word in words]
    return ids

def padded(token_batch, batch_pad=0):
    """
    Inputs:
      token_batch: List (length batch size) of lists of ints.
      batch_pad: Int. Length to pad to. If 0, pad to maximum length sequence in token_batch.
    Returns:
      List (length batch_size) of padded of lists of ints.
        All are same length - batch_pad if batch_pad!=0, otherwise the maximum length in token_batch
    """
    maxlen = max(map(lambda x: len(x), token_batch)) if batch_pad == 0 else batch_pad
    return map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), token_batch)

def pad_chars(char_ids, word_len, num_words):
    """
    Inputs:
      char_ids: List (length batch size) of lists (length context_len or question_len) of lists of ints
      word_len: Int. Length to pad to.
    Returns:
      List (length batch size) of padded lists (length context_len or question_len) of padded lists of ints
    """
    padded_words = [[word + [PAD_ID]*(word_len - len(word)) for word in batch] for batch in char_ids]
    PAD_WORD = [PAD_ID]*word_len
    padded_batches = [batch + [PAD_WORD]*(num_words - len(batch)) for batch in padded_words]
    return padded_batches

def refill_batches(batches, word2id, char2id, context_file, qn_file, ans_file, batch_size, context_len, question_len, word_len, discard_long):
    """
    Adds more batches into the "batches" list.

    Inputs:
      batches: list to add batches to
      word2id: dictionary mapping word (string) to word id (int)
      char2id: dictionary mapping character (string) to character id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.
    """
    print "Refilling batches..."
    tic = time.time()
    examples = [] # list of (qn_ids, context_ids, ans_span, ans_tokens) triples
    context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline() # read the next line from each

    while context_line and qn_line and ans_line: # while you haven't reached the end

        # Convert tokens to word ids
        context_tokens, context_ids = sentence_to_token_ids(context_line, word2id)
        qn_tokens, qn_ids = sentence_to_token_ids(qn_line, word2id)
        ans_span = intstr_to_intlist(ans_line)

        # Convert word tokens to lists of char ids
        char_context_ids = words_to_char_ids(context_tokens, char2id)
        char_qn_ids = words_to_char_ids(qn_tokens, char2id)

        # read the next line from each file
        context_line, qn_line, ans_line = context_file.readline(), qn_file.readline(), ans_file.readline()

        # get ans_tokens from ans_span
        assert len(ans_span) == 2
        if ans_span[1] < ans_span[0]:
            print "Found an ill-formed gold span: start=%i end=%i" % (ans_span[0], ans_span[1])
            continue
        ans_tokens = context_tokens[ans_span[0] : ans_span[1]+1] # list of strings

        # discard or truncate too-long questions
        if len(qn_ids) > question_len:
            if discard_long:
                continue
            else: # truncate
                qn_ids = qn_ids[:question_len]

        # discard or truncate too-long contexts
        if len(context_ids) > context_len:
            if discard_long:
                continue
            else: # truncate
                context_ids = context_ids[:context_len]

        # truncate too-long words
        char_context_ids = [word[:word_len] for word in char_context_ids]
        char_qn_ids = [word[:word_len] for word in char_qn_ids]

        # add to examples
        examples.append((context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens, char_context_ids, char_qn_ids))

        # stop refilling if you have 160 batches
        if len(examples) == batch_size * 160:
            break

    # Once you've either got 160 batches or you've reached end of file:

    # Sort by question length
    # Note: if you sort by context length, then you'll have batches which contain the same context many times (because each context appears several times, with different questions)
    examples = sorted(examples, key=lambda e: len(e[2]))

    # Make into batches and append to the list batches
    for batch_start in xrange(0, len(examples), batch_size):

        # Note: each of these is a list length batch_size of lists of ints (except on last iter when it might be less than batch_size)
        context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch, char_context_ids_batch, char_qn_ids_batch = zip(*examples[batch_start:batch_start+batch_size])

        batches.append((context_ids_batch, context_tokens_batch, qn_ids_batch, qn_tokens_batch, ans_span_batch, ans_tokens_batch, char_context_ids_batch, char_qn_ids_batch))

    # shuffle the batches
    random.shuffle(batches)

    toc = time.time()
    print "Refilling batches took %.2f seconds" % (toc-tic)
    return


def get_batch_generator(word2id, char2id, context_path, qn_path, ans_path, batch_size, context_len, question_len, word_len, discard_long):
    """
    This function returns a generator object that yields batches.
    The last batch in the dataset will be a partial batch.
    Read this to understand generators and the yield keyword in Python: https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do

    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      context_file, qn_file, ans_file: paths to {train/dev}.{context/question/answer} data files
      batch_size: int. how big to make the batches
      context_len, question_len: max length of context and question respectively
      discard_long: If True, discard any examples that are longer than context_len or question_len.
        If False, truncate those exmaples instead.
    """
    context_file, qn_file, ans_file = open(context_path), open(qn_path), open(ans_path)
    batches = []

    while True:
        if len(batches) == 0: # add more batches
            refill_batches(batches, word2id, char2id, context_file, qn_file, ans_file, batch_size, context_len, question_len, word_len, discard_long)
        if len(batches) == 0:
            break

        # Get next batch. These are all lists length batch_size
        (context_ids, context_tokens, qn_ids, qn_tokens, ans_span, ans_tokens, char_context_ids, char_qn_ids) = batches.pop(0)
 
        # Pad context_ids and qn_ids
        qn_ids = padded(qn_ids, question_len) # pad questions to length question_len
        context_ids = padded(context_ids, context_len) # pad contexts to length context_len
        
        # Pad char_context_ids and char_qn_ids
        char_context_ids = pad_chars(char_context_ids, word_len, context_len)
        char_qn_ids = pad_chars(char_qn_ids, word_len, question_len)

        # Make qn_ids into a np array and create qn_mask
        qn_ids = np.array(qn_ids) # shape (batch_size, question_len)
        qn_mask = (qn_ids != PAD_ID).astype(np.int32) # shape (batch_size, question_len)

        # Make char_qn_ids into a np array and create char_qn_mask
        char_qn_ids = np.array(char_qn_ids)  # shape (batch_size, question_len, word_len)
        char_qn_mask = (char_qn_ids != PAD_ID).astype(np.int32) # shape (batch_size, question_len, word_len)

        # Make context_ids into a np array and create context_mask
        context_ids = np.array(context_ids) # shape (batch_size, context_len)
        context_mask = (context_ids != PAD_ID).astype(np.int32) # shape (batch_size, context_len)

        # Make char_context_ids into a np array and create char_context_mask
        char_context_ids = np.array(char_context_ids) # shape (batch_size, context_len)
        char_context_mask = (char_context_ids != PAD_ID).astype(np.int32) # shape (batch_size, context_len, word_len)

        # Make ans_span into a np array
        ans_span = np.array(ans_span) # shape (batch_size, 2)

        # Make into a Batch object
        batch = Batch(context_ids, context_mask, context_tokens, qn_ids, qn_mask, qn_tokens, ans_span, ans_tokens, char_context_ids, char_qn_ids, char_context_mask, char_qn_mask)

        yield batch

    return
