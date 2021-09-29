# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import numpy as np
from transformers import BertTokenizer
import string
import torch
from random import randint, shuffle, choice
from random import random as rand
from w2v_util import pad_sequences
from config import set_args
import transformers

transformer_args = set_args()


class ABSATokenizer(BertTokenizer):
    def subword_tokenize(self, tokens, labels): # for AE
        split_tokens, split_labels= [], []
        idx_map=[]
        for ix, token in enumerate(tokens):
            sub_tokens=self.wordpiece_tokenizer.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                if labels[ix]=="B" and jx>0:
                    split_labels.append("I")
                else:
                    split_labels.append(labels[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, idx_map





class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text_a=None, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                    input_ids=None,
                    input_mask=None,
                    segment_ids=None,

                    tokens_term_ids=None,
                    tokens_sentence_ids=None,

                    term_input_ids=None,
                    term_input_mask=None,
                    term_segment_ids=None,

                    sentence_input_ids=None,
                    sentence_input_mask=None,
                    sentence_segment_ids=None,

                    tokens_term_sentence_ids=None,
                    label_id=None,

                    masked_lm_labels = None,
                    masked_pos = None,
                    masked_weights = None,

                    position_ids=None,

                    valid_ids=None,
                    label_mask=None

                    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.label_id = label_id

        self.masked_lm_labels = masked_lm_labels,
        self.masked_pos = masked_pos,
        self.masked_weights = masked_weights

        self.tokens_term_ids = tokens_term_ids
        self.tokens_sentence_ids = tokens_sentence_ids

        self.term_input_ids = term_input_ids
        self.term_input_mask = term_input_mask
        self.term_segment_ids = term_segment_ids

        self.sentence_input_ids = sentence_input_ids
        self.sentence_input_mask = sentence_input_mask
        self.sentence_segment_ids = sentence_segment_ids

        self.tokens_term_sentence_ids= tokens_term_sentence_ids

        self.position_ids = position_ids

        self.valid_ids = valid_ids
        self.label_mask = label_mask



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the SemEval Aspect Extraction ."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self,dataset):
        """See base class."""
        # return ["O", "B", "I"] #different for different dataset

        if dataset == 'conll2003':
            if transformer_args.overlap_only:
                return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
            else:
                return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        elif dataset == 'wnut2017':
            return ['O', 'B-location', 'I-location', 'B-corporation', 'I-corporation', 'B-person', 'I-person',
                    'B-product', 'I-product', 'B-creative-work', 'I-creative-work',
                    'B-group', 'I-group']
        elif dataset == 'wikigold':
            if transformer_args.overlap_only:
                return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
            else:
                return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        elif dataset == 'ontonote':
            if transformer_args.overlap_only:
                return ['O', 'B-PERSON', 'I-PERSON', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
            else:
                return ['O', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC',
                        'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE',
                        'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT',
                        'B-EVENT', 'I-EVENT','B-WORK_OF_ART','I-WORK_OF_ART',
                        'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE',
                        'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME',
                        'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY',
                        'B-QUANTITY', 'I-QUANTITY', 'B-ORDINAL', 'I-ORDINAL',
                        'B-CARDINAL', 'I-CARDINAL'
                        ]
        elif dataset == 'btc':
            if transformer_args.overlap_only:
                return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
            else:
                return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        elif dataset == 'ieer':
            if transformer_args.overlap_only:
                return ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
            else:
                return ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG',
                        'B-PCT', 'I-PCT', 'B-MON', 'I-MON',
                        'B-TIM', 'I-TIM', 'B-DAT', 'I-DAT',
                        'B-DUR', 'I-DUR','B-CAR','I-CAR',
                        'B-MEA', 'I-MEA'
                        ]
        elif dataset == 'ritter':
            return ['O', 'B-person', 'I-person', 'B-geo-loc', 'I-geo-loc', 'B-facility', 'I-facility',
                    'B-company', 'I-company', 'B-sportsteam', 'I-sportsteam',
                    'B-musicartist', 'I-musicartist', 'B-product', 'I-product',
                    'B-tvshow', 'I-tvshow','B-movie','I-movie',
                    'B-other', 'I-other'
                    ]
        elif dataset == 're3d':
            if transformer_args.overlap_only:
                return ['O','B-Person', 'I-Person','B-Organisation', 'I-Organisation','B-Location', 'I-Location']
            else:
                return ['O', 'B-Person', 'I-Person', 'B-DocumentReference', 'I-DocumentReference', 'B-Location', 'I-Location',
                        'B-MilitaryPlatform', 'I-MilitaryPlatform', 'B-Money', 'I-Money',
                        'B-Nationality', 'I-Nationality', 'B-Organisation', 'I-Organisation',
                        'B-Quantity', 'I-Quantity','B-Temporal','I-Temporal',
                        'B-Weapon', 'I-Weapon'
                        ]
        elif dataset == 'gum':
            if transformer_args.overlap_only:
                return ['O','B-person', 'I-person','B-organization', 'I-organization','B-place', 'I-place']
            else:
                return ['O', 'B-person', 'I-person', 'B-place', 'I-place', 'B-organization', 'I-organization',
                        'B-quantity', 'I-quantity', 'B-time', 'I-time',
                        'B-event', 'I-event', 'B-abstract', 'I-abstract',
                        'B-substance', 'I-substance','B-object','I-object',
                        'B-animal', 'I-animal','B-plant', 'I-plant'
                        ]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['tokens'] #no text b appearently
            label = lines[ids]['labels']
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label) )
        return examples

class DtcProcessor(DataProcessor):
    """Processor for document text classification."""

    def get_labels(self,ntasks):
        return [t for t in range(ntasks)]

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence #no need to split for us
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

class DscProcessor(DataProcessor):
    """Processor for document text classification."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        return ['-1','1']


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['sentence']
            text_b = None
            label = lines[ids]['polarity']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class AeProcessor(DataProcessor):
    """Processor for the SemEval Aspect Extraction ."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class."""
        return ["O", "B", "I"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['tokens'] #no text b appearently
            label = lines[ids]['labels']
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label) )
        return examples


class AscProcessor(DataProcessor):
    """Processor for the SemEval Aspect Sentiment Classification."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class."""
        return ["positive", "negative", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['term']
            text_b = lines[ids]['sentence']
            label = lines[ids]['polarity']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class SgProcessor(DataProcessor):
    """Processor for the Sentence Generation Task."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def _create_examples(self, lines, set_type):
        #no label, or, label is the sentence itself
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['sentence']
            text_b = lines[ids]['sentence']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples



class StringProcessor(DataProcessor):
    """Processor for the SemEval Aspect Sentiment Classification."""

    def get_examples(self, lines):
        """See base class."""
        return self._create_examples(lines)

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines:
            text_a = line
            examples.append(
                InputExample(text_a=text_a))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode):
    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    #text_b for sentence (segment 1); text_a for aspect (segment 0)
    # text_a = lines[ids]['term']
    # text_b = lines[ids]['sentence']
    # label = lines[ids]['polarity']

    if transformer_args.task == 'asc': # for pair
        label_map={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}
    elif transformer_args.task == 'nli':
        label_map={'neutral': 0, 'entailment': 1, 'contradiction': 2}
    elif transformer_args.task == 'ae':
        label_map={'B': 0, 'I': 1, 'O': 2}

    features = []
    for (ex_index, example) in enumerate(examples):
        if mode!="ae":
            tokens_a = tokenizer.tokenize(example.text_a)
        else: #only do subword tokenization.
            tokens_a, labels_a, example.idx_map= tokenizer.subword_tokenize([token.lower() for token in example.text_a], example.label )

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # token_a has a max_length
        if transformer_args.exp in ['3layer_aspect','2layer_aspect_transfer','2layer_aspect_dynamic']:
            term_position = tokens.index('[SEP]')-1
            while term_position < transformer_args.max_term_length: #[CLS],t,[SEP]
                input_ids.insert(term_position,0)
                input_mask.insert(term_position,0)
                segment_ids.insert(term_position,0)
                term_position+=1

            max_seq_length = max_seq_length
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if mode!="ae":
            label_id = label_map[example.label]
        else:
            label_id = [-1] * len(input_ids) #-1 is the index to ignore
            #truncate the label length if it exceeds the limit.
            lb=[label_map[label] for label in labels_a]
            if len(lb) > max_seq_length - 2:
                lb = lb[0:(max_seq_length - 2)]
            label_id[1:len(lb)+1] = lb


        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features_bert_sep(examples, label_list, max_term_length, max_sentence_length, tokenizer, mode):
    # 'sep' means separate representation for sentence and term"""
    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    '''seperate the sentence and term, instead of merging everything together'''
    label_map={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)


        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_b) > max_term_length - 2:
            tokens_b = tokens_b[0:(max_sentence_length - 2)]

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_term_length - 2:
            tokens_a = tokens_a[0:(max_term_length - 2)]

        term_tokens = []
        term_segment_ids = []
        term_tokens.append("[CLS]")
        term_segment_ids.append(0)
        for token in tokens_a:
            term_tokens.append(token)
            term_segment_ids.append(0)
        term_tokens.append("[SEP]")
        term_segment_ids.append(0)

        sentence_tokens = []
        sentence_segment_ids = []
        sentence_tokens.append("[CLS]")
        sentence_segment_ids.append(0)
        for token in tokens_b:
            sentence_tokens.append(token)
            sentence_segment_ids.append(0)
        sentence_tokens.append("[SEP]")
        sentence_segment_ids.append(0)

        term_input_ids = tokenizer.convert_tokens_to_ids(term_tokens)
        sentence_input_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        term_input_mask = [1] * len(term_input_ids)
        # Zero-pad up to the sequence length.
        while len(term_input_ids) < max_term_length:
            term_input_ids.append(0)
            term_input_mask.append(0)
            term_segment_ids.append(0)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        sentence_input_mask = [1] * len(sentence_input_ids)
        # Zero-pad up to the sequence length.
        while len(sentence_input_ids) < max_sentence_length:
            sentence_input_ids.append(0)
            sentence_input_mask.append(0)
            sentence_segment_ids.append(0)


        assert len(term_input_ids) == max_term_length
        assert len(term_input_mask) == max_term_length
        assert len(term_segment_ids) == max_term_length

        label_id = label_map[example.label]


        features.append(
                InputFeatures(
                        term_input_ids=term_input_ids,
                        term_input_mask=term_input_mask,
                        term_segment_ids=term_segment_ids,
                        sentence_input_ids=sentence_input_ids,
                        sentence_input_mask=sentence_input_mask,
                        sentence_segment_ids=sentence_segment_ids,
                        label_id=label_id))
    return features


def convert_examples_to_features_w2v_dsc(examples, label_list,tokenizer,args):

    # prepare for word2vector experiments

    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    label_map={'-1': 0, '1': 1}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_ids = tokenizer.texts_to_sequences([example.text_a.translate(str.maketrans('', '', string.punctuation)).lower()])
        tokens_a_ids =  pad_sequences(tokens_a_ids, maxlen=args.max_seq_length,padding='post',value=0)[0]

        # print('example.text_a: ',example.text_a.translate(str.maketrans('', '', string.punctuation)).lower())
        # print('tokens_a_ids: ',tokens_a_ids)
        # print('tokens_a_ids: ',len(tokens_a_ids))

        # exit()
        if 'newsgroup' in args.task:
            label_id = example.label
        else:
            label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        tokens_term_ids=tokens_a_ids,
                        tokens_sentence_ids=tokens_a_ids,
                        label_id=label_id))
    return features

def convert_examples_to_features_w2v(examples, label_list,tokenizer,args):

    # prepare for word2vector experiments

    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    label_map={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_ids = tokenizer.texts_to_sequences([example.text_a.translate(str.maketrans('', '', string.punctuation)).lower()])
        tokens_b_ids = tokenizer.texts_to_sequences([example.text_b.translate(str.maketrans('', '', string.punctuation)).lower()])

        # if len(tokens_a_ids[0])<= 0 or len(tokens_b_ids[0])<= 0:
        #     print('empty')
        #     continue
        # assert len(tokens_a_ids[0]) > 0
        # assert len(tokens_b_ids[0]) > 0

        tokens_a_ids =  pad_sequences(tokens_a_ids, maxlen=args.max_term_length,padding='post',value=0)[0]
        tokens_b_ids =  pad_sequences(tokens_b_ids, maxlen=args.max_sentence_length,padding='post',value=0)[0]

        label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        tokens_term_ids=tokens_a_ids,
                        tokens_sentence_ids=tokens_b_ids,
                        label_id=label_id))

    return features


def convert_examples_to_features_w2v_as(examples, label_list,tokenizer,args):
    # w2v also considers aspect (by adding aspect at the beginning)
    # prepare for word2vector experiments

    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    if transformer_args.task == 'asc':
        label_map={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}
    elif transformer_args.task == 'nli':
        label_map={'neutral': 0, 'entailment': 1, 'contradiction': 2}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_ids = tokenizer.texts_to_sequences([example.text_a.translate(str.maketrans('', '', string.punctuation)).lower()])
        tokens_b_ids = tokenizer.texts_to_sequences([example.text_b.translate(str.maketrans('', '', string.punctuation)).lower()])

        tokens_ids = tokenizer.texts_to_sequences(
               [example.text_a.translate(str.maketrans('', '', string.punctuation)).lower() + ' ' +
               example.text_b.translate(str.maketrans('', '', string.punctuation)).lower()])

        # print([example.text_a.translate(str.maketrans('', '', string.punctuation)).lower()])
        # print([example.text_a.translate(str.maketrans('', '', string.punctuation)).lower() + ' ' +
        #        example.text_b.translate(str.maketrans('', '', string.punctuation)).lower()])
        # print('tokens_ids: ',tokens_ids)
        # print('tokens_a_ids: ',tokens_a_ids)

        # if len(tokens_a_ids[0])<= 0 or len(tokens_b_ids[0])<= 0:
        #     print('empty')
        #     continue
        # assert len(tokens_a_ids[0]) > 0
        # assert len(tokens_b_ids[0]) > 0

        tokens_ids =  pad_sequences(tokens_ids, maxlen=args.max_seq_length,padding='post',value=0)[0]
        tokens_a_ids =  pad_sequences(tokens_a_ids, maxlen=args.max_term_length,padding='post',value=0)[0]
        tokens_b_ids =  pad_sequences(tokens_b_ids, maxlen=args.max_sentence_length,padding='post',value=0)[0]

        label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        tokens_term_sentence_ids=tokens_ids,
                        tokens_term_ids=tokens_a_ids,
                        tokens_sentence_ids=tokens_b_ids,
                        label_id=label_id))

    return features



def convert_examples_to_features_ner_w2v(examples, label_list,tokenizer,args):

    label_map = {label : i for i, label in enumerate(label_list,1)}
    word_index = tokenizer.word_index
    features = []
    for (ex_index, example) in enumerate(examples):
        label_ids =[]
        label_mask = []
        tokens_ids = []
        textlist = []

        text_a = '[CLS] ' + example.text_a + ' [SEP]'
        labels = ['[CLS]'] + example.label + ['[SEP]']

        for word in text_a.lower().lower().split(' '): # we take into account case
            textlist.append(word_index.get(word))

        assert len(textlist) == len(labels)

        for text_id,text in enumerate(textlist):
            label_mask.append(1)
            tokens_ids.append(text)
            label_ids.append(label_map[labels[text_id]])


        while len(label_mask) < args.max_seq_length: label_mask.append(0)
        while len(tokens_ids) < args.max_seq_length: tokens_ids.append(0)
        while len(label_ids) < args.max_seq_length: label_ids.append(0)

        # print('len(label_mask): ',len(label_mask))
        # print('len(tokens_ids): ',len(tokens_ids))
        # print('len(label_ids): ',len(label_ids))

        assert len(label_mask) == len(tokens_ids) == len(label_ids) == args.max_seq_length

        features.append(
                InputFeatures(
                        tokens_sentence_ids=tokens_ids,
                        label_id=label_ids,
                        label_mask=label_mask))

    return features


#adopt from https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb
def convert_examples_to_features_ner(examples, label_list, max_seq_length, tokenizer, mode):
    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    #text_b for sentence (segment 1); text_a for aspect (segment 0)
    # text_a = lines[ids]['term']
    # text_b = lines[ids]['sentence']
    # label = lines[ids]['polarity']

    # print(tokenizer)

    overlap_class = \
        ['O',
         'B-PER','I-PER','B-person','I-person','B-PERSON','I-PERSON','B-Person','I-Person',
         'B-ORG','I-ORG','B-organization','I-organization','B-Organisation','I-Organisation',
         'B-LOC','I-LOC','B-location','I-location','B-place','I-place'
         ]


    features = []
    label_all_tokens = True

    for (ex_index, example) in enumerate(examples):
        tokenized_input = \
            tokenizer(example.text_a, is_split_into_words=True,max_length=max_seq_length,padding='max_length',truncation=True)
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
        word_ids = tokenized_input.word_ids(batch_index=0)

        if transformer_args.overlap_only:
            is_continue = False
            for label in example.label:
                if label not in overlap_class:
                    is_continue = True
                    break
            if is_continue: continue
            # print('example.label: ',example.label)


        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-1)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_map[example.label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label_map[example.label[word_idx]] if label_all_tokens else -1)
            previous_word_idx = word_idx

        input_ids = tokenized_input['input_ids']
        segment_ids = tokenized_input['token_type_ids']
        input_mask = tokenized_input['attention_mask']

        # print('labels: ',labels)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        # label_id = [-1] * len(input_ids) #-1 is the index to ignore
        # #truncate the label length if it exceeds the limit.
        # lb=[label_map[label] for label in labels_a]
        # if len(lb) > max_seq_length - 2:
        #     lb = lb[0:(max_seq_length - 2)]
        # label_id[1:len(lb)+1] = lb


        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_ids))
    return features



def convert_examples_to_features_dtc(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    #TODO: input document only

    features = []
    for (ex_index,example) in enumerate(examples):
        labels_a = example.label
        tokens_a = tokenizer.tokenize(example.text_a)

        # print('labels_a: ',labels_a)
        # print('example.text_a: ',example.text_a)
        # print('tokens_a: ',tokens_a)


        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=labels_a))
    return features




def convert_examples_to_features_dsc(examples, label_list, max_seq_length, tokenizer,mode):
    """Loads a data file into a list of `InputBatch`s."""
    #TODO: input document only
    label_map={'-1': 0, '1': 1}
    if transformer_args.task == 'dsc':
        label_map={'-1': 0, '1': 1}
    elif transformer_args.task == 'ssc':
        label_map={'0': 0, '1': 1,'2': 2}

    features = []
    for (ex_index,example) in enumerate(examples):
        labels_a = label_map[example.label]
        tokens_a = tokenizer.tokenize(example.text_a)

        # print('labels_a: ',labels_a)
        # print('example.text_a: ',example.text_a)
        # print('tokens_a: ',tokens_a)


        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=labels_a))
    return features


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens



def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]





def convert_example_feature_lm_prompt():
    #TODO: need to deal with prompt version

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id] 
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == 'prompt-demo':
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail: 
            logger.info("Automatically convert the template to GPT-3's in-context learning.")
            assert data_args.template_list is None

            old_template = data_args.template
            new_template = old_template + ''
            old_template = old_template.replace('*cls*', '')
            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template:
                sent_num = 2
            for instance_id in range(data_args.gpt3_in_context_num):
                sub_template = old_template + ''
                # Replace sent_id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * instance_id + sent_id))
                # Replace mask
                sub_template = sub_template.replace("*mask*", "*labelx_{}*".format(instance_id))
                if data_args.gpt3_in_context_tail:
                    new_template = new_template + sub_template # Put context at the end
                else:
                    new_template = sub_template + new_template # Put context at the beginning
            logger.info("| {} => {}".format(data_args.template, new_template))
            data_args.template = new_template
        else:
            logger.info("Automatically convert the template to using demonstrations.")
            if data_args.template_list is not None:
                for i in range(len(data_args.template_list)):
                    old_template = data_args.template_list[i]
                    new_template = old_template + ''
                    old_template = old_template.replace('*cls*', '')
                    # Single sentence or sentence pair?
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ''
                        # Replace sent id
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * label_id + sent_id))
                        # Replace mask
                        sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                        new_template = new_template + sub_template
                    logger.info("| {} => {}".format(data_args.template_list[i], new_template))
                    data_args.template_list[i] = new_template
            else:
                old_template = data_args.template
                new_template = old_template + ''
                old_template = old_template.replace('*cls*', '')
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ''
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace("_{}".format(sent_id), "_{}".format(sent_num + sent_num * label_id + sent_id))
                    # Replace mask
                    sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                    new_template = new_template + sub_template
                logger.info("| {} => {}".format(data_args.template, new_template))
                data_args.template = new_template

    
    

