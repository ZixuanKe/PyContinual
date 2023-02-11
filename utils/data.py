
from typing import Any, Dict, List, NewType, Optional, Tuple, Union
from transformers import (
    DataCollatorForLanguageModeling,
    # get_scheduler,
)
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
import numpy as np
from dataclasses import dataclass
import torch

label_list_dict = \
{
    'conll2003': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
    'wnut2017': ['O', 'B-location', 'I-location', 'B-corporation', 'I-corporation', 'B-person', 'I-person',
                    'B-product', 'I-product', 'B-creative-work', 'I-creative-work',
                    'B-group', 'I-group'],
    'wikigold': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
    'ontonote': ['O', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC',
                        'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE',
                        'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT',
                        'B-EVENT', 'I-EVENT','B-WORK_OF_ART','I-WORK_OF_ART',
                        'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE',
                        'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME',
                        'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY',
                        'B-QUANTITY', 'I-QUANTITY', 'B-ORDINAL', 'I-ORDINAL',
                        'B-CARDINAL', 'I-CARDINAL'
                        ],
    'btc': ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
    'ieer': ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG',
                        'B-PCT', 'I-PCT', 'B-MON', 'I-MON',
                        'B-TIM', 'I-TIM', 'B-DAT', 'I-DAT',
                        'B-DUR', 'I-DUR','B-CAR','I-CAR',
                        'B-MEA', 'I-MEA'
                        ],
    'ritter': ['O', 'B-person', 'I-person', 'B-geo-loc', 'I-geo-loc', 'B-facility', 'I-facility',
                    'B-company', 'I-company', 'B-sportsteam', 'I-sportsteam',
                    'B-musicartist', 'I-musicartist', 'B-product', 'I-product',
                    'B-tvshow', 'I-tvshow','B-movie','I-movie',
                    'B-other', 'I-other'
                    ],
    're3d': ['O', 'B-Person', 'I-Person', 'B-DocumentReference', 'I-DocumentReference', 'B-Location', 'I-Location',
                        'B-MilitaryPlatform', 'I-MilitaryPlatform', 'B-Money', 'I-Money',
                        'B-Nationality', 'I-Nationality', 'B-Organisation', 'I-Organisation',
                        'B-Quantity', 'I-Quantity','B-Temporal','I-Temporal',
                        'B-Weapon', 'I-Weapon'
                        ],
    'gum': ['O', 'B-person', 'I-person', 'B-place', 'I-place', 'B-organization', 'I-organization',
                        'B-quantity', 'I-quantity', 'B-time', 'I-time',
                        'B-event', 'I-event', 'B-abstract', 'I-abstract',
                        'B-substance', 'I-substance','B-object','I-object',
                        'B-animal', 'I-animal','B-plant', 'I-plant'
                        ]
}



label_to_id_dict = \
{
    'conll2003': {l: i for i, l in enumerate(label_list_dict['conll2003'])},
    'wnut2017': {l: i for i, l in enumerate(label_list_dict['wnut2017'])},
    'wikigold': {l: i for i, l in enumerate(label_list_dict['wikigold'])},
    'ontonote': {l: i for i, l in enumerate(label_list_dict['ontonote'])},
    'btc': {l: i for i, l in enumerate(label_list_dict['btc'])},
    'ieer': {l: i for i, l in enumerate(label_list_dict['ieer'])},
    'ritter': {l: i for i, l in enumerate(label_list_dict['ritter'])},
    're3d': {l: i for i, l in enumerate(label_list_dict['re3d'])},
    'gum': {l: i for i, l in enumerate(label_list_dict['gum'])},
}




def preprocess_function(examples, text_column, summary_column, args):
    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Temporarily set max_target_length for training.
    padding = "max_length" if args.pad_to_max_length else False

    inputs = examples[text_column]
    targets = examples[summary_column]
    task_id = examples['task']
    if 'cls_labels' in examples: cls_labels = examples['cls_labels']


    inputs = [prefix + inp for inp in inputs]

    model_inputs = args.tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

    # Setup the tokenizer for targets
    with args.tokenizer.as_target_tokenizer():
        labels = args.tokenizer(targets, max_length=args.max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != args.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]


    model_inputs["labels"] = labels["input_ids"]
    model_inputs['task'] = task_id
    if 'cls_labels' in examples: model_inputs['cls_labels'] = cls_labels

    return model_inputs


def tokenize_and_align_labels(examples, text_column, summary_column, label_to_id, b_to_i_label, eval_t, args):

    # Temporarily set max_target_length for training.
    padding = "max_length" if args.pad_to_max_length else False

    tokenized_inputs = args.tokenizer(
        examples[text_column],
        max_length=args.max_length,
        padding=padding,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples[summary_column]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                if eval_t is None or args.ft_task == eval_t: #only do this when matching
                    # print('eval_t: ', eval_t)
                    # print('args.ft_task: ', args.ft_task)
                    # print('label_to_id: ', label_to_id)
                    label_ids.append(label_to_id[label[word_idx]])
                else:
                    label_ids.append(label_to_id_dict[args.task_name][label[word_idx]])

            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if args.label_all_tokens:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["cls_labels"] = labels
    tokenized_inputs["labels"] = labels


    task_id = examples['task']

    inputs = examples[text_column]

    tokenized_inputs['task'] = task_id

    return tokenized_inputs


InputDataClass = NewType("InputDataClass", Any)


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0]:] = example
    return result



@dataclass
class MyDataCollatorForSeq2Seq:

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    mlm_probability: float = 0.15

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids


        special_tokens_mask = features.pop("special_tokens_mask", None)  # excloude special token
        features["inputs_ids_mlm"], features["labels_mlm"], features["masked_indices"] = \
            self.torch_mask_tokens(features["input_ids"].clone(),special_tokens_mask=special_tokens_mask)

        # note: must clone, feature will be overwritten otherwise

        return features



    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        inputs_ori = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, masked_indices



def concate_batch(batch_a, batch_b,args):
    cat_batch = {'input_ids': torch.cat([batch_a['input_ids'],batch_b['input_ids'][:args.per_device_train_batch_size]]),
                   'attention_mask': torch.cat([batch_a['attention_mask'],batch_b['attention_mask'][:args.per_device_train_batch_size]]),
                   'labels': torch.cat([batch_a['labels'],batch_b['labels'][:args.per_device_train_batch_size]]),
                   'cls_labels': torch.cat([batch_a['cls_labels'],batch_b['cls_labels'][:args.per_device_train_batch_size]]),
                    'inputs_ids_mlm': torch.cat([batch_a['inputs_ids_mlm'],batch_b['inputs_ids_mlm'][:args.per_device_train_batch_size]]),
                    'labels_mlm': torch.cat([batch_a['labels_mlm'],batch_b['labels_mlm'][:args.per_device_train_batch_size]]),
                   'task': torch.cat([batch_a['task'],batch_b['task'][:args.per_device_train_batch_size]])}

    return cat_batch