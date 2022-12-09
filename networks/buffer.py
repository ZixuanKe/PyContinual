# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple, Dict
from torchvision import transforms


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size: #total batch size
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['input_ids', 'labels', 'logits', 'attention_mask', 'task', 'nsp_labels']

    def init_tensors(self,
                    input_ids: torch.Tensor,
                    labels: torch.Tensor,
                    logits: torch.Tensor,
                    attention_mask: torch.Tensor,
                    task: torch.Tensor,
                    nsp_labels: torch.Tensor) -> None:

        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.float32 if attr_str.endswith('gits') else torch.int64
                setattr(self, attr_str, torch.zeros((self.buffer_size,*attr.shape[1:]), dtype=typ, device=self.device))


    def get_size(self):
        return self.num_seen_examples
    
    def __len__(self):
        return self.num_seen_examples

    def add_data(self, input_ids,  labels=None, logits=None, attention_mask=None, task=None, nsp_labels=None):

        if not hasattr(self, 'input_ids'):
            self.init_tensors(input_ids,labels,logits, attention_mask,task,nsp_labels)

        for i in range(input_ids.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.input_ids[index] = input_ids[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task is not None:
                    self.task[index] = task[i].to(self.device)
                if nsp_labels is not None:
                    self.nsp_labels[index] = nsp_labels[i].to(self.device)
                if attention_mask is not None:
                    self.attention_mask[index] = attention_mask[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.input_ids.shape[0]):
            size = min(self.num_seen_examples, self.input_ids.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.input_ids.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.input_ids[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple
    
    def get_datadict(self, size: int, transform: transforms=None) -> Dict:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.input_ids.shape[0]):
            size = min(self.num_seen_examples, self.input_ids.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.input_ids.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_dict = {'input_ids': torch.stack([transform(ee.cpu())
                            for ee in self.input_ids[choice]]).to(self.device),}
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_dict[attr_str] = attr[choice]

        return ret_dict

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.input_ids]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0