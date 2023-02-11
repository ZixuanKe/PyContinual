# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
import os
from tqdm.auto import tqdm
import torch.nn as nn
import json
from networks.transformers.bart import MyBartForConditionalGeneration,MyBartForSequenceClassification,MyBartForTokenClassification
from networks.transformers.roberta import MyRobertaForSequenceClassification,MyRobertaForTokenClassification
from networks.transformers.bert import MyBertForSequenceClassification,MyBertForTokenClassification
import torch.distributed as dist
from collections import Counter


#TODO: MBPA++ https://github.com/h3lio5/episodic-lifelong-learning


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
    def __init__(self, buffer_size, device= 'cuda', n_tasks=None, args=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        self.args = args
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples','attention_mask','labels','cls_labels', 'examples_mlm', 'labels_mlm', 'task']  #TODO: add CLS label

        if 'roberta' in args.model_name_or_path:
            if args.task_name in args.ner_datasets:
                MODEL = MyRobertaForTokenClassification
            elif args.task_name in args.classification:
                MODEL = MyRobertaForSequenceClassification
        elif 'bart' in args.model_name_or_path:
            if args.task_name in args.ner_datasets:
                MODEL = MyBartForTokenClassification
            elif args.task_name in args.classification:
                MODEL = MyBartForSequenceClassification
            elif args.task_name in args.generation:
                MODEL = MyBartForConditionalGeneration
        elif 'bert' in args.model_name_or_path:
            if args.task_name in args.ner_datasets:
                MODEL = MyRobertaForTokenClassification
            elif args.task_name in args.classification:
                MODEL = MyBertForSequenceClassification




    def init_tensors(self,
                     examples: torch.Tensor,
                     attention_mask: torch.Tensor,
                     labels: torch.Tensor,
                     cls_labels: torch.Tensor,
                     examples_mlm: torch.Tensor,
                     labels_mlm: torch.Tensor,
                     task: torch.Tensor) -> None: # used in "eval()"

        for attr_str in self.attributes:

            attr = eval(attr_str)

            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32

                if attr_str == 'task': # do not use 0 becuase will task==0 needs to be used later
                    setattr(self, attr_str, torch.zeros((self.buffer_size,*attr.shape[1:]), dtype=typ, device=self.device) -1)
                else:
                    setattr(self, attr_str, torch.zeros((self.buffer_size,*attr.shape[1:]), dtype=typ, device=self.device))


    def get_size(self):
        return self.num_seen_examples


    def append_memory_batch(self,batch,size,t=None):

        if t is not None:
            buf_inputs, buf_attention_mask, buf_labels, buf_cls_labels, buf_inputs_mlm, buf_labels_mlm, buf_task = self.get_task_data(size=size,t=t)

        else:
            buf_inputs, buf_attention_mask, buf_labels, buf_cls_labels, buf_inputs_mlm, buf_labels_mlm, buf_task = self.get_data(size)

        buf = {
            'input_ids':buf_inputs.long(),
            'attention_mask': buf_attention_mask.long(),
            'labels': buf_labels.long(),
            'cls_labels': buf_cls_labels.long(),
            'inputs_ids_mlm': buf_inputs_mlm.long(),
            'labels_mlm': buf_labels_mlm.long(),
            'task': buf_task.long(),
        }

        buf['input_ids'] = torch.cat([batch['input_ids'],buf_inputs])
        buf['attention_mask'] = torch.cat([batch['attention_mask'],buf_attention_mask])
        buf['labels'] = torch.cat([batch['labels'],buf_labels])
        buf['cls_labels'] = torch.cat([batch['cls_labels'],buf_cls_labels])
        buf['inputs_ids_mlm'] = torch.cat([batch['inputs_ids_mlm'],buf_inputs_mlm])
        buf['labels_mlm'] = torch.cat([batch['labels_mlm'],buf_labels_mlm])
        buf['task']  = torch.cat([batch['task'],buf_task])

        return buf


    def retrive_memory(self,size=None,t=None):

        if t is not None:
            buf_inputs, buf_attention_mask, buf_labels, buf_cls_labels, buf_inputs_mlm, buf_labels_mlm, buf_task = self.get_task_data(size=size,t=t)

        else:
            buf_inputs, buf_attention_mask, buf_labels, buf_cls_labels, buf_inputs_mlm, buf_labels_mlm, buf_task = self.get_data(size)

        buf = {
            'input_ids':buf_inputs.long(),
            'attention_mask': buf_attention_mask.long(),
            'labels': buf_labels.long(),
            'cls_labels': buf_cls_labels.long(),
            'inputs_ids_mlm': buf_inputs_mlm.long(),
            'labels_mlm': buf_labels_mlm.long(),
            'task': buf_task.long(),
        }


        return buf

    def add_data(self, model,examples, attention_mask=None,  labels=None, cls_labels=None, examples_mlm=None, labels_mlm=None, task=None):

        if not hasattr(self, 'examples'):
            self.init_tensors(examples, attention_mask, labels, cls_labels, examples_mlm,labels_mlm, task)

        # it is random and not consider task
        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                if examples is not None:
                    self.examples[index] = examples[i].to(self.device)
                if attention_mask is not None:
                    self.attention_mask[index] = attention_mask[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if cls_labels is not None:
                    self.cls_labels[index] = cls_labels[i].to(self.device)
                if examples_mlm is not None:
                    self.examples_mlm[index] = examples_mlm[i].to(self.device)
                if labels_mlm is not None:
                    self.labels_mlm[index] = labels_mlm[i].to(self.device)
                if task is not None:
                    self.task[index] = task[i].to(self.device)


    def add_from_loader(self,model,train_dataloader):

        # Init

        # There are modes
        # 1. make sure each class has a fixed number
        # 2. only gives fixed total size, so more samples are saved for each task at the beining.


        progress_bar = tqdm(range(len(train_dataloader)))


        class_count = {c:0 for c in range(self.args.taskcla[self.args.ft_task][1])}
        sample_count = 0
        input_ids_save = []
        attention_mask_save = []
        labels_save = []
        cls_labels_ids_save = []
        inputs_ids_mlm_save = []
        attention_mask_mlm_save = []
        task_save = []

        for step, batch in enumerate(train_dataloader):


            input_ids = self.gather_by_cat(batch['input_ids'])
            attention_mask = self.gather_by_cat(batch['attention_mask'])
            labels = self.gather_by_cat(batch['labels'])
            cls_labels = self.gather_by_cat(batch['cls_labels'])
            inputs_ids_mlm = self.gather_by_cat(batch['inputs_ids_mlm'])
            labels_mlm = self.gather_by_cat(batch['labels_mlm'])
            task = self.gather_by_cat(batch['task'])

            # for ner, the labels is a list

            if self.args.task_name in self.args.ccd_datasets or self.args.task_name in self.args.asc_datasets:
                cls_list = cls_labels.cpu().numpy()

                for cls in range(task.size(0)):
                    cur_label = cls_list[cls]
                    if class_count[cur_label] <= self.args.replay_sample_per_class: # can decided by each label
                        class_count[cur_label]+=1
                        input_ids_save.append(input_ids[cls].clone())
                        attention_mask_save.append(attention_mask[cls].clone())
                        labels_save.append(labels[cls].clone())
                        cls_labels_ids_save.append(cls_labels[cls].clone())
                        inputs_ids_mlm_save.append(inputs_ids_mlm[cls].clone())
                        attention_mask_mlm_save.append(labels_mlm[cls].clone())
                        task_save.append(task[cls].clone())

            elif self.args.task_name in self.args.ner_datasets and self.args.sample_cap is None:
                cls_list = cls_labels.cpu().numpy()
                for cls in range(task.size(0)):
                    cur_label = cls_list[cls]

                    for k, v in class_count.items():
                        if v <= self.args.replay_sample_per_class and k in cur_label: # need a for loop to look at each one
                            input_ids_save.append(input_ids[cls].clone())
                            attention_mask_save.append(attention_mask[cls].clone())
                            labels_save.append(labels[cls].clone())
                            cls_labels_ids_save.append(cls_labels[cls].clone())
                            inputs_ids_mlm_save.append(inputs_ids_mlm[cls].clone())
                            attention_mask_mlm_save.append(labels_mlm[cls].clone())
                            task_save.append(task[cls].clone())
                            class_count[k] += 1

            else: # based on replay_sample_per_task
                for cls in range(task.size(0)):
                    if sample_count <= self.args.replay_sample_per_task:
                        sample_count += 1
                        input_ids_save.append(input_ids[cls].clone())
                        attention_mask_save.append(attention_mask[cls].clone())
                        labels_save.append(labels[cls].clone())
                        cls_labels_ids_save.append(cls_labels[cls].clone())
                        inputs_ids_mlm_save.append(inputs_ids_mlm[cls].clone())
                        attention_mask_mlm_save.append(labels_mlm[cls].clone())
                        task_save.append(task[cls].clone())

            progress_bar.update(1)
            progress_bar.set_description('Memory Compute Iter ')  # show the loss, mean while

        print('class_count: ', class_count)
        print('input_ids_save: ', len(input_ids_save))

        input_ids = torch.stack(input_ids_save)
        attention_mask = torch.stack(attention_mask_save)
        labels = torch.stack(labels_save)
        cls_labels = torch.stack(cls_labels_ids_save)
        inputs_ids_mlm = torch.stack(inputs_ids_mlm_save)
        labels_mlm = torch.stack(attention_mask_mlm_save)
        task = torch.stack(task_save)
        # print('input_ids: ', input_ids.size())

        self.add_data(
            model=model,
            examples=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            cls_labels=cls_labels,
            examples_mlm=inputs_ids_mlm,
            labels_mlm=labels_mlm,
            task=task
        )

        return self

    def gather_by_cat(self,head_impt):
        head_impt_list = [torch.zeros_like(head_impt) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list=head_impt_list,
                        tensor=head_impt.contiguous())  # everyone need to do this
        head_impt_cat = torch.cat(head_impt_list)
        return head_impt_cat


    def get_data(self, size: int, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple


    def get_task_data(self,size: int,t: int, transform: transforms=None):

        indices = []
        for task_id, task in enumerate(self.task.cpu().numpy()[:self.num_seen_examples]):
            if task == t:
                indices.append(task_id)

        if size is None:
            choice = indices
        else:
            choice = np.random.choice(indices, size=size, replace=False)

        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple




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
        size = min(self.num_seen_examples, self.examples.shape[0])
        choice = range(size)

        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0




    def get_all_keys(self,model):

        print(self.num_seen_examples)
        examples = self.examples[:self.num_seen_examples]
        attention_mask = self.attention_mask[:self.num_seen_examples]
        labels = self.labels[:self.num_seen_examples]
        cls_labels = self.cls_labels[:self.num_seen_examples]
        examples_mlm = self.examples_mlm[:self.num_seen_examples]
        labels_mlm = self.labels_mlm[:self.num_seen_examples]
        task = self.task[:self.num_seen_examples]

        with torch.no_grad():
            batch = {'input_ids': examples, 'attention_mask': attention_mask,
                           'labels': labels,
                           'cls_labels': cls_labels, 'task': task}

            outputs = model(batch, only_return_output=True)  # no gradinent needed

        return outputs.hidden_states[-1][:, 0, :] # roberta/bert only


    def get_keys(self, batch, model):
        with torch.no_grad():
            input_batch = {'input_ids': batch['input_ids'],
                           'attention_mask':batch['attention_mask'],
                           'labels': batch['labels'],
                           'cls_labels':batch['cls_labels'],
                           'task': batch['task']}

            outputs = model(input_batch)
        return outputs.hidden_states[-1][:, 0, :] # roberta/bert only

    def get_neighbours_til(self,eval_t,size=None):
        # if I know the task ID
        # we still need size
        """
        Returns samples from buffer using nearest neighbour approach
        """

        indices = []
        for task_id, task in enumerate(self.task.cpu().numpy()[:self.num_seen_examples]):
            if task == eval_t:
                indices.append(task_id)


        if size is None:
            choice = np.random.choice(indices, size=indices, replace=False)
        else:
            choice = np.random.choice(indices, size=size, replace=False)

        examples = torch.stack([ee.cpu()for ee in self.examples[choice]]).long().to(self.device)
        attention_mask = torch.stack([ee.cpu()for ee in self.attention_mask[choice]]).long().to(self.device)
        labels = torch.stack([ee.cpu()for ee in self.labels[choice]]).long().to(self.device)
        cls_labels = torch.stack([ee.cpu()for ee in self.cls_labels[choice]]).long().to(self.device)
        examples_mlm = torch.stack([ee.cpu()for ee in self.examples_mlm[choice]]).long().to(self.device)
        labels_mlm = torch.stack([ee.cpu()for ee in self.labels_mlm[choice]]).long().to(self.device)
        task = torch.stack([ee.cpu()for ee in self.task[choice]]).long().to(self.device)



        return examples, attention_mask, labels, cls_labels, examples_mlm, labels_mlm, task



    def get_neighbours(self, keys,model, k=20):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        all_keys = self.get_all_keys(model)

        # if self.args.task_name in self.args.classification:
        #     num_class = sum([_[1] for _ in self.args.taskcla[:self.args.ft_task]])
        #     k = k * num_class

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        samples = []
        for key in keys:
            sim = cos(key, all_keys)
            selection = torch.topk(sim, k)
            indices = selection.indices

            neighbours = (self.examples[indices].long().cuda(),
                          self.attention_mask[indices].long().cuda(),
                          self.labels[indices].long().cuda(),
                          self.cls_labels[indices].long().cuda(),
                          self.examples_mlm[indices].long().cuda(),
                          self.labels_mlm[indices].long().cuda(),
                          self.task[indices].long().cuda())
            samples.append(neighbours)

        return samples


    def process_lst(self,lst):
        return [np.array(x) for x in lst]

    def process_array(self,lst):
        return [x.tolist() for x in lst]

    def process_int(self,lst):
        return [int(x) for x in lst]

    def save(self, path):
        obj = [self.examples, self.attention_mask, self.labels, self.cls_labels, self.task, self.examples_mlm, self.labels_mlm, self.num_seen_examples]
        torch.save(obj, path)

    def load(self, path):
        self.examples, self.attention_mask, self.labels, self.cls_labels, self.task, self.examples_mlm, self.labels_mlm, self.num_seen_examples = torch.load(path,map_location='cpu')
        self.examples = self.examples.long().cuda()
        self.attention_mask = self.attention_mask.long().cuda()
        self.labels = self.labels.long().cuda()
        self.cls_labels = self.cls_labels.long().cuda()
        self.examples_mlm = self.examples_mlm.long().cuda()
        self.labels_mlm = self.labels_mlm.long().cuda()
        self.task = self.task.long().cuda()