from __future__ import print_function

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from config import set_args
transformer_args = set_args()


# [docs]class TensorDataset(Dataset[Tuple[Tensor, ...]]):
#     r"""Dataset wrapping tensors.
#
#     Each sample will be retrieved by indexing tensors along the first dimension.
#
#     Args:
#         *tensors (Tensor): tensors that have the same size of the first dimension.
#     """
#     tensors: Tuple[Tensor, ...]
#
#     def __init__(self, *tensors: Tensor) -> None:
#         assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
#         self.tensors = tensors
#
#     def __getitem__(self, index):
#         return tuple(tensor[index] for tensor in self.tensors)
#
#     def __len__(self):
#         return self.tensors[0].size(0)


class InstanceSample(TensorDataset):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, *tensors):
        super().__init__(*tensors)
        k=transformer_args.nce_k
        mode='exact'
        is_sample=True
        percent=1.0
        self.image_tasks = ['celeba']

        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = transformer_args.nclasses
        num_samples = len(tensors[0])

        if transformer_args.task in self.image_tasks:
            label = tensors[-1]
        else:
            label = tensors[-2]


        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):

        if transformer_args.task in self.image_tasks:
            target = self.tensors[-1][index]
        else:
            target = self.tensors[-2][index]


    # sample contrastive examples
        if self.mode == 'exact':
            pos_idx = index
        elif self.mode == 'relax':
            pos_idx = np.random.choice(self.cls_positive[target], 1) #TODO: need to fix the seed
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.mode)
        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))

        # print('index: ',index)
        # print('sample_idx: ',sample_idx)

        return tuple([tensor[index] for tensor in self.tensors] + [index] + [sample_idx])

