import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append("./approaches/")
from memory import ContrastMemory
from torch.nn.modules.loss import _Loss
from torch import Tensor

eps = 1e-7




class MyContrastive(nn.Module):
    # https://github.com/facebookresearch/moco/blob/3631be074a0a14ab85c206631729fe035e54b525/moco/builder.py#L155
    def __init__(self,args):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MyContrastive, self).__init__()
        self.args = args

        #Z function as a simple 2 layers of MLP
        self.contrast_encoder = nn.Sequential(
                      nn.Linear(self.args.bert_hidden_size, self.args.bert_hidden_size)).cuda()


        self.T = self.args.temp
        self.ce=torch.nn.CrossEntropyLoss()


    def forward(self, aug_x, order_x,weights,tasks=None):
        """
        Input:
            im_q: a batch of query images, in our case, Aug(X)
            im_k: a batch of key images, in our case, F_0, F_1, F_2...., F_n
        Output:
            logits, targets; so that you can use cross entropy
        """
        if self.args.contrastive_with_mlp:
            # compute query features
            # print('aug_x: ',aug_x.size())

            aug_x = self.contrast_encoder(aug_x)  # queries: NxC
            aug_x = nn.functional.normalize(aug_x, dim=1)

            # print('aug_x: ',aug_x.size())
            # print('underly_x: ',underly_x.size())

            order_x = self.contrast_encoder(order_x)  # keys: NxC
            order_x = nn.functional.normalize(order_x, dim=1)
            # print('underly_x: ',underly_x.size())

        else:


            aug_x = nn.functional.normalize(aug_x, dim=-1)
            order_x = nn.functional.normalize(order_x, dim=-1)



        #we don't need to separate neg and pos
        #  logits: NxK
        if self.args.contrastive_with_mlp:
            l_order = torch.einsum('nci,nkc->nki', [aug_x.unsqueeze(-1), order_x.permute(0,2,1)]).squeeze(-1) # compute similarity for negative
        else:
            l_order = torch.einsum('nci,nkc->nki', [aug_x.unsqueeze(-1), order_x]).squeeze(-1) # compute similarity for negative

        # logits: Nx(1+K)
        logits = l_order

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        amix_loss = self.ce(logits, tasks)

        return amix_loss



#Also refere to https://github.com/Lee-Gihun/MixCo-Mixup-Contrast/blob/main/moco/utils/loss_fn.py
class LabelSmoothingCrossEntropy(_Loss):
    def __init__(self, eps: float = 0.1, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(size_average, reduce, reduction)
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_input = F.log_softmax(input, dim=-1)
        loss = (- target * log_input).sum(-1)
        if self.reduction == "none":
            ret = loss
        elif self.reduction == "mean":
            ret = loss.mean()
        elif self.reduction == "sum":
            ret = loss.sum()
        else:
            raise ValueError(self.reduction + " is not valid")
        return ret



class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss




#implement contrastive loss:
#try to make everything inside the loss function, contrastive loss is simply a change of loss function
#adapt from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, scores=None, args=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) #mask out different label, only consider the same label
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))


        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # logit can be high

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases, not the other, so labels==None case is fine
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = (torch.exp(logits) * logits_mask) #negative here, even eps, the results are very bad
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))# finish contrastive loss

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+eps) #postive here

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss




class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embed_s = Embed(opt.s_dim, opt.feat_dim).to(self.device)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim).to(self.device)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m).to(self.device)
        self.criterion_t = ContrastLoss(opt.n_data).to(self.device)
        self.criterion_s = ContrastLoss(opt.n_data).to(self.device)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """

        # print('f_s: ',f_s.size())
        # print('f_t: ',f_t.size())

        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
