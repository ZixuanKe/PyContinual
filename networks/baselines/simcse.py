import torch
import torch.nn as nn

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        # assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last", "last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs,args):

        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states


        if args.contrast_type == 'add_hard_neg':
            n_sample = last_hidden.size(0)
            naive_last_hidden = last_hidden[:n_sample//3*2]
            hard_last_hidden = last_hidden[n_sample//3*2:]
            attention_mask = attention_mask[:n_sample//3*2]

            # print('naive_last_hidden: ',naive_last_hidden.size())
            # print('hard_last_hidden: ',hard_last_hidden.size())

            if self.pooler_type == "avg":
                naive_pool = ((naive_last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
            if self.pooler_type in ['cls_before_pooler', 'cls']:
                naive_pool = naive_last_hidden[:, 0]
            hard_pool = hard_last_hidden[:,-args.n_tokens,:] # different pool results in different representation

            return torch.cat([naive_pool,hard_pool])



        elif args.contrast_type == 'naive':
            if self.pooler_type in ['cls_before_pooler', 'cls']:
                return last_hidden[:, 0]
            elif self.pooler_type == "avg":
                return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
            elif self.pooler_type == "avg_first_last":
                first_hidden = hidden_states[0]
                last_hidden = hidden_states[-1]
                pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                return pooled_result
            elif self.pooler_type == "avg_top2":
                second_last_hidden = hidden_states[-2]
                last_hidden = hidden_states[-1]
                pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                return pooled_result
            elif self.pooler_type == "last":
                return last_hidden[:, -1]
            else:
                raise NotImplementedError


def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.args.pooler_type
    cls.pooler = Pooler(cls.args.pooler_type)
    if cls.args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.args.temp)
    cls.init_weights()


def sequence_level_contrast(z1=None,z2=None,z3=None,sim=Similarity(temp=0.05)):


    # model gather is put outside because gather cannot deal with the case that tensor has differnt sizes
    # print('z1: ',z1.size())
    # print('z2: ',z2.size())
    # print('z3: ',z3.size())

    cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))
    # print('cos_sim: ',cos_sim.size())

    # Hard negative

    if z3 is not None:
        z1_z3_cos = sim(z1.unsqueeze(1), z3.unsqueeze(0))
        # print('z1_z3_cos: ', z1_z3_cos.size())

        cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)
    # print('cos_sim: ',cos_sim.size())

    labels = torch.arange(cos_sim.size(0)).long().cuda()

    loss_fct = nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels)

    return loss




