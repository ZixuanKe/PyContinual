"""
    Modified RobertaForSequenceClassification, RobertaForMaskedLM to accept **kwargs in forward.
"""
import pdb
from tkinter import HIDDEN
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput, ModelOutput, Seq2SeqLMOutput
from utils import utils
from networks import classification,generation

class MyModel(nn.Module):

    def __init__(self, model,teacher=None,args=None):
        super().__init__()
        #please make sure there is no chanable layers in this class, other than "model"
        self.model = model
        self.teacher = teacher
        self.config = model.config
        self.args = args
        self.my_contrastive = utils.MyContrastive()
        self.sim = None
        self.sigmoid = nn.Sigmoid()
        # self.pre_model = copy.deepcopy(self.model) # to large
        self.mse_loss = nn.MSELoss()
        self.cos = nn.CosineSimilarity()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

        self.frequency_table = torch.Tensor([1 for _ in range(args.ntasks)]).float().cuda()

    def sim_matrix(self,a, b, eps=1e-8):
        """Batch version of CosineSimilarity."""
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)

        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    def frequency_norm(self, frequency,eps=5e-5):
        frequency = (frequency - frequency.mean()) / (frequency.std()+eps)  # 2D, we need to deal with this for each layer
        return frequency

    def forward(self,inputs,
                self_fisher=None,
                masks=None,
                mask_pre=None,
                buffer=None,
                subnetwork_importance=None):

        contrast_loss = None
        sum_loss = None
        logits = None
        ppl = None

        input_ids =  inputs['input_ids']
        labels = inputs['labels']
        attention_mask = inputs['attention_mask']

        task = inputs["task"]

        if self.args.task_name in self.args.classification:
            cls_labels = inputs['cls_labels']
            nsp_labels = inputs['nsp_labels'] if 'nsp_labels' in inputs.keys() else None
            loss, logits, hidden_states = classification.run_forward(input_ids, attention_mask, task, cls_labels, self, self_fisher,masks, mask_pre, nsp_labels)
        elif self.args.task_name in self.args.generation:
            loss, logits, hidden_states = generation.run_forward(input_ids, attention_mask, task, labels,self,self_fisher,masks, mask_pre)

        return MyRobertaOutput(
            loss = loss,
            contrast_loss = contrast_loss,
            sum_loss = sum_loss,
            logits = logits,
            ppl = ppl,
            hidden_states = hidden_states
        )



class MyRobertaOutput(ModelOutput):
    loss: torch.FloatTensor = None
    contrast_loss: torch.FloatTensor = None
    sum_loss: torch.FloatTensor = None
    logits = None
    past_key_values = None
    hidden_states = None
    decoder_hidden_states = None
    decoder_attentions = None
    cross_attentions = None
    encoder_last_hidden_state = None
    encoder_hidden_states = None
    encoder_attentions = None
    ppl = None