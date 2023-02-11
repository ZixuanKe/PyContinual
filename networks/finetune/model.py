"""
    Modified RobertaForSequenceClassification, RobertaForMaskedLM to accept **kwargs in forward.
"""
import pdb
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput, ModelOutput, Seq2SeqLMOutput
import utils
from networks.finetune import classification,generation

class MyModel(nn.Module):

    def __init__(self, model,teacher=None,mlm_model=None,args=None):
        super().__init__()
        #please make sure there is no chanable layers in this class, other than "model"
        self.model = model
        self.teacher = teacher
        self.config = model.config
        self.args = args
        self.sim = None
        self.sigmoid = nn.Sigmoid()
        self.mse_loss = nn.MSELoss()
        self.cos = nn.CosineSimilarity()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)
        self.mlm_model = mlm_model
        self.frequency_table = torch.Tensor([1 for _ in range(args.ntasks)]).float().cuda()
        self.kd_loss =  utils.model.DistillKL(1)



    def forward(self,inputs,
                stage=None,
                self_fisher=None,
                masks=None,
                mask_pre=None,
                prune_model=None,prune_loss=None,head_mask=None,
                only_return_output=False,
                ):


        contrast_loss = None
        sum_loss = None
        logits = None
        ppl = None
        hidden_states= None

        if 'input_ids' in inputs and inputs['input_ids'] is not None:
            input_ids =  inputs['input_ids']
            inputs_embeds = None
        else:
            inputs_embeds = inputs['inputs_embeds']
            input_ids = None

        labels = inputs['labels']
        attention_mask = inputs['attention_mask']

        task = inputs["task"]

        if stage is not None and 'post-train' in stage and 'mlm' in self.args.baseline:
            inputs_ids_mlm = inputs['inputs_ids_mlm']
            labels_mlm = inputs['labels_mlm']
            outputs = self.mlm_model(input_ids=inputs_ids_mlm, labels=labels_mlm, attention_mask=attention_mask,output_hidden_states=True)
            loss = outputs.loss


        elif prune_loss is not None and 'distill' in prune_loss and self.args.task_name in self.args.classification: # detect impt for roberta
            #  use original ids

            outputs = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask,
                                     head_mask=head_mask,
                                     output_hidden_states=True, output_attentions=True,only_return_output=True)
            teacher_outputs = self.teacher(input_ids=input_ids, inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask,
                                           head_mask=head_mask,
                                           output_hidden_states=True, output_attentions=True,only_return_output=True)


            loss = self.kd_loss(teacher_outputs.hidden_states[-1], outputs.hidden_states[-1])  # no need for mean

        # TODO give mask, even if it could be None
        elif self.args.task_name in self.args.classification:
            cls_labels = inputs['cls_labels']
            loss, logits, hidden_states = classification.run_forward(input_ids, attention_mask, task, cls_labels,self,self_fisher,masks, mask_pre,
                                                      inputs_embeds=inputs_embeds,
                                                      head_mask=head_mask,
                                                      only_return_output=only_return_output
                                                      )
        elif self.args.task_name in self.args.generation:
            loss, logits, hidden_states = generation.run_forward(input_ids, attention_mask, task, labels,self,self_fisher,masks, mask_pre,
                                                  inputs_embeds=inputs_embeds,
                                                  prune_model=prune_model, prune_loss=prune_loss, head_mask=head_mask,
                                                  only_return_output=only_return_output
                                                                 )


        return MyRobertaOutput(
            loss = loss,
            contrast_loss = contrast_loss,
            sum_loss = sum_loss,
            logits = logits,
            ppl = ppl,
            hidden_states=hidden_states
        )



class MyRobertaOutput(ModelOutput):
    loss: torch.FloatTensor = None
    contrast_loss: torch.FloatTensor = None
    sum_loss: torch.FloatTensor = None
    logits = None
    past_key_values = None
    decoder_hidden_states = None
    decoder_attentions = None
    cross_attentions = None
    encoder_last_hidden_state = None
    encoder_hidden_states = None
    encoder_attentions = None
    ppl = None
    hidden_states = None
    attentions = None
