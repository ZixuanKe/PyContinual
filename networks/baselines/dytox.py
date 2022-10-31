from fileinput import close
import math
import torch
import torch.nn as nn
import numpy as np
import json
from networks.roberta import MyRobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

class RobertaClassificationHeadDyTox(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class DyToxTAB(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.args = args

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.layernorm1 = nn.LayerNorm([self.args.max_length + 1, config.hidden_size])
        self.layernorm2 = nn.LayerNorm([self.all_head_size])
        self.MLP = nn.Linear(self.all_head_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        theta,
        x,
    ):
        '''
            x: [batch_size, seq_len, hidden_size]
            theta: [batch_size, hidden_size] 
            z: [batch_size, seq_len+1, hidden_size]
            query_layer: [batch_size, 1, head_num, head_size]
            key, value_layer: [batch_size, seq_len+1, head_num, head_size]
            atten_probs: [batch_size, 1, head_num, head_num]
            context_layer: [batch_size, 1, head_num, head_size] -> [batch_size, all_head_size]
        '''
        theta = theta.unsqueeze(1)
        z = self.layernorm1(torch.cat([x, theta], dim=1))

        query_layer = self.transpose_for_scores(self.query(theta))
        key_layer = self.transpose_for_scores(self.key(z))  
        value_layer = self.transpose_for_scores(self.value(z))
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = context_layer.squeeze(1)
        outputs = self.MLP(outputs)

        return outputs

class MyRobertaForSequenceClassificationDyTox(MyRobertaForSequenceClassification):
    def __init__(self, config, taskcla, args, **kwargs):
        super().__init__(config, taskcla, args)
        self.taskcla = taskcla
        self.config = config
        self.args = args
        ## self.roberta is SAB, self.classifiers is clf
        self.task_embedder = nn.Embedding(args.ntasks, config.hidden_size)
        self.TAB = DyToxTAB(config, args)
        self.classifiers = nn.ModuleList() # overwrite !!
        for _, n in taskcla:
            config.num_labels = n
            classifier = RobertaClassificationHeadDyTox(config)
            self.classifiers.append(classifier)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
        my_loss=None,
        nsp_labels=None,
    ):

        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if task is not None:
            theta = self.task_embedder(task)
        else:
            theta = self.task_embedder(torch.ones(input_ids.size(0), device=input_ids.device) * self.ft_task)

        sequence_output = self.TAB(theta=theta, x=outputs[0])   # outputs should be the same as 

        loss = 0
        logits = None

        if labels is not None and task is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = []
            for t_id, t in enumerate(task):  # different task in the same batch
                logit = self.classifiers[t](sequence_output[t_id].unsqueeze(0)) #no stack can be performed
                num_labels = self.taskcla[t][1]
                cur_loss = loss_fct(logit.view(-1, num_labels), labels[t_id].view(-1))
                loss += cur_loss
                logits.append(logit)

            if 'mtl' not in self.args.baseline and 'comb' not in self.args.baseline:
                logits = torch.cat(logits)

            loss = loss / len(task)

        else:
            logits = []
            for t in range(self.args.ntasks):
                logit = self.classifiers[t](sequence_output)
                logits.append(logit)
            logits = torch.cat(logits,dim=1)


        if my_loss is not None:
            loss += my_loss

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
