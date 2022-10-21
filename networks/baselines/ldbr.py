from fileinput import close
import math
import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.cluster import KMeans, MiniBatchKMeans
from networks.roberta import MyRobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaClassificationHeadLDBR(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(128 * 2, config.hidden_size)
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

class MyRobertaForSequenceClassificationLDBR(MyRobertaForSequenceClassification):
    def __init__(self, config, taskcla, args, **kwargs):
        super().__init__(config, taskcla, args)
        self.taskcla = taskcla
        self.config = config
        self.args = args
        self.G = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.Tanh(),
        )
        self.S = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.Tanh()
        )
        # Task id predictor, different from self.classfiers
        self.task_classifier = nn.Linear(128, args.ntasks)
        self.nsp_classifier = nn.Linear(128, args.ntasks)
        self.classifiers = nn.ModuleList() # overwrite !!
        for task, n in taskcla:
            config.num_labels = n
            classifier = RobertaClassificationHeadLDBR(config )
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
        
        sequence_output = outputs[0][:,0,:]
        outputs.hidden_states = outputs[0][:,0,:]

        loss = 0
        logits = None

        ## for ldbr, we must provide task id to do Task Id prediction!!!
        general_knowledge = self.G(sequence_output)
        specific_knowledge = self.S(sequence_output) # size = (B, 128)
        total_knowledge = torch.cat([general_knowledge, specific_knowledge], dim=-1) # (B, 256)
        loss_fct = nn.CrossEntropyLoss()
        logits = []
        if labels is not None and task is not None:
            for t_id, t in enumerate(task):  # different task in the same batch
                logit = self.classifiers[t](total_knowledge[t_id].unsqueeze(0))
                num_labels = self.taskcla[t][1]
                cur_loss = loss_fct(logit.view(-1, num_labels), labels[t_id].view(-1))
                loss += cur_loss
                logits.append(logit)

            task_predict_logit = self.task_classifier(specific_knowledge)
            specific_loss = loss_fct(task_predict_logit.view(-1, self.args.ntasks), task.view(-1))
            loss += specific_loss
        else:
            logits = []
            for t in range(self.args.ntasks):
                logit = self.classifiers[t](total_knowledge)
                logits.append(logit)

        logits = (logits, specific_knowledge, general_knowledge)

        if nsp_labels is not None:
            nsp_logits = self.nsp_classifier(general_knowledge)
            general_loss = loss_fct(nsp_logits.view(-1, self.args.ntasks), nsp_labels.long().view(-1))
            loss += general_loss

        if my_loss is not None:
            loss += my_loss

        return SequenceClassifierOutput(
            loss=loss / input_ids.shape[0],
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
def regularization(ouputs, teacher_outputs):
    # TODO: the name "logits" is not suitable, actually they are hidden representations
    g_outputs, s_outputs = ouputs.logits[-1], ouputs.logits[-2]
    g_teacher_outputs, s_teacher_outputs = teacher_outputs.logits[-1], teacher_outputs.logits[-2]
    loss_func = nn.MSELoss()
    return (loss_func(g_outputs, g_teacher_outputs) + loss_func(s_outputs, s_teacher_outputs)) * 0.5


def select_samples_to_store(model, buffer, data_loader, task_id):

    x_list = []
    mask_list = []
    y_list = []
    fea_list = []
    nsp_list = []

    model.eval()
    with torch.no_grad():
        for inputs in data_loader:
            x = inputs['input_ids']
            mask = inputs['attention_mask']
            y = inputs['cls_labels']
            nsp_labels = inputs['nsp_labels']
            x = x.cuda()
            mask = mask.cuda()
            y = y.cuda()
            outputs = model(
                input_ids = x, 
                attention_mask = mask,
            )
            roberta_emb = outputs.hidden_states
            x_list.append(x.to("cpu"))
            mask_list.append(mask.to("cpu"))
            y_list.append(y.to("cpu"))
            nsp_list.append(nsp_labels.to('cpu'))
            # Kmeans on bert embedding
            fea_list.append(roberta_emb.to("cpu"))
    x_list = torch.cat(x_list, dim=0).data.cpu().numpy()
    mask_list = torch.cat(mask_list, dim=0).data.cpu().numpy()
    y_list = torch.cat(y_list, dim=0).data.cpu().numpy()
    fea_list = torch.cat(fea_list, dim=0).data.cpu().numpy()
    nsp_list = torch.cat(nsp_list, dim=0).data.cpu().numpy()
    n_clu = model.args.buffer_size_per_dataset
    estimator = KMeans(n_clusters=n_clu, random_state=2022)
    estimator.fit(fea_list)
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_

    examples = []
    labels = []
    attention_mask = []
    task = []
    nsp_labels = []
    for clu_id in range(n_clu):
        index = [i for i in range(len(label_pred)) if label_pred[i] == clu_id]
        closest = float("inf")
        closest_x = None
        closest_mask = None
        closest_y = None
        closest_nsp_label = None
        for j in index:
            dis = np.sqrt(np.sum(np.square(centroids[clu_id] - fea_list[j])))
            if dis < closest:
                closest_x = x_list[j]
                closest_mask = mask_list[j]
                closest_y = y_list[j]
                closest_nsp_label = nsp_list[j]
                closest = dis

        if closest_x is not None:
            examples.append(closest_x)
            labels.append(closest_y)
            attention_mask.append(closest_mask)
            task.append(task_id)
            nsp_labels.append(closest_nsp_label)

    buffer.add_data(
        torch.tensor(examples).to(buffer.device), 
        attention_mask=torch.tensor(attention_mask).to(buffer.device), 
        labels=torch.tensor(labels).to(buffer.device), 
        task=torch.tensor(task).to(buffer.device),
        nsp_labels=torch.tensor(nsp_labels).to(buffer.device)
    )
    
    print("Buffer size:{}".format(len(buffer)))

def process_dataset(dataset, tokenizer):
    import random
    def process_one(inp):
        sent = inp.split(' ')
        sep_token = tokenizer.sep_token
        if sep_token in sent: # asc
            sep_place = sent.index(sep_token)
        else:
            sep_place = len(sent) - 1
        if sep_place == 1:
            ## nsp requires sentence length more than 1 !!!!
            sent = [sent[0], tokenizer.sep_token, sent[1], sent[2]]
            sep_place = 2
            
        place = random.randint(1, sep_place - 1)
        mask = random.random() > 0.5
        
        if mask:    # label = 1, normal sequence
            sent = sent[:place] + [tokenizer.sep_token] + sent[place:sep_place] + sent[sep_place:]
        else:       # label = 0, abnormal sequence
            sent = sent[place:sep_place] + [tokenizer.sep_token] + sent[:place] + sent[sep_place:]
        
        inp = ' '.join(inp)
        return inp, mask

    inputs, nsp_label = [], []
    for sent in dataset['source']:
        inp, mask = process_one(sent)
        inputs.append(inp)
        nsp_label.append(mask)

    dataset = dataset.add_column("nsp_labels", nsp_label)
    dataset = dataset.remove_columns('source')
    dataset = dataset.add_column("source", inputs)
    return dataset