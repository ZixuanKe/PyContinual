from networks.baselines import supsup
from networks.baselines import ewc, hat, ldbr
import torch

def run_forward(input_ids,attention_mask,task,cls_labels,my_model,self_fisher,masks=None, mask_pre=None, nsp_labels=None):

        if 'supsup' in my_model.args.baseline:
            if 'mtl' in my_model.args.baseline: # these are only useful for supsup
                supsup.set_model_sim(my_model.model, 'both')  # if nothing
                supsup.set_model_specific_task(my_model, task)  # in case nothing is used
                supsup.set_model_share_task(my_model, 0)  # alwasy use the same, as shared knwoeldeg accorss all
            elif 'ncl' in my_model.args.baseline:
                supsup.set_model_sim(my_model.model, 'specific')  # if nothing
                supsup.set_model_specific_task(my_model, 0)  # alwasys use the same

            else:
                supsup.set_model_sim(my_model.model, 'specific')  # if nothing

                if 'forward' in my_model.args.baseline:
                    task_dup = task.repeat(2)
                    supsup.set_model_specific_task(my_model,task_dup)  # in case nothing is used
                else:
                    supsup.set_model_specific_task(my_model, task)  # in case nothing is used

        else:
            if my_model.args.is_reference:
                outputs = my_model.teacher(input_ids=input_ids, labels=cls_labels, attention_mask=attention_mask, output_hidden_states=True, task=task, nsp_labels=nsp_labels)
            else:
                outputs = my_model.model(input_ids=input_ids, labels=cls_labels, attention_mask=attention_mask, output_hidden_states=True, task=task, nsp_labels=nsp_labels)

            loss = outputs.loss
            logits = outputs.logits
            hidden_states = outputs.hidden_states

        if 'ewc' in my_model.args.baseline and my_model.training and self_fisher is not None:  # only if we are training

            loss += ewc.loss_compute(my_model,self_fisher)

        elif 'ldbr' in my_model.args.baseline and my_model.training and my_model.args.ft_task > 0:
            
            with torch.no_grad():
                teacher_outputs = my_model.teacher(input_ids=input_ids, labels=cls_labels, attention_mask=attention_mask, output_hidden_states=True, task=task, nsp_labels=nsp_labels)
            loss += ldbr.regularization(outputs, teacher_outputs)

        elif ('adapter_hat' in my_model.args.baseline or 'adapter_cat' in my_model.args.baseline
                or 'adapter_bcl' in my_model.args.baseline
                or 'adapter_ctr' in my_model.args.baseline
                or 'adapter_classic' in my_model.args.baseline) and my_model.training and not my_model.args.is_cat: # no need for testing

            loss += hat.loss_compute(masks,mask_pre,my_model.args)

        return loss, logits, hidden_states

