from networks.baselines import supsup
import torch
from networks.baselines import ewc, hat






def run_forward(input_ids,attention_mask,task,labels,my_model,self_fisher,masks=None, mask_pre=None,):

    if not my_model.training:  # must be if 'l2p' in my_model.args.baseline
        #TODO: Pool is not training, but why???
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, my_model.args.num_beams).view(-1).to(
                input_ids.device)
        )  # same as the beam search
        task = task.index_select(0, expanded_return_idx)

        if 'supsup' in my_model.args.baseline:
            if 'mtl' in my_model.args.baseline:
                supsup.set_model_sim(my_model.model, 'both')  # if nothing
                supsup.set_model_specific_task(my_model, task)  # I can easily know what task it is by looking at task
                supsup.set_model_share_task(my_model, 0)  # alwasy use the same, as shared knwoeldeg accorss all
            elif 'ncl' in my_model.args.baseline:
                supsup.set_model_sim(my_model.model, 'specific')  # if nothing
                supsup.set_model_specific_task(my_model, 0)  # alwasys use the same
            else:
                supsup.set_model_sim(my_model.model, 'specific')  # if nothing
                if 'ggg' in my_model.args.baseline:
                    supsup.set_model_specific_task(my_model, task)  # I can easily know what task it is
                else:
                    supsup.set_model_specific_task(my_model, 'None')  # we don't know the id



        return None,None,None

    # TODO: bellow for training -------------------------------------

    if 'supsup' in my_model.args.baseline:
        if 'mtl' in my_model.args.baseline:

            supsup.set_model_sim(my_model.model, 'both')  # if nothing
            supsup.set_model_specific_task(my_model, task)  # in case nothing is used
            supsup.set_model_share_task(my_model, 0)  # alwasy use the same, as shared knwoeldeg accorss all
        elif 'ncl' in my_model.args.baseline:
            supsup.set_model_sim(my_model.model, 'specific')  # if nothing
            supsup.set_model_specific_task(my_model, 0)  # alwasys use the same
        else:
            supsup.set_model_sim(my_model.model, 'specific')  # if nothing
            supsup.set_model_specific_task(my_model, task)  # in case nothing is used

    else:
        if my_model.args.is_reference:
            outputs = my_model.teacher(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                 output_hidden_states=True)
        else:
            outputs = my_model.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                 output_hidden_states=True)

        loss = outputs.loss
        logits = outputs.logits
        hidden_states = outputs.encoder_hidden_states    # TODO: to consistent with classification


    if 'ewc' in my_model.args.baseline and my_model.training and self_fisher is not None:
        loss += ewc.loss_compute(my_model, self_fisher)


    elif ('adapter_hat' in my_model.args.baseline or 'adapter_cat' in my_model.args.baseline
            or 'adapter_bcl' in my_model.args.baseline
            or 'adapter_ctr' in my_model.args.baseline
            or 'adapter_classic' in my_model.args.baseline) and my_model.training and not my_model.args.is_cat: # no need for testing

        loss += hat.loss_compute(masks, mask_pre, my_model.args)

    return loss, logits, hidden_states
