from networks.baselines import supsup
import torch
from networks.baselines import ewc, hat
import utils






def run_forward(input_ids,attention_mask,task,labels,my_model,self_fisher,masks=None, mask_pre=None,
                inputs_embeds=None,
                prune_model=None,prune_loss=None,head_mask=None,cross_attn_head_mask=None,output_mask=None, # all for the softmask
                intermediate_mask=None,decoder_head_mask=None,decoder_output_mask=None, decoder_intermediate_mask=None,

                ):

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



        return None,None

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
        if 'cat' in my_model.args.baseline and my_model.args.is_reference: # for CAT
            outputs = my_model.teacher(input_ids=input_ids, inputs_embeds=inputs_embeds,labels=labels, attention_mask=attention_mask,
                                 output_hidden_states=True)

            loss = outputs.loss
            logits = outputs.logits

        elif 'softmask' in my_model.args.baseline and prune_model and 'distill' in prune_loss:
            #TODO: use the output of decoder to conduct distillation, move to here
            # we also need to add some mask
            kd_loss = utils.model.DistillKL(1)

            # inputs_ids_dup = input_ids.repeat(2, 1)
            # labels_dup = labels.repeat(2, 1)
            # attention_mask_dup = attention_mask.repeat(2, 1)

            outputs = my_model.model(input_ids=input_ids, inputs_embeds=inputs_embeds,labels=labels, attention_mask=attention_mask,
                                     head_mask=head_mask,
                                     cross_attn_head_mask=cross_attn_head_mask,
                                     output_mask=output_mask,
                                     intermediate_mask=intermediate_mask,
                                     decoder_head_mask=decoder_head_mask,
                                     decoder_output_mask=decoder_output_mask,
                                     decoder_intermediate_mask=decoder_intermediate_mask,
                                     output_hidden_states=True, output_attentions=True)

            teacher_outputs = my_model.teacher(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                     output_hidden_states=True, output_attentions=True)

            # logits = outputs.logits.view(input_ids.size(0), 2, -1, 50265)
            # logits = outputs.decoder_hidden_states[-1].view(input_ids.size(0), 2, -1, 768)

            # z1 = logits[:, 0]
            # z2 = logits[:, 1]

            # # print('outputs.logits: ',outputs.logits.size())
            # # print('teacher_outputs.logits: ',teacher_outputs.logits.size())
            #
            logits = None
            loss = kd_loss(teacher_outputs.logits, outputs.logits)  # no need for mean
            # loss = kd_loss(z1, z2)  # no need for mean
            # loss = outputs.loss
            # TODO: NAN if no teacher


        else:
            outputs = my_model.model(input_ids=input_ids, inputs_embeds=inputs_embeds,labels=labels, attention_mask=attention_mask,
                                 output_hidden_states=True)

            loss = outputs.loss
            logits = outputs.logits


    if 'ewc' in my_model.args.baseline and my_model.training and self_fisher is not None:
        loss += ewc.loss_compute(my_model, self_fisher)


    elif ('adapter_hat' in my_model.args.baseline or 'adapter_cat' in my_model.args.baseline
            or 'adapter_bcl' in my_model.args.baseline
            or 'adapter_ctr' in my_model.args.baseline
            or 'adapter_classic' in my_model.args.baseline) and my_model.training and not my_model.args.is_cat: # no need for testing

        loss += hat.loss_compute(masks, mask_pre, my_model.args)

    return loss, logits
