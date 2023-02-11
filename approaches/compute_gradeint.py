import logging
import torch
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
import utils

        # before training ***********************************************************************************************
def compute(self,model,loss,mask_back,weights_before,epoch,batch,step,accelerator):
    # replay here
    model_ori = accelerator.unwrap_model(model)

    if ('ldbr' in self.args.baseline or 'derpp' in self.args.baseline or 'mer' in self.args.baseline) \
            and not (self.args.buffer is None or self.args.buffer.is_empty()) \
            and step % self.args.replay_freq == 0:

        replay_batch = self.args.buffer.get_datadict(
            size=batch['input_ids'].shape[0])
        if self.args.task_name in self.args.classification:
            replay_batch['cls_labels'] = replay_batch['labels']
        replay_outputs = model(replay_batch)

        loss += replay_outputs.loss * self.args.replay_beta
        if 'derpp' in self.args.baseline:
            loss += self.mse(
                replay_outputs.hidden_states[-1], replay_batch['logits']) * self.args.replay_alpha
    # We keep track of the loss at each epoch
    loss = loss / self.args.gradient_accumulation_steps

    if 'lamaml' not in self.args.baseline or (step + 1) % self.args.meta_task_size == 0:
        accelerator.backward(loss)

    if accelerator.is_main_process and epoch < 1 and step < 1:
        for n, p in model.named_parameters():
            if p.grad is not None:
                print('n,p,： ', n, p.size())

    if self.args.ft_task > 0 and \
            ('adapter_hat' in self.args.baseline
             or 'adapter_cat' in self.args.baseline
             or 'adapter_bcl' in self.args.baseline
             or 'adapter_ctr' in self.args.baseline
             or 'adapter_classic' in self.args.baseline):
        for n, p in model.named_parameters():
            if n in mask_back and p.grad is not None:
                # print('block n: ',mask_back[n])
                # block  n: module.model.model.decoder.layers.9.output_adapters.adapters.adapter.adapter_down.linear_down.adapters.fc1.weight
                p.grad.data *= mask_back[n]
            elif n in self.tsv_para and p.grad is not None and 'hat' not in self.args.baseline:
                # open for general
                p.grad.data *= utils.get_view_for_tsv(
                    n, model_ori, self.args)
            # elif 'adapter_cat' in self.args.baseline: #TODO: to open the mask, utils.mask already consder it
            #     p.grad.data *= utils.get_similar_mask(self.args.similarity, model, accelerator, self.args)

    if 'agem' in self.args.baseline:
        model_ori = accelerator.unwrap_model(model)
        from networks.baselines import agem
        if not (self.args.buffer is None or self.args.buffer.is_empty()):
            agem.store_grad(
                model_ori.model.parameters, self.args.grad_xy, self.args.grad_dims)
            model_ori.model.zero_grad()

            replay_batch = self.args.buffer.get_datadict(
                self.args.buffer_size_per_dataset)
            # TODO: consider data loader if needed for efficient
            replay_batch['cls_labels'] = replay_batch['labels']
            outputs = model_ori(replay_batch)
            # also make it cannot deal with task with different #classes, as DREPP
            accelerator.backward(outputs.loss)

            agem.store_grad(
                model_ori.model.parameters, self.args.grad_er, self.args.grad_dims)
            #
            dot_prod = torch.dot(
                self.args.grad_xy, self.args.grad_er)
            if dot_prod.item() < 0:
                g_tilde = agem.project(
                    gxy=self.args.grad_xy, ger=self.args.grad_er)
                agem.overwrite_grad(
                    model_ori.model.parameters, g_tilde, self.args.grad_dims)
            else:
                agem.overwrite_grad(
                    model_ori.model.parameters, self.args.grad_xy, self.args.grad_dims)

    if 'mer' in self.args.baseline:
        # Within batch Reptile meta-update:
        model_ori = accelerator.unwrap_model(model)
        weights_after = model_ori.state_dict()
        model_ori.load_state_dict(
            {
                name: weights_before[name] + (
                        (weights_after[name] - weights_before[name]) * self.args.mer_beta)
                for name in weights_before
            }
        )

    if 'adapter_hat' in self.args.baseline \
            or 'adapter_cat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_ctr' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        # Compensate embedding gradients
        for n, p in model.named_parameters():
            if ('adapters.e' in n or 'model.e' in n) and p.grad is not None:
                num = torch.cosh(
                    torch.clamp(self.args.s * p.data, -self.args.thres_cosh,
                                self.args.thres_cosh)) + 1
                den = torch.cosh(p.data) + 1
                p.grad.data *= self.args.smax / self.args.s * num / den

    return model