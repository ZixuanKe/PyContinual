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

def update(self,model,optimizer,outputs,loss,writer,lr_scheduler,progress_bar,global_step,completed_steps,epoch,step,accelerator):
    if 'lamaml' in self.args.baseline:
        if (step + 1) % self.args.meta_task_size == 0:
            self.meta_learner.step_and_zero_grad()
            self.fast_weights = None
            optimizer.zero_grad()
        lr_scheduler.step()
        progress_bar.update(1)
        global_step += 1
        completed_steps += 1
    else:
        optimizer.step()
        global_step += 1
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        completed_steps += 1
    progress_bar.set_description(
        'Train Iter (Epoch=%3d,loss=%5.3f)' % ((epoch, loss.item())))  # show the loss, mean while

    if 'adapter_hat' in self.args.baseline \
            or 'adapter_cat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_ctr' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        # Constrain embeddings
        for n, p in model.named_parameters():
            if 'adapters.e' in n or 'model.e' in n:
                p.data = torch.clamp(
                    p.data, -self.args.thres_emb, self.args.thres_emb)

    if accelerator.is_main_process:
        utils.util.log_loss(
            writer, scalar_value=loss.item(), global_step=global_step)
        if outputs.sum_loss is not None:
            utils.util.log_loss(writer, loss_name=' summerization loss', scalar_value=outputs.sum_loss.item(
            ), global_step=global_step)
        if outputs.contrast_loss is not None:
            utils.util.log_loss(writer, loss_name=' contrast loss', scalar_value=outputs.contrast_loss.item(
            ), global_step=global_step)


    return