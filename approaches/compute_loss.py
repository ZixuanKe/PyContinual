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
from copy import deepcopy

        # before training ***********************************************************************************************
def compute(self,model,train_loader,outputs,self_fisher,mask_pre,batch,step,accelerator):
    weights_before = None
    self.args.s = (self.args.smax - 1 / self.args.smax) * step / len(
        train_loader) + 1 / self.args.smax
    if 'ewc' in self.args.baseline:
        outputs = model(batch, self_fisher=self_fisher)
    elif 'adapter_hat' in self.args.baseline or 'adapter_cat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_ctr' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        masks = utils.model.mask(model, accelerator, self.args)
        outputs = model(
            batch, masks=masks, mask_pre=mask_pre)
    elif 'mer' in self.args.baseline:
        model_ori = accelerator.unwrap_model(model)
        weights_before = deepcopy(
            model_ori.state_dict())
        outputs = model(batch)
    elif 'lamaml' in self.args.baseline:

        if not (self.args.buffer is None or self.args.buffer.is_empty()) and step % self.args.replay_freq == 0:
            replay_batch = self.args.buffer.get_datadict(
                size=batch['input_ids'].shape[0])
            if self.args.task_name in self.args.classification:
                replay_batch['cls_labels'] = replay_batch['labels']

            for key in batch.keys():
                if key == 'labels' and self.args.task_name in self.args.classification:
                    continue
                batch[key] = torch.cat(
                    (batch[key], replay_batch[key]), dim=0)

        self.fast_weights = self.meta_learner.inner_update(
            self.fast_weights, batch, is_train=True)
        meta_outputs = self.meta_learner.meta_loss(
            self.fast_weights, batch, is_train=True)
        if outputs is None or (step % self.args.meta_task_size == 0):
            outputs = meta_outputs
        else:
            outputs.loss += meta_outputs.loss / \
                            batch['input_ids'].shape[0]

    else:
        outputs = model(batch)

    return self, model, outputs, weights_before