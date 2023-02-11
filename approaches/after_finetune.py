import logging
import os
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
from networks.baselines import ewc, hat, cat, ldbr, derpp, agem

                # after training ***********************************************************************************************

def compute(self,model,train_pool_loader, self_fisher, mask_pre, accelerator):
    # copy everyone
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    accelerator.wait_for_everyone() # after training

    if accelerator.is_main_process:  # onlyh discriminator is saved. I don't need anything about geenrator
        unwrapped_model.model.save_pretrained(self.args.output_dir)
        self.args.tokenizer.save_pretrained(self.args.output_dir)
        if 'adapter' in self.args.baseline:
            unwrapped_model.model.save_adapter(self.args.output_dir, 'adapter')
        if 'prompt' in self.args.baseline or 'l2p' in self.args.baseline:
            torch.save(unwrapped_model.model.keys, os.path.join(self.args.output_dir,  'keys'))
            torch.save(unwrapped_model.model.prompt_pool, os.path.join(self.args.output_dir, 'prompt_pool'))
        if 'adapter_cat' in self.args.baseline:
            torch.save(self.similarity.similarities, os.path.join(self.args.output_dir,  'similarities'))

    accelerator.wait_for_everyone() # after training

    if 'ewc' in self.args.baseline:
        ewc.compute(train_pool_loader, model, self_fisher, accelerator, self.args)

    elif 'ldbr' in self.args.baseline:
        train_loader_replay = accelerator.prepare(train_pool_loader)
        ldbr.select_samples_to_store(model.model, self.args.buffer, train_loader_replay, self.args.ft_task)
        torch.save(self.args.buffer, os.path.join(self.args.output_dir, 'buffer'))

    elif 'adapter_hat' in self.args.baseline \
            or 'adapter_cat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_ctr' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        self.args.s = self.args.smax
        mask = utils.mask(model, accelerator, self.args)
        hat.compute(model, accelerator, mask_pre, mask, self.args)

    elif 'derpp' in self.args.baseline or 'agem' in self.args.baseline or 'mer' in self.args.baseline or 'lamaml' in self.args.baseline:
        # add data to the buffer
        train_loader_replay = accelerator.prepare(train_pool_loader)
        if accelerator.is_main_process:  # only find some to keep, no training
            derpp.derpp_ecompute(train_loader_replay, model, self.args.buffer, self.args)
        if 'lamaml' in self.args.baseline:
            torch.save(self.meta_learner.alpha_lr[:], os.path.join(self.args.output_dir, 'alpha_lr'))
        if 'agem' in self.args.baseline:
            torch.save(self.args.grad_dims, os.path.join(self.args.output_dir, 'grad_dims'))
            torch.save(self.args.grad_xy, os.path.join(self.args.output_dir, 'grad_xy'))
            torch.save(self.args.grad_er, os.path.join(self.args.output_dir, 'grad_er'))

    return self
        # after training ***********************************************************************************************


