import logging
import math

import numpy as np
import os
import torch
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from utils import utils
from networks.baselines import ewc, hat, cat



                # after training ***********************************************************************************************

def compute(self,model,train_pool_loader, self_fisher, mask_pre, accelerator):
    # copy everyone
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:  # onlyh discriminator is saved. I don't need anything about geenrator
        unwrapped_model = accelerator.unwrap_model(model)
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

    elif 'adapter_hat' in self.args.baseline   \
            or 'adapter_cat' in self.args.baseline \
            or 'adapter_bcl' in self.args.baseline \
            or 'adapter_ctr' in self.args.baseline \
            or 'adapter_classic' in self.args.baseline:
        self.args.s = self.args.smax
        mask = utils.mask(model, accelerator, self.args)
        hat.compute(model, accelerator, mask_pre, mask, self.args)

    return self
        # after training ***********************************************************************************************


