from networks.transformers.bart import MyBartForConditionalGeneration,MyBartForSequenceClassification,MyBartForTokenClassification,MyBartEncoder
import torch
from transformers.models.bart.configuration_bart import BartConfig
from torch import nn
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput



class BARTSoftPromptMixinGeneration:

    @torch.no_grad()
    def generate(self, **kwargs):

        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        kwargs["input_ids"] = None

        return super().generate(**kwargs)





class MyBartForConditionalGenerationSoftPromptInfer(BARTSoftPromptMixinGeneration, MyBartForConditionalGeneration):
    def __init__(self, config, **kwargs):
        BARTSoftPromptMixinGeneration.__init__(self)
        MyBartForConditionalGeneration.__init__(self, config, **kwargs)

