# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers import BertConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"


# Change log ------------------------------------
# above copy from trasnformers == 4.10.2
# We add adapters in each layers
# we add attribute in conig, see configuration_utils.py.py
# we add argument in apply_chunking_to_forward, for some layers, see modeling_utils.py
# the code for adapter is in networks.adapters


# Change log ------------------------------------
import sys
from transformers.models.bert.modeling_bert import BertModel,BertSelfOutput,BertSelfOutput,BertEncoder,BertOutput,BertLayer,BertAttention
sys.path.append("./networks/base/")
from adapters import BertAdapterMask,BertAdapter,BertAdapterUcl, BertAdapterOwm



class MyBertSelfOutput(BertSelfOutput):
    def __init__(self, config,args):
        super().__init__(config)

        if args.use_imp:
            from networks.base.adapters import BertAdapterCapsuleMaskImp as BertAdapterCapsuleMask
            from networks.base.adapters import BertAdapterCapsuleImp as BertAdapterCapsule
        else:
            from networks.base.adapters import BertAdapterCapsuleMask
            from networks.base.adapters import BertAdapterCapsule


        if args.apply_bert_attention_output:
            print('apply to attention')
            if args.build_adapter:
                self.adapter = BertAdapter(args)
            if args.build_adapter_ucl:
                self.adapter_ucl = BertAdapterUcl(args)
            if args.build_adapter_owm:
                self.adapter_owm = BertAdapterOwm(args)
            elif args.build_adapter_mask:
                self.adapter_mask = BertAdapterMask(args)
            elif args.build_adapter_capsule_mask:
                self.adapter_capsule_mask = BertAdapterCapsuleMask(args)
            elif args.build_adapter_capsule:
                self.adapter_capsule = BertAdapterCapsule(args)

        self.args = args

    def forward(self, hidden_states, input_tensor,**kwargs):

        # add parameters --------------
        s,t,x_list,h_list=None,None,None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        if 'x_list' in kwargs: x_list = kwargs['x_list']
        if 'h_list' in kwargs: h_list = kwargs['h_list']
        # other parameters --------------

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.args.apply_bert_attention_output:
            if self.args.build_adapter:
                hidden_states = self.adapter(hidden_states)
            elif self.args.build_adapter_ucl:
                hidden_states = self.adapter_ucl(hidden_states)
            elif self.args.build_adapter_owm:
                output_dict = self.adapter_owm(hidden_states)
                hidden_states = output_dict['outputs']
                x_list = output_dict['x_list']
                h_list = output_dict['h_list']
            elif self.args.build_adapter_mask:
                hidden_states = self.adapter_mask(hidden_states,t,s)
            elif self.args.build_adapter_capsule_mask:
                output_dict = self.adapter_capsule_mask(hidden_states,t,s)
                hidden_states = output_dict['outputs']
            elif self.args.build_adapter_capsule:
                hidden_states = self.adapter_capsule(hidden_states,t,s)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        if self.args.apply_bert_attention_output:
            if self.args.build_adapter_capsule_mask: return {'outputs':hidden_states}
            elif self.args.build_adapter_owm: return {'outputs':hidden_states,'x_list':x_list,'h_list':h_list}
            else: return hidden_states
        else: return hidden_states

class MyBertAttention(BertAttention):
    def __init__(self, config,args):
        super().__init__(config)
        self.output = MyBertSelfOutput(config,args)
        self.self = BertSelfAttention(config)
        self.args = args
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,**kwargs):

        # add parameters --------------
        s,t,x_list,h_list=None,None,None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        if 'x_list' in kwargs: x_list = kwargs['x_list']
        if 'h_list' in kwargs: h_list = kwargs['h_list']
        # other parameters --------------


        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions
        )
        if self.args.apply_bert_attention_output:
            if self.args.build_adapter_capsule_mask:
                output_dict = self.output(self_outputs[0], hidden_states,
                                               t=t,s=s)
                attention_output = output_dict['outputs']

            elif self.args.build_adapter_owm:
                output_dict = self.output(self_outputs[0], hidden_states)
                attention_output = output_dict['outputs']
                x_list = output_dict['x_list']
                h_list = output_dict['h_list']

            elif self.args.build_adapter_capsule:
                attention_output = self.output(self_outputs[0], hidden_states,t=t,s=s)
            else:
                attention_output = self.output(self_outputs[0], hidden_states, t=t,s=s)
        else:
            attention_output = self.output(self_outputs[0], hidden_states,t=t,s=s)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        if self.args.apply_bert_attention_output:
            if self.args.build_adapter_capsule_mask: return {'outputs':outputs}
            elif self.args.build_adapter_owm: return {'outputs':outputs,'x_list':x_list,'h_list':h_list}
            else: return outputs
        else: return outputs


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class MyBertOutput(BertOutput):
    def __init__(self, config,args):
        super().__init__(config)

        if args.use_imp:
            from networks.base.adapters import BertAdapterCapsuleMaskImp as BertAdapterCapsuleMask
            from networks.base.adapters import BertAdapterCapsuleImp as BertAdapterCapsule
        else:
            from networks.base.adapters import BertAdapterCapsuleMask
            from networks.base.adapters import BertAdapterCapsule


        if args.apply_bert_output:
            print('apply to output')
            if args.build_adapter:
                self.adapter = BertAdapter(args)
            elif args.build_adapter_ucl:
                self.adapter_ucl = BertAdapterUcl(args)
            elif args.build_adapter_owm:
                self.adapter_owm = BertAdapterOwm(args)
            elif args.build_adapter_mask:
                self.adapter_mask = BertAdapterMask(args)
            elif args.build_adapter_capsule_mask:
                self.adapter_capsule_mask = BertAdapterCapsuleMask(args)
            elif args.build_adapter_capsule:
                self.adapter_capsule = BertAdapterCapsule(args)

        self.args = args

    def forward(self, hidden_states, input_tensor,**kwargs):

        # add parameters --------------
        s,t,x_list,h_list=None,None,None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        if 'x_list' in kwargs: x_list = kwargs['x_list']
        if 'h_list' in kwargs: h_list = kwargs['h_list']
        # other parameters --------------


        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.args.apply_bert_output:
            if self.args.build_adapter:
                hidden_states = self.adapter(hidden_states)
            elif self.args.build_adapter_ucl:
                hidden_states = self.adapter_ucl(hidden_states)

            elif self.args.build_adapter_owm:
                output_dict = self.adapter_owm(hidden_states)
                hidden_states = output_dict['outputs']
                x_list = output_dict['x_list']
                h_list = output_dict['h_list']

            elif self.args.build_adapter_mask:
                hidden_states = self.adapter_mask(hidden_states,t,s)

            elif self.args.build_adapter_capsule_mask:
                output_dict = self.adapter_capsule_mask(hidden_states,t,s)
                hidden_states=output_dict['outputs']

            elif self.args.build_adapter_capsule:
                hidden_states = self.adapter_capsule(hidden_states,t,s)


        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        if self.args.apply_bert_output:
            if self.args.build_adapter_capsule_mask: return {'outputs':hidden_states}
            elif self.args.build_adapter_owm: return {'outputs':hidden_states,'x_list':x_list,'h_list':h_list}
            else: return hidden_states
        else:
            return hidden_states


class MyBertLayer(BertLayer):
    def __init__(self, config,args):
        super().__init__(config)
        self.attention = MyBertAttention(config,args)
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = MyBertAttention(config,args)
        self.output = MyBertOutput(config,args)
        self.args = args
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,**kwargs
    ):

        # add parameters --------------
        s,t,x_list,h_list=None,None,None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        if 'x_list' in kwargs: x_list = kwargs['x_list']
        if 'h_list' in kwargs: h_list = kwargs['h_list']
        # other parameters --------------

        if self.args.apply_bert_attention_output:
            if self.args.build_adapter_owm:
                output_dict = self.attention(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    output_attentions=output_attentions
                )
                self_attention_outputs = output_dict['outputs']
                x_list = output_dict['x_list']
                h_list = output_dict['h_list']

            elif self.args.build_adapter_capsule_mask:
                output_dict = self.attention(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    output_attentions=output_attentions,
                    t=t,s=s,
                )
                self_attention_outputs= output_dict['outputs']

            elif self.args.build_adapter_capsule:
                self_attention_outputs = self.attention(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    output_attentions=output_attentions,
                    t=t,s=s,
                )
            else:
                self_attention_outputs = self.attention(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    output_attentions=output_attentions,
                    t=t,s=s
                )
        else:

            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                t=t,s=s,
            )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        if self.args.apply_bert_output:
            if self.args.build_adapter_capsule_mask:
                output_dict = apply_chunking_to_forward(
                    self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output,
                    t=t,s=s,
                )
                layer_output=output_dict['outputs']

            elif self.args.build_adapter_owm:
                output_dict = apply_chunking_to_forward(
                    self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
                layer_output=output_dict['outputs']
                h_list.append(output_dict['h_list'])
                x_list.append(output_dict['x_list'])

            elif self.args.build_adapter_capsule:
                layer_output = apply_chunking_to_forward(
                    self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output,
                    t=t,s=s,
                )
            else:
                layer_output = apply_chunking_to_forward(
                    self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output,
                    t=t,s=s,
                )
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output,
                t=t,s=s,
            )

        outputs = (layer_output,) + outputs
        if self.args.apply_bert_output or self.args.apply_bert_attention_output:
            if self.args.build_adapter_capsule_mask: return {'outputs':outputs}
            elif self.args.build_adapter_owm: return {'outputs':outputs,'x_list':x_list,'h_list':h_list}
            else: return outputs
        else: return outputs

    def feed_forward_chunk(self, attention_output,
                           t=None,s=1,):
        intermediate_output = self.intermediate(attention_output)

        if self.args.apply_bert_output:
            if self.args.build_adapter_owm:
                output_dict = self.output(intermediate_output, attention_output)
                layer_output=output_dict['outputs']
                h_list = output_dict['h_list']
                x_list = output_dict['x_list']

            elif self.args.build_adapter_capsule_mask:
                output_dict = self.output(intermediate_output, attention_output,
                                           t=t,s=s,)
                layer_output=output_dict['outputs']

            elif self.args.build_adapter_capsule:
                layer_output = self.output(intermediate_output, attention_output,
                                           t=t,s=s,)

            else:
                layer_output = self.output(intermediate_output, attention_output,
                                           t=t,s=s,)

        else:
            layer_output = self.output(intermediate_output, attention_output,
                                       t=t,s=s,)
        if self.args.apply_bert_output:
            if self.args.build_adapter_capsule_mask: return {'outputs':layer_output}
            elif self.args.build_adapter_owm: return {'outputs':layer_output,'x_list':x_list,'h_list':h_list}
            else: return layer_output
        else: return layer_output


class MyBertEncoder(BertEncoder):
    def __init__(self, config,args):
        super().__init__(config)
        self.layer = nn.ModuleList([MyBertLayer(config,args) for _ in range(config.num_hidden_layers)])
        self.args = args
    def compute_layer_outputs(self,
                              output_attentions,layer_module,
                              hidden_states,attention_mask,layer_head_mask,
                              encoder_hidden_states,encoder_attention_mask,**kwargs):

        # add parameters --------------

        s,t,x_list,h_list=None,None,None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        if 'x_list' in kwargs: x_list = kwargs['x_list']
        if 'h_list' in kwargs: h_list = kwargs['h_list']
        # other parameters --------------



        if getattr(self.config, "gradient_checkpointing", False):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:

            if self.args.build_adapter_owm:
                output_dict = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions
                )
                layer_outputs = output_dict['outputs']
                x_list.append(output_dict['x_list'])
                h_list.append(output_dict['h_list'])

            elif self.args.build_adapter_capsule_mask:
                output_dict = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    t=t,s=s
                )
                layer_outputs = output_dict['outputs']

            elif self.args.build_adapter_capsule:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    t=t,s=s
                )

            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    t=t,s=s
                )

        return layer_outputs,x_list,h_list

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,**kwargs
    ):


        # add parameters --------------
        s,t,x_list,h_list=None,None,None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        if 'x_list' in kwargs: x_list = kwargs['x_list']
        if 'h_list' in kwargs: h_list = kwargs['h_list']
        # other parameters --------------

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None


        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs,x_list,h_list = self.compute_layer_outputs(
                          output_attentions,layer_module,
                          hidden_states,attention_mask,layer_head_mask,
                          encoder_hidden_states,encoder_attention_mask,
                          t=t,s=s,x_list=x_list,h_list=h_list
                        )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)


        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            if self.args.build_adapter_capsule_mask: return {'outputs':tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)}
            elif self.args.build_adapter_owm: return {'outputs':tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None),'h_list':h_list,'x_list':x_list}
            else: return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions,cross_attentions=all_cross_attentions
        )


class MyBertModel(BertModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, args,add_pooling_layer=True):
        super().__init__(config)
        self.args = args
        self.encoder = MyBertEncoder(config,args)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,**kwargs
    ):


        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """

        # add parameters --------------
        s,t,x_list,h_list=None,None,None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        if 'x_list' in kwargs: x_list = kwargs['x_list']
        if 'h_list' in kwargs: h_list = kwargs['h_list']
        # other parameters --------------

        x_list = [] #accumulate for every forward pass
        h_list = []

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids,  token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs,x_list,h_list = self.compute_encoder_outputs(
                                embedding_output,extended_attention_mask,head_mask,
                                encoder_hidden_states,encoder_extended_attention_mask,output_attentions,
                                output_hidden_states,return_dict,t=t,s=s,x_list=x_list,h_list=h_list)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            if self.args.build_adapter_capsule_mask: return {'outputs':(sequence_output, pooled_output) + encoder_outputs[1:]}
            elif self.args.build_adapter_owm: return {'outputs':(sequence_output, pooled_output) + encoder_outputs[1:],'x_list':x_list,'h_list':h_list}
            else: return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def compute_encoder_outputs(self,
                                embedding_output,extended_attention_mask,head_mask,
                                encoder_hidden_states,encoder_extended_attention_mask,output_attentions,
                                output_hidden_states,return_dict,**kwargs):


        # add parameters --------------
        s,t,x_list,h_list=None,None,None,None
        if 't' in kwargs: t = kwargs['t']
        if 's' in kwargs: s = kwargs['s']
        if 'x_list' in kwargs: x_list = kwargs['x_list']
        if 'h_list' in kwargs: h_list = kwargs['h_list']
        # other parameters --------------

        if self.args.build_adapter_owm:
            output_dict = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,x_list=x_list,h_list=h_list
            )

            encoder_outputs=output_dict['outputs']
            x_list=output_dict['x_list']
            h_list=output_dict['h_list']

        elif self.args.build_adapter_capsule_mask:
            output_dict = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict, t=t,s=s
            )

            encoder_outputs=output_dict['outputs']

        elif self.args.build_adapter_capsule:
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict, t=t,s=s,
            )
        else:
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,t=t,s=s
            )

        return encoder_outputs,x_list,h_list



from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors,**kwargs
) -> torch.Tensor:
    """
    This function chunks the :obj:`input_tensors` into smaller input tensor parts of size :obj:`chunk_size` over the
    dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to each chunk independently to save memory.

    If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this function will yield the same result as
    directly applying :obj:`forward_fn` to :obj:`input_tensors`.

    Args:
        forward_fn (:obj:`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (:obj:`int`):
            The chunk size of a chunked tensor: :obj:`num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (:obj:`int`):
            The dimension over which the :obj:`input_tensors` should be chunked.
        input_tensors (:obj:`Tuple[torch.Tensor]`):
            The input tensors of ``forward_fn`` which will be chunked

    Returns:
        :obj:`torch.Tensor`: A tensor with the same shape as the :obj:`forward_fn` would have given if applied`.


    Examples::

        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states

        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    """

    # add parameters --------------
    s, t, x_list, h_list = None, None, None, None
    if 't' in kwargs: t = kwargs['t']
    if 's' in kwargs: s = kwargs['s']
    if 'x_list' in kwargs: x_list = kwargs['x_list']
    if 'h_list' in kwargs: h_list = kwargs['h_list']
    # other parameters --------------


    assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)
    tensor_shape = input_tensors[0].shape[chunk_dim]
    assert all(
        input_tensor.shape[chunk_dim] == tensor_shape for input_tensor in input_tensors
    ), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    # num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    # assert num_args_in_forward_chunk_fn == len(
    #     input_tensors
    # ), "forward_chunk_fn expects {} arguments, but only {} input tensors are given".format(
    #     num_args_in_forward_chunk_fn, len(input_tensors)
    # )

    if chunk_size > 0:
        assert (
            input_tensors[0].shape[chunk_dim] % chunk_size == 0
        ), "The dimension to be chunked {} has to be a multiple of the chunk size {}".format(
            input_tensors[0].shape[chunk_dim], chunk_size
        )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk,t=t,s=s) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors,
                      t=t,s=s)
