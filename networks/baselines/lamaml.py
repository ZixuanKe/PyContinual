from torch.nn.functional import gelu, elu
import torch.nn.functional as F
import torch.nn as nn
import math
import torch
import random
from typing import List, Optional, Tuple, Union
from collections import OrderedDict
from transformers import RobertaForSequenceClassification, BartForConditionalGeneration, BartModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    ModelOutput
)


class MyRobertaOutput(ModelOutput):
    loss: torch.FloatTensor = None
    contrast_loss: torch.FloatTensor = None
    sum_loss: torch.FloatTensor = None
    logits = None
    past_key_values = None
    hidden_states = None
    decoder_hidden_states = None
    decoder_attentions = None
    cross_attentions = None
    encoder_last_hidden_state = None
    encoder_hidden_states = None
    encoder_attentions = None
    ppl = None


class Learner:

    def __init__(self, config, taskcla, args, model):
        self.args = args
        self.taskcla = taskcla
        self.config = config
        self.model = model

    def step_and_zero_grad(self):
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.args.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(
            self.alpha_lr.parameters(), self.args.grad_clip_norm)
        self.opt_lr.step()
        for i, p in enumerate(self.model.parameters()):
            if p.grad is not None:
                p.data = p.data - p.grad * nn.functional.relu(self.alpha_lr[i])
        self.opt_lr.zero_grad()
        self.alpha_lr.zero_grad()
        self.model.zero_grad()

    def define_task_lr_params(self, alpha_init=1e-3, saved_alpha_lr=None):
        # Setup learning parameters
        if saved_alpha_lr is None:
            self.alpha_lr = nn.ParameterList([])

            for p in self.model.parameters():
                self.alpha_lr.append(nn.Parameter(
                    alpha_init * torch.ones(p.shape, requires_grad=True)))
        else:
            self.alpha_lr = saved_alpha_lr

        self.opt_lr = torch.optim.SGD(
            list(self.alpha_lr.parameters()), lr=1e-1)

    def prepare(self, accelerator):
        self.alpha_lr, self.opt_lr = accelerator.prepare(
            self.alpha_lr, self.opt_lr)

    def meta_loss(self, fast_weights, batch, is_train=True):

        if fast_weights is None:
            fast_weights = OrderedDict(self.model.named_parameters())

        if self.args.task_name in self.args.ner_datasets:
            raise NotImplementedError

        elif self.args.task_name in self.args.classification:
            loss, logits = self.functional_robertaforsequenceclassification(
                fast_weights, self.config,
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                is_train=is_train,
                labels=batch['cls_labels'],
                task=batch['task']
            )

        elif self.args.task_name in self.args.generation:
            loss, logits = self.functional_bartforconditionalgeneration(
                fast_weights, self.config,
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                is_train=is_train,
                labels=batch['labels']
            )
        else:
            raise NotImplementedError

        return MyRobertaOutput(
            loss=loss,
            contrast_loss=None,
            sum_loss=None,
            logits=logits,
            ppl=None,
            hidden_states=None
        )

    def inner_update(self, fast_weights, batch, is_train=True):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        outputs = self.meta_loss(fast_weights, batch, is_train=is_train)
        loss = outputs.loss

        if fast_weights is None:
            fast_weights = OrderedDict(self.model.named_parameters())

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        grads = torch.autograd.grad(loss, list(fast_weights.values(
        )), create_graph=True, retain_graph=True, allow_unused=True)

        for i in range(len(grads)):
            if grads[i] is not None:
                torch.clamp(grads[i], min=-self.args.grad_clip_norm,
                            max=self.args.grad_clip_norm)

        fast_weights_var = list(
            map(lambda p: p[1][0] - p[0] * nn.functional.relu(p[1][1]) if p[0] is not None else p[1][0], zip(grads, zip(list(fast_weights.values()), self.alpha_lr))))

        fast_weights = OrderedDict({k: v for (k, v) in zip(
            list(fast_weights.keys()), fast_weights_var)})

        return fast_weights

    def functional_bartforconditionalgeneration(
        self, fast_weights, config,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        is_train=True,
    ):
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, config.pad_token_id, config.decoder_start_token_id
                )
        outputs = functional_bart(fast_weights, config, input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                  decoder_attention_mask=decoder_attention_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs,
                                  past_key_values=past_key_values, decoder_inputs_embeds=decoder_inputs_embeds, is_train=is_train)

        lm_logits = F.linear(
            outputs[0], weight=fast_weights['model.shared.weight'])
        masked_lm_loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return masked_lm_loss, lm_logits

    def functional_robertaforsequenceclassification(
        self, fast_weights, config, input_ids=None, attention_mask=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
        encoder_attention_mask=None, is_train=True, labels=None, task=None
    ):
        outputs = functional_roberta(fast_weights, config, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                     position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                                     encoder_attention_mask=encoder_attention_mask, is_train=is_train)
        sequence_output = outputs[0]
        loss = 0
        logits = None
        if labels is not None and task is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = []
            for t_id, t in enumerate(task):
                x = sequence_output[t_id].unsqueeze(0)
                x = F.dropout(
                    x[:, 0, :], p=config.hidden_dropout_prob, training=is_train)
                x = F.linear(
                    x, fast_weights[f'classifiers.{t}.dense.weight'], fast_weights[f'classifiers.{t}.dense.bias'])
                x = torch.tanh(x)
                x = F.dropout(x, p=config.hidden_dropout_prob,
                              training=is_train)
                logit = F.linear(
                    x, fast_weights[f'classifiers.{t}.out_proj.weight'], fast_weights[f'classifiers.{t}.out_proj.bias'])
                num_labels = self.taskcla[t][1]
                cur_loss = loss_fct(
                    logit.view(-1, num_labels), labels[t_id].view(-1))
                loss += cur_loss
                logits.append(logit)

            loss = loss / len(task)
        else:
            logits = []
            for t in range(self.args.ntasks):
                x = F.linear(
                    sequence_output, fast_weights[f'classifiers.{t}.dense.weight'], fast_weights[f'classifiers.{t}.dense.bias'])
                x = F.dropout(x, p=config.hidden_dropout_prob,
                              training=is_train)
                logit = F.linear(
                    x, fast_weights[f'classifiers.{t}.out_proj.weight'], fast_weights[f'classifiers.{t}.out_proj.bias'])
                logits.append(logit)
            logits = torch.cat(logits, dim=-1)

        return loss, logits


def functional_roberta(fast_weights, config, input_ids=None, attention_mask=None, token_type_ids=None,
                       position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                       encoder_attention_mask=None, is_train=True):

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError(
            "You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)
    if token_type_ids is None:
        token_type_ids = torch.zeros(
            input_shape, dtype=torch.long, device=device)

    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        if config.is_decoder:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, None, :].repeat(
                batch_size, seq_length, 1) <= seq_ids[None, :, None]
            causal_mask = causal_mask.to(torch.long)
            extended_attention_mask = causal_mask[:, None,
                                                  :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
            input_shape, attention_mask.shape))

    extended_attention_mask = extended_attention_mask.to(dtype=next(
        (p for p in fast_weights.values())).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_shape, device=device)

        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(encoder_hidden_shape,
                                                                                                                           encoder_attention_mask.shape))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=next((p for p in fast_weights.values())).dtype)
        encoder_extended_attention_mask = (
            1.0 - encoder_extended_attention_mask) * -10000.0
    else:
        encoder_extended_attention_mask = None

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(
                0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(
                config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=next(
            (p for p in fast_weights.values())).dtype)
    else:
        head_mask = [None] * config.num_hidden_layers

    embedding_output = functional_embedding(fast_weights, config, input_ids, position_ids,
                                            token_type_ids, inputs_embeds, is_train=is_train)

    encoder_outputs = functional_encoder(fast_weights, config, embedding_output,
                                         attention_mask=extended_attention_mask,
                                         head_mask=head_mask, encoder_hidden_states=encoder_hidden_states,
                                         encoder_attention_mask=encoder_extended_attention_mask, is_train=is_train)

    sequence_output = encoder_outputs
    outputs = (sequence_output,)
    return outputs


def functional_embedding(fast_weights, config, input_ids, position_ids,
                         token_type_ids, inputs_embeds=None, is_train=True):

    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]

    seq_length = input_shape[1]
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if position_ids is None:
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
    if token_type_ids is None:
        token_type_ids = torch.zeros(
            input_shape, dtype=torch.long, device=device)

    if inputs_embeds is None:
        inputs_embeds = F.embedding(
            input_ids, fast_weights['roberta.embeddings.word_embeddings.weight'], padding_idx=0)

    position_embeddings = F.embedding(
        position_ids, fast_weights['roberta.embeddings.position_embeddings.weight'])

    token_type_embeddings = F.embedding(
        token_type_ids, fast_weights['roberta.embeddings.token_type_embeddings.weight'])

    embeddings = inputs_embeds + position_embeddings + token_type_embeddings

    embeddings = F.layer_norm(embeddings, [config.hidden_size],
                              weight=fast_weights['roberta.embeddings.LayerNorm.weight'],
                              bias=fast_weights['roberta.embeddings.LayerNorm.bias'],
                              eps=config.layer_norm_eps)

    embeddings = F.dropout(
        embeddings, p=config.hidden_dropout_prob, training=is_train)

    return embeddings


def transpose_for_scores(config, x):
    new_x_shape = x.size()[:-1] + (config.num_attention_heads,
                                   int(config.hidden_size / config.num_attention_heads))
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)


def functional_self_attention(fast_weights, config, layer_idx,
                              hidden_states, attention_mask, head_mask,
                              encoder_hidden_states, encoder_attention_mask,
                              is_train=True):

    attention_head_size = int(config.hidden_size / config.num_attention_heads)
    all_head_size = config.num_attention_heads * attention_head_size

    mixed_query_layer = F.linear(hidden_states,
                                 fast_weights['roberta.encoder.layer.' +
                                              layer_idx+'.attention.self.query.weight'],
                                 fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.query.bias'])

    if encoder_hidden_states is not None:
        mixed_key_layer = F.linear(encoder_hidden_states,
                                   fast_weights['roberta.encoder.layer.' +
                                                layer_idx+'.attention.self.key.weight'],
                                   fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.key.bias'])
        mixed_value_layer = F.linear(encoder_hidden_states,
                                     fast_weights['roberta.encoder.layer.' +
                                                  layer_idx+'.attention.self.value.weight'],
                                     fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.value.bias'])
        attention_mask = encoder_attention_mask
    else:
        mixed_key_layer = F.linear(hidden_states,
                                   fast_weights['roberta.encoder.layer.' +
                                                layer_idx+'.attention.self.key.weight'],
                                   fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.key.bias'])
        mixed_value_layer = F.linear(hidden_states,
                                     fast_weights['roberta.encoder.layer.' +
                                                  layer_idx+'.attention.self.value.weight'],
                                     fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.value.bias'])

    query_layer = transpose_for_scores(config, mixed_query_layer)
    key_layer = transpose_for_scores(config, mixed_key_layer)
    value_layer = transpose_for_scores(config, mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

    if is_train:
        attention_probs = F.dropout(
            attention_probs, p=config.attention_probs_dropout_prob)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = context_layer
    return outputs


def functional_out_attention(fast_weights, config, layer_idx,
                             hidden_states, input_tensor,
                             is_train=True):

    hidden_states = F.linear(hidden_states,
                             fast_weights['roberta.encoder.layer.' +
                                          layer_idx+'.attention.output.dense.weight'],
                             fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.output.dense.bias'])

    hidden_states = F.dropout(
        hidden_states, p=config.hidden_dropout_prob, training=is_train)
    hidden_states = F.layer_norm(hidden_states + input_tensor, [config.hidden_size],
                                 weight=fast_weights['roberta.encoder.layer.' +
                                                     layer_idx+'.attention.output.LayerNorm.weight'],
                                 bias=fast_weights['roberta.encoder.layer.' +
                                                   layer_idx+'.attention.output.LayerNorm.bias'],
                                 eps=config.layer_norm_eps)

    return hidden_states


def functional_attention(fast_weights, config, layer_idx,
                         hidden_states, attention_mask=None, head_mask=None,
                         encoder_hidden_states=None, encoder_attention_mask=None,
                         is_train=True):

    self_outputs = functional_self_attention(fast_weights, config, layer_idx,
                                             hidden_states, attention_mask, head_mask,
                                             encoder_hidden_states, encoder_attention_mask, is_train)

    attention_output = functional_out_attention(fast_weights, config, layer_idx,
                                                self_outputs, hidden_states, is_train)
    return attention_output


def functional_intermediate(fast_weights, config, layer_idx, hidden_states, is_train=True):
    weight_name = 'roberta.encoder.layer.' + \
        layer_idx + '.intermediate.dense.weight'
    bias_name = 'roberta.encoder.layer.' + layer_idx + '.intermediate.dense.bias'
    hidden_states = F.linear(
        hidden_states, fast_weights[weight_name], fast_weights[bias_name])
    hidden_states = gelu(hidden_states)

    return hidden_states


def functional_output(fast_weights, config, layer_idx, hidden_states, input_tensor, is_train=True):

    hidden_states = F.linear(hidden_states,
                             fast_weights['roberta.encoder.layer.' +
                                          layer_idx+'.output.dense.weight'],
                             fast_weights['roberta.encoder.layer.'+layer_idx+'.output.dense.bias'])

    hidden_states = F.dropout(
        hidden_states, p=config.hidden_dropout_prob, training=is_train)
    hidden_states = F.layer_norm(hidden_states + input_tensor, [config.hidden_size],
                                 weight=fast_weights['roberta.encoder.layer.' +
                                                     layer_idx+'.output.LayerNorm.weight'],
                                 bias=fast_weights['roberta.encoder.layer.' +
                                                   layer_idx+'.output.LayerNorm.bias'],
                                 eps=config.layer_norm_eps)
    return hidden_states


def functional_layer(fast_weights, config, layer_idx, hidden_states, attention_mask,
                     head_mask, encoder_hidden_states, encoder_attention_mask, is_train=True):

    self_attention_outputs = functional_attention(fast_weights, config, layer_idx,
                                                  hidden_states, attention_mask, head_mask,
                                                  encoder_hidden_states, encoder_attention_mask, is_train)

    attention_output = self_attention_outputs
    intermediate_output = functional_intermediate(
        fast_weights, config, layer_idx, attention_output, is_train)
    layer_output = functional_output(fast_weights, config, layer_idx,
                                     intermediate_output, attention_output, is_train)

    return layer_output


def functional_encoder(fast_weights, config, hidden_states, attention_mask,
                       head_mask, encoder_hidden_states, encoder_attention_mask, is_train=True):

    for i in range(0, config.num_hidden_layers):
        layer_outputs = functional_layer(fast_weights, config, str(i),
                                         hidden_states, attention_mask, head_mask[i],
                                         encoder_hidden_states, encoder_attention_mask, is_train)
        hidden_states = layer_outputs

    outputs = hidden_states
    return outputs


def functional_bart_attention(fast_weights, config, hidden_states, layer_idx, attention_type, key_value_states=None, past_key_value=None, attention_mask=None, layer_head_mask=None, is_train=True):

    if attention_type == 'decoder' or attention_type == 'cross':
        num_heads = config.decoder_attention_heads
    else:
        num_heads = config.encoder_attention_heads
    head_dim = config.d_model // num_heads
    scaling = head_dim ** (-0.5)

    def _shape(tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()
    if attention_type == 'encoder':
        query_states = F.linear(hidden_states,
                                fast_weights['model.encoder.layers.' +
                                             str(layer_idx)+'.self_attn.q_proj.weight'],
                                fast_weights['model.encoder.layers.' +
                                             str(layer_idx)+'.self_attn.q_proj.bias']
                                ) * scaling
    elif attention_type == 'decoder':
        query_states = F.linear(hidden_states,
                                fast_weights['model.decoder.layers.' +
                                             str(layer_idx)+'.self_attn.q_proj.weight'],
                                fast_weights['model.decoder.layers.' +
                                             str(layer_idx)+'.self_attn.q_proj.bias']
                                ) * scaling
    elif attention_type == 'cross':
        query_states = F.linear(hidden_states,
                                fast_weights['model.decoder.layers.' +
                                             str(layer_idx)+'.encoder_attn.q_proj.weight'],
                                fast_weights['model.decoder.layers.' +
                                             str(layer_idx)+'.encoder_attn.q_proj.bias']
                                ) * scaling
    else:
        raise AttributeError
    if (is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]):
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        key_states = _shape(F.linear(key_value_states,
                                     fast_weights['model.decoder.layers.' +
                                                  str(layer_idx)+'.encoder_attn.k_proj.weight'],
                                     fast_weights['model.decoder.layers.'+str(layer_idx)+'.encoder_attn.k_proj.bias']), -1, bsz)
        value_states = _shape(F.linear(key_value_states,
                              fast_weights['model.decoder.layers.' +
                                           str(layer_idx)+'.encoder_attn.v_proj.weight'],
                              fast_weights['model.decoder.layers.'+str(layer_idx)+'.encoder_attn.v_proj.bias']), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        if attention_type == 'encoder':
            key_states = F.linear(hidden_states,
                                  fast_weights['model.encoder.layers.' +
                                               str(layer_idx)+'self_attn.k_proj.weight'],
                                  fast_weights['model.encoder.layers.' +
                                               str(layer_idx)+'self_attn.k_proj.bias']
                                  )
            value_states = F.linear(hidden_states,
                                    fast_weights['model.encoder.layers.' +
                                                 str(layer_idx)+'self_attn.v_proj.weight'],
                                    fast_weights['model.encoder.layers.' +
                                                 str(layer_idx)+'self_attn.v_proj.bias']
                                    )
        elif attention_type == 'decoder':
            key_states = F.linear(hidden_states,
                                  fast_weights['model.decoder.layers.' +
                                               str(layer_idx)+'.self_attn.k_proj.weight'],
                                  fast_weights['model.decoder.layers.' +
                                               str(layer_idx)+'.self_attn.k_proj.bias']
                                  )
            value_states = F.linear(hidden_states,
                                    fast_weights['model.decoder.layers.' +
                                                 str(layer_idx)+'.self_attn.v_proj.weight'],
                                    fast_weights['model.decoder.layers.' +
                                                 str(layer_idx)+'.self_attn.v_proj.bias']
                                    )
        elif attention_type == 'cross':
            key_states = F.linear(hidden_states,
                                  fast_weights['model.decoder.layers.' +
                                               str(layer_idx)+'.encoder_attn.k_proj.weight'],
                                  fast_weights['model.decoder.layers.' +
                                               str(layer_idx)+'.encoder_attn.k_proj.bias']
                                  )
            value_states = F.linear(hidden_states,
                                    fast_weights['model.decoder.layers.' +
                                                 str(layer_idx)+'.encoder_attn.v_proj.weight'],
                                    fast_weights['model.decoder.layers.' +
                                                 str(layer_idx)+'.encoder_attn.v_proj.bias']
                                    )
        key_states = _shape(key_states, -1, bsz)
        value_states = _shape(value_states, -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        if attention_type == 'encoder':
            key_states = F.linear(hidden_states,
                                  fast_weights['model.encoder.layers.' +
                                               str(layer_idx)+'.self_attn.k_proj.weight'],
                                  fast_weights['model.encoder.layers.' +
                                               str(layer_idx)+'.self_attn.k_proj.bias']
                                  )
            value_states = F.linear(hidden_states,
                                    fast_weights['model.encoder.layers.' +
                                                 str(layer_idx)+'.self_attn.v_proj.weight'],
                                    fast_weights['model.encoder.layers.' +
                                                 str(layer_idx)+'.self_attn.v_proj.bias']
                                    )
        elif attention_type == 'decoder':
            key_states = F.linear(hidden_states,
                                  fast_weights['model.decoder.layers.' +
                                               str(layer_idx)+'.self_attn.k_proj.weight'],
                                  fast_weights['model.decoder.layers.' +
                                               str(layer_idx)+'.self_attn.k_proj.bias']
                                  )
            value_states = F.linear(hidden_states,
                                    fast_weights['model.decoder.layers.' +
                                                 str(layer_idx)+'.self_attn.v_proj.weight'],
                                    fast_weights['model.decoder.layers.' +
                                                 str(layer_idx)+'.self_attn.v_proj.bias']
                                    )
        elif attention_type == 'cross':
            key_states = F.linear(hidden_states,
                                  fast_weights['model.decoder.layers.' +
                                               str(layer_idx)+'.encoder_attn.k_proj.weight'],
                                  fast_weights['model.decoder.layers.' +
                                               str(layer_idx)+'.encoder_attn.k_proj.bias']
                                  )
            value_states = F.linear(hidden_states,
                                    fast_weights['model.decoder.layers.' +
                                                 str(layer_idx)+'.encoder_attn.v_proj.weight'],
                                    fast_weights['model.decoder.layers.' +
                                                 str(layer_idx)+'.encoder_attn.v_proj.bias']
                                    )
        key_states = _shape(key_states, -1, bsz)
        value_states = _shape(value_states, -1, bsz)

    if attention_type == 'decoder' or attention_type == 'cross':
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * num_heads, -1, head_dim)

    query_states = _shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    if attn_weights.size() != (bsz * num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(
            bsz, num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(num_heads,)}, but is"
                f" {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(
            1, -1, 1, 1) * attn_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_probs = nn.functional.dropout(
        attn_weights, p=config.attention_dropout, training=is_train)
    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * num_heads, tgt_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, tgt_len, head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, num_heads, tgt_len, head_dim)
    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned aross GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, config.d_model)

    if attention_type == 'encoder':
        attn_output = F.linear(attn_output,
                               fast_weights['model.encoder.layers.' +
                                            str(layer_idx)+'.self_attn.out_proj.weight'],
                               fast_weights['model.encoder.layers.' +
                                            str(layer_idx)+'.self_attn.out_proj.bias']
                               )
    elif attention_type == 'decoder':
        attn_output = F.linear(attn_output,
                               fast_weights['model.decoder.layers.' +
                                            str(layer_idx)+'.self_attn.out_proj.weight'],
                               fast_weights['model.decoder.layers.' +
                                            str(layer_idx)+'.self_attn.out_proj.bias']
                               )
    elif attention_type == 'cross':
        attn_output = F.linear(attn_output,
                               fast_weights['model.decoder.layers.' +
                                            str(layer_idx)+'.encoder_attn.out_proj.weight'],
                               fast_weights['model.decoder.layers.' +
                                            str(layer_idx)+'.encoder_attn.out_proj.bias']
                               )
    attn_weights_reshaped = None

    return attn_output, attn_weights_reshaped, past_key_value


def functional_bart_encoder_layer(fast_weights, config, layer_idx, hidden_states, attention_mask, layer_head_mask, is_train=True):
    residual = hidden_states

    hidden_states, _, _ = functional_bart_attention(fast_weights, config, hidden_states=hidden_states, layer_idx=layer_idx,
                                                    attention_type='encoder', attention_mask=attention_mask, layer_head_mask=layer_head_mask, is_train=is_train)
    hidden_states = nn.functional.dropout(
        hidden_states, p=config.dropout, training=is_train)
    hidden_states = residual + hidden_states
    hidden_states = F.layer_norm(hidden_states, [config.hidden_size],
                                 weight=fast_weights['model.encoder.layers.' +
                                                     str(layer_idx)+'.self_attn_layer_norm.weight'],
                                 bias=fast_weights['model.encoder.layers.'+str(layer_idx)+'.self_attn_layer_norm.bias'])
    residual = hidden_states
    hidden_states = F.linear(hidden_states, fast_weights['model.encoder.layers.'+str(
        layer_idx)+'.fc1.weight'], fast_weights['model.encoder.layers.'+str(layer_idx)+'.fc1.bias'])
    hidden_states = F.gelu(hidden_states)
    hidden_states = nn.functional.dropout(
        hidden_states, p=config.activation_dropout, training=is_train)
    hidden_states = F.linear(hidden_states, fast_weights['model.encoder.layers.'+str(
        layer_idx)+'.fc2.weight'], fast_weights['model.encoder.layers.'+str(layer_idx)+'.fc2.bias'])
    hidden_states = nn.functional.dropout(
        hidden_states, p=config.dropout, training=is_train)
    hidden_states = residual + hidden_states
    hidden_states = F.layer_norm(hidden_states, [config.hidden_size],
                                 weight=fast_weights['model.encoder.layers.' +
                                                     str(layer_idx)+'.final_layer_norm.weight'],
                                 bias=fast_weights['model.encoder.layers.'+str(layer_idx)+'.final_layer_norm.bias'])

    if hidden_states.dtype == torch.float16 and (
        torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
    ):
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(
            hidden_states, min=-clamp_value, max=clamp_value)

    return hidden_states


def functional_bart_decoder_layer(fast_weights, config, layer_idx, hidden_states, attention_mask=None, encoder_hidden_states=None,
                                  encoder_attention_mask=None, layer_head_mask=None, cross_attn_layer_head_mask=None, past_key_value=None, is_train=True):
    residual = hidden_states
    # Self Attention
    # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    self_attn_past_key_value = past_key_value[:
                                              2] if past_key_value is not None else None
    # add present self-attn cache to positions 1,2 of present_key_value tuple
    hidden_states, self_attn_weights, present_key_value = functional_bart_attention(
        fast_weights, config,
        hidden_states=hidden_states,
        layer_idx=layer_idx,
        attention_type='decoder',
        past_key_value=self_attn_past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        is_train=is_train
    )
    hidden_states = nn.functional.dropout(
        hidden_states, p=config.dropout, training=is_train)
    hidden_states = residual + hidden_states
    hidden_states = F.layer_norm(hidden_states, [config.d_model],
                                 weight=fast_weights['model.decoder.layers.' +
                                                     str(layer_idx)+'.self_attn_layer_norm.weight'],
                                 bias=fast_weights['model.decoder.layers.'+str(layer_idx)+'.self_attn_layer_norm.bias'])
    # Cross-Attention Block
    cross_attn_present_key_value = None
    cross_attn_weights = None
    if encoder_hidden_states is not None:
        residual = hidden_states

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:
                                                   ] if past_key_value is not None else None
        hidden_states, cross_attn_weights, cross_attn_present_key_value = functional_bart_attention(
            fast_weights, config, layer_idx=layer_idx, attention_type='cross',
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            is_train=is_train
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=config.dropout, training=is_train)
        hidden_states = residual + hidden_states
        hidden_states = F.layer_norm(hidden_states, [config.d_model],
                                     weight=fast_weights['model.decoder.layers.'+str(
                                         layer_idx)+'.encoder_attn_layer_norm.weight'],
                                     bias=fast_weights['model.decoder.layers.' +
                                                       str(layer_idx)+'.encoder_attn_layer_norm.bias']
                                     )

        # add cross-attn to positions 3,4 of present_key_value tuple
        present_key_value = present_key_value + cross_attn_present_key_value

    # Fully Connected
    residual = hidden_states
    hidden_states = F.gelu(F.linear(hidden_states, fast_weights['model.decoder.layers.'+str(
        layer_idx)+'.fc1.weight'], fast_weights['model.decoder.layers.'+str(layer_idx)+'.fc1.bias']))
    hidden_states = nn.functional.dropout(
        hidden_states, p=config.activation_dropout, training=is_train)
    hidden_states = F.linear(hidden_states, fast_weights['model.decoder.layers.'+str(
        layer_idx)+'.fc2.weight'], fast_weights['model.decoder.layers.'+str(layer_idx)+'.fc2.bias'])
    hidden_states = nn.functional.dropout(
        hidden_states, p=config.dropout, training=is_train)
    hidden_states = residual + hidden_states
    hidden_states = F.layer_norm(hidden_states, [config.d_model],
                                 weight=fast_weights['model.decoder.layers.' +
                                                     str(layer_idx)+'.final_layer_norm.weight'],
                                 bias=fast_weights['model.decoder.layers.' +
                                                   str(layer_idx)+'.final_layer_norm.bias']
                                 )

    return hidden_states


def functional_bart_positional_embedding(fast_weights, config, input_ids, past_key_values_length=0, is_decoder=False):
    bsz, seq_len = input_ids.shape[:2]
    weight = fast_weights['model.decoder.embed_positions.weight'] if is_decoder else fast_weights['model.encoder.embed_positions.weight']
    positions = torch.arange(
        past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=weight.device
    ).expand(bsz, -1)
    return F.embedding(positions + 2, weight)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(
        bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def functional_bart_encoder(fast_weights, config, input_ids=None, attention_mask=None, is_train=True):

    embed_dim = config.d_model
    layerdrop = config.encoder_layerdrop
    dropout = config.dropout

    embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

    input = input_ids
    input_ids = input_ids.view(-1, input_ids.shape[-1])

    inputs_embeds = F.embedding(
        input_ids, weight=fast_weights['model.shared.weight']) * embed_scale

    embed_pos = functional_bart_positional_embedding(
        fast_weights, config, input, is_decoder=False)
    embed_pos = embed_pos.to(inputs_embeds.device)

    hidden_states = inputs_embeds + embed_pos

    hidden_states = F.layer_norm(hidden_states, [
                                 embed_dim], fast_weights['model.encoder.layernorm_embedding.weight'], fast_weights['model.encoder.layernorm_embedding.bias'])
    hidden_states = nn.functional.dropout(
        hidden_states, p=dropout, training=is_train)

    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

    for idx in range(config.encoder_layers):

        dropout_probability = random.uniform(0, 1)
        if is_train and (dropout_probability < layerdrop):  # skip the layer
            layer_outputs = (None, None)
        else:
            layer_outputs = functional_bart_encoder_layer(
                fast_weights, config, str(idx),
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=None,
                is_train=is_train
            )

            hidden_states = layer_outputs

    return hidden_states


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def functional_bart_decoder(fast_weights, config, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                            head_mask=None, cross_attn_head_mask=None, past_key_values=None, is_train=True):
    dropout = config.dropout
    layerdrop = config.decoder_layerdrop
    embed_dim = config.d_model
    embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

    def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                combined_attention_mask
            )

        return combined_attention_mask
    input = input_ids
    input_shape = input.shape
    input_ids = input_ids.view(-1, input_shape[-1])

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    inputs_embeds = F.embedding(
        input, weight=fast_weights['model.shared.weight']) * embed_scale

    attention_mask = _prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )

    # expand encoder attention mask
    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        encoder_attention_mask = _expand_mask(
            encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

    # embed positions
    positions = functional_bart_positional_embedding(
        fast_weights, config, input, past_key_values_length, is_decoder=True)
    positions = positions.to(inputs_embeds.device)

    hidden_states = inputs_embeds + positions
    hidden_states = F.layer_norm(hidden_states, [
                                 embed_dim], weight=fast_weights['model.decoder.layernorm_embedding.weight'], bias=fast_weights['model.decoder.layernorm_embedding.bias'])

    hidden_states = nn.functional.dropout(
        hidden_states, p=dropout, training=is_train)

    for idx in range(config.decoder_layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        dropout_probability = random.uniform(0, 1)
        if is_train and (dropout_probability < layerdrop):
            continue

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_outputs = functional_bart_decoder_layer(
            fast_weights, config, str(idx),
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            layer_head_mask=(
                head_mask[idx] if head_mask is not None else None),
            cross_attn_layer_head_mask=(
                cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
            ),
            past_key_value=past_key_value,
            is_train=is_train
        )
        hidden_states = layer_outputs

    return hidden_states


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def functional_bart(fast_weights, config, input_ids, attention_mask, decoder_input_ids=None, decoder_attention_mask=None, decoder_head_mask=None,
                    cross_attn_head_mask=None, encoder_outputs=None, past_key_values=None, decoder_inputs_embeds=None, is_train=True):

    if decoder_input_ids is None and decoder_inputs_embeds is None:
        if input_ids is None:
            raise ValueError(
                "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                "passed, `input_ids` cannot be `None`. Please pass either "
                "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
            )
        decoder_input_ids = shift_tokens_right(
            input_ids, config.pad_token_id, config.decoder_start_token_id
        )

    encoder_outputs = functional_bart_encoder(
        fast_weights, config,
        input_ids=input_ids,
        attention_mask=attention_mask,
        is_train=is_train
    )

    # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
    decoder_outputs = functional_bart_decoder(
        fast_weights, config, is_train=is_train,
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_outputs,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values,
    )

    return (decoder_outputs, encoder_outputs)


if __name__ == '__main__':

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    fast_weights = OrderedDict(model.named_parameters())
    input_ids = torch.Tensor([[101,  1303,  1110,  1199,  3087,  1106,  4035, 13775,   102],
                              [101,   178,  1274,  1204,  1176,  1115,  4170,   182,   102]]).to(torch.long)
    attention_mask = torch.Tensor([[1,  1,  1,  1,  1,  1,  1, 1, 1],
                                   [1,  1,  1,  1,  1,  1,  1, 1, 1]]).to(torch.long)
    model.eval()
    print(model.model(input_ids, attention_mask)[0])
    print(functional_bart(fast_weights, model.config, input_ids=input_ids,
          attention_mask=attention_mask, is_train=False)[0])
