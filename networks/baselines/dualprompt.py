from base64 import encode
from turtle import forward
import random
from networks.transformers.bart import MyBartEncoder, MyBartEncoderLayer, _expand_mask, MyBartDecoder, MyBartModel, shift_tokens_right, MyOutput, MyBartForConditionalGeneration
from networks.transformers.roberta import MyRobertaLayer, RobertaPreTrainedModel, BertModelAdaptersMixin, RobertaEmbeddings, RobertaPooler, ModelWithHeadsAdaptersMixin, RobertaClassificationHead
import torch
import torch.nn as nn
from utils import utils
import os
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.adapters.composition import adjust_tensors_for_parallel
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput, TokenClassifierOutput, BaseModelOutput

class DualPromptRobertaLayer(MyRobertaLayer):
    def __init__(self, config, args, add_g_prompt=False, add_e_prompt=False):
        super().__init__(config, args)
        self.add_g_prompt = add_g_prompt
        self.add_e_prompt = add_e_prompt
    
    def _cat_prompt_to_input(self, hidden_states, prompt_embeds):
        """
        Concatenates prompt embeddings to inputs (hidden states of the previous layer).
        """
        hidden_states = torch.cat([prompt_embeds, hidden_states], dim=1)

        return hidden_states

    def _extend_attention_mask(self, attention_mask, prompt_embeds):
        """
        Extends attention_mask to match the input_ids's shape.
        """

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        n_tokens = prompt_embeds.shape[1]
        return torch.cat(
            [
                torch.full(
                    (n_batches, 1, 1, n_tokens), 1).to(
                    attention_mask.device),
                attention_mask
            ],
            dim=3,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        g_prompt=None,
        e_prompt=None,
        **kwargs
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        if self.add_g_prompt and g_prompt is not None:
            hidden_states = self._cat_prompt_to_input(hidden_states, g_prompt)
            attention_mask = self._extend_attention_mask(attention_mask, g_prompt)
        elif self.add_e_prompt and e_prompt is not None:
            hidden_states = self._cat_prompt_to_input(hidden_states, e_prompt)
            attention_mask = self._extend_attention_mask(attention_mask, e_prompt)
        
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs, attention_mask

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class DualPromptRobertaEncoder(nn.Module):
    def __init__(self, config, args, start_g, end_g, start_e, end_e):
        """
        We add G-Prompt to [start_g, end_g]th layers and add E-Prompt ot [start_e, end_t]th layers.
        """
        super().__init__()
        self.config = config
        self.args = args
        self.layer = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            add_g_prompt = True if start_g <= i <= end_g else False
            add_e_prompt = True if start_e <= i <= end_e else False
            self.layer.append(DualPromptRobertaLayer(config, args, add_g_prompt=add_g_prompt, add_e_prompt=add_e_prompt))

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        g_prompt=None,
        e_prompt=None,
        **kwargs
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

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
                layer_outputs, attention_mask = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    g_prompt=g_prompt,
                    e_prompt=e_prompt,
                )

            hidden_states = layer_outputs[0]
            (attention_mask,) = adjust_tensors_for_parallel(hidden_states, attention_mask)

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class DualPromptRobertaModel(BertModelAdaptersMixin, RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, args, add_pooling_layer=True, start_g=1, end_g=2, start_e=3, end_e=5):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = DualPromptRobertaEncoder(config, args, start_g, end_g, start_e, end_e)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        self._init_adapter_modules()

         # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
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
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            g_prompt=None,
            e_prompt=None,
            **kwargs
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
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
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        embedding_output = self.invertible_adapters_forward(embedding_output)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            g_prompt=g_prompt,
            e_prompt=e_prompt,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class DualPromptRobertaForSequenceClassification(ModelWithHeadsAdaptersMixin, RobertaPreTrainedModel):
    def __init__(self, config, taskcla, args, tokenizer=None, Lg=5, Le=20, start_g=1, end_g=2, start_e=3, end_e=5):
        """
        G-Prompt: [Lg, embed_dim]
        E-Prompt: {e_1, ..., e_T}, e_i [Le, embed_dim]
        Learnable key (for task agnostic setting): {(k_1, e_1), ..., (k_T, e_T)}, k_i [last_hidden_dim]
        """
        super().__init__(config)
        self.taskcla = taskcla
        self.config = config
        self.args = args
        self.tokenizer = None
        self.Lg = Lg
        self.Le = Le
        self.start_g = start_g
        self.end_g = end_g
        self.start_e = start_e
        self.end_e = end_e
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.roberta = DualPromptRobertaModel(config, args, add_pooling_layer=False,
                                              start_g=start_g, end_g=end_g, start_e=start_e, end_e=end_e)

        self.classifiers = nn.ModuleList()
        for task, n in taskcla:
            config.num_labels = n
            self.classifiers.append(RobertaClassificationHead(config))

        # G-Prompt, E-Prompt and learnable keys.
        # We maintain {e_1, ..., e_T} as a large matrix of shape (T, Le * hidden_dim) for efficient look-up in task
        # agnostic setting.
        init_g_prompt_value = torch.FloatTensor(self.Lg, self.config.hidden_size).uniform_(-0.5, 0.5)
        self.g_prompt = nn.Embedding(self.Lg, self.config.hidden_size)
        self.g_prompt.weight = nn.parameter.Parameter(init_g_prompt_value)
        init_e_prompts_value = torch.FloatTensor(self.args.ntasks, self.Le * self.config.hidden_size).uniform_(-0.5, 0.5)
        self.e_prompts = nn.Embedding(self.args.ntasks, self.Le * self.config.hidden_size)
        self.e_prompts.weight = nn.parameter.Parameter(init_e_prompts_value)
        self.keys = nn.Embedding(self.args.ntasks, self.config.hidden_size)

        # Hyperparameter for the loss function.
        self.lam = 1.0  # Follow the original paper.

    def _prepare_prompts(self, input_ids, task_id=None):
        """
        Prepare G-Prompt and E-Prompt according to task_id or matching function in task agnostic setting.
        
        If task_id = 0, we use cosine similarity for E-prompt selections; otherwise, we directly select E-Prompt with
        task_id.
        """
        inputs_embeds = self.roberta.embeddings(input_ids)

        # Use the frozen pre-trained model to get the query features: q(x) = f(x)[0,:]
        q = self.roberta(inputs_embeds=inputs_embeds)[0][:, 0, :]
        if task_id is not None:
            sim = utils.sim_matrix(q, self.keys.weight[task_id].reshape(1, -1))
            e_prompt = self.e_prompts.weight[task_id].repeat(inputs_embeds.size(0), 1).reshape(
                inputs_embeds.size(0), -1, self.config.hidden_size
            )
            matching_loss = sim.mean()
        else:
            sim = utils.sim_matrix(q, self.keys.weight)
            selection = torch.topk(sim, 1, dim=1)
            e_prompt = self.e_prompts.weight[selection.indices].reshape(
                inputs_embeds.size(0), -1, self.config.hidden_size
            )
            matching_loss = selection.values.mean()

        g_prompt = self.g_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        return inputs_embeds, g_prompt, e_prompt, matching_loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
        my_loss=None,
        nsp_labels=None,
        **kwargs
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # for evaluation and training, it's assumed to use the same prompt within the batch.
        inputs_embeds, g_prompt, e_prompt, matching_loss = self._prepare_prompts(input_ids)

        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            g_prompt=g_prompt,
            e_prompt=e_prompt,
        )
        sequence_output = outputs[0]
        # logits = self.classifiers[self.args.task](sequence_output)
        # self.num_labels = self.taskcla[self.args.task][1]

        loss = 0
        logits = None
        if labels is not None and task is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = []
            for t_id, t in enumerate(task):  # different task in the same batch
                if self.args.is_transfer or self.args.is_reference:
                    logit = self.readouts[self.args.eval_t][t](sequence_output[t_id].unsqueeze(0)) #no stack can be performed
                else:
                    logit = self.classifiers[t](sequence_output[t_id].unsqueeze(0)) #no stack can be performed
                num_labels = self.taskcla[t][1]
                cur_loss = loss_fct(logit.view(-1, num_labels), labels[t_id].view(-1))
                loss += cur_loss
                logits.append(logit)

            if 'mtl' not in self.args.baseline and 'comb' not in self.args.baseline and 'agem' not in self.args.baseline and 'derpp' not in self.args.baseline:
                # TODO: all replay methods should not calculate this.
                logits = torch.cat(logits)

            loss = loss / len(task)

        else:
            logits = []
            for t in range(self.args.ntasks):
                # [B, hidden] -> [B, labels]
                logit = self.classifiers[t](sequence_output)
                logits.append(logit)
            logits = torch.cat(logits,dim=-1)

        # Add matching loss.
        loss += self.lam * matching_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class DualPromptRobertaForTokenClassification(DualPromptRobertaForSequenceClassification):

    def __init__(self, config, taskcla, args, tokenizer=None, Lg=5, Le=20, start_g=1, end_g=2, start_e=3, end_e=5):
        super().__init__(config, taskcla, args, tokenizer, Lg, Le, start_g, end_g, start_e, end_e)
        self.classifiers = nn.ModuleList()
        for task, n in taskcla:
            config.num_labels = n
            classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.classifiers.append(classifier)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
        my_loss=None,
        nsp_labels=None,
        **kwargs
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # no replay, no multitask ! (we have to prepare different prompts for different task)
        inputs_embeds, g_prompt, e_prompt, matching_loss = self._prepare_prompts(input_ids)

        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            g_prompt=g_prompt,
            e_prompt=e_prompt,
        )
        # adapt prompt to token classification !!
        sequence_output = outputs[0][...,-self.args.max_length:,:]

        sequence_output = self.dropout(sequence_output)
        # logits = self.classifiers[self.args.task](sequence_output)
        # self.num_labels = self.taskcla[self.args.task][1]

        loss = 0
        logits = None
        if labels is not None and task is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = []
            for t_id, t in enumerate(task):  # different task in the same batch
                if self.args.is_transfer or self.args.is_reference:
                    logit = self.readouts[self.args.eval_t][t](sequence_output[t_id].unsqueeze(0)) #no stack can be performed
                else:
                    logit = self.classifiers[t](sequence_output[t_id].unsqueeze(0)) #no stack can be performed
                num_labels = self.taskcla[t][1]
                cur_loss = loss_fct(logit.view(-1, num_labels), labels[t_id].view(-1))
                loss += cur_loss
                logits.append(logit)

            if 'mtl' not in self.args.baseline and 'comb' not in self.args.baseline and 'agem' not in self.args.baseline and 'derpp' not in self.args.baseline:
                # TODO: all replay methods should not calculate this.
                logits = torch.cat(logits)

            loss = loss / len(task)

        else:
            logits = []
            for t in range(self.args.ntasks):
                # [B, hidden] -> [B, labels]
                logit = self.classifiers[t](sequence_output)
                logits.append(logit)
            logits = torch.cat(logits,dim=-1)

        # Add matching loss.
        loss += self.lam * matching_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class DualPromptBartEncoderLayer(MyBartEncoderLayer):
    def __init__(self, config, args, add_g_prompt=False, add_e_prompt=False):
        super().__init__(config, args)
        self.add_g_prompt = add_g_prompt
        self.add_e_prompt = add_e_prompt
    
    def _cat_prompt_to_input(self, hidden_states, prompt_embeds):
        
        hidden_states = torch.cat([prompt_embeds, hidden_states], dim=1)
        return hidden_states
    
    def _extend_attention_mask(self, attention_mask, prompt_embeds):
        
        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)
        n_batches = attention_mask.shape[0]
        n_tokens = prompt_embeds.shape[1]

        tmp = torch.cat(
            [
                torch.full(
                    (n_batches, 1, attention_mask.shape[-2], n_tokens), 1
                ).to(attention_mask.device).long(),
                attention_mask
            ],
            dim=-1
        )

        return torch.cat(
            [
                torch.full(
                    (n_batches, 1, n_tokens, tmp.shape[-1]), 1).to(
                    attention_mask.device).long(),
                tmp
            ],
            dim=-2,
        )
    
    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        output_attentions=False,
        teacher_encoder_hidden_state=None,
        g_prompt=None,
        e_prompt=None,
        **kwargs
    ):
        if self.add_g_prompt and g_prompt is not None:
            hidden_states = self._cat_prompt_to_input(hidden_states, g_prompt)
            attention_mask = self._extend_attention_mask(attention_mask, g_prompt)
        elif self.add_e_prompt and e_prompt is not None:
            hidden_states = self._cat_prompt_to_input(hidden_states, e_prompt)
            attention_mask = self._extend_attention_mask(attention_mask, e_prompt)

        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.attention_adapters(hidden_states, residual, 'encoder', self.self_attn_layer_norm) # TODO: seems set to ignore by adapter_transformer by default
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.output_adapters(hidden_states, residual, 'encoder' , self.final_layer_norm)


        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs, attention_mask

class DualPromptBartEncoder(MyBartEncoder):
    def __init__(self, config, start_g, end_g, start_e, end_e, embed_tokens=None, args=None):
        super().__init__(config, embed_tokens, args)
        self.layers = nn.ModuleList()
        for i in range(config.encoder_layers):
            add_g_prompt = True if start_g <= i <= end_g else False
            add_e_prompt = True if start_e <= i <= end_e else False
            self.layers.append(DualPromptBartEncoderLayer(config, args, add_g_prompt=add_g_prompt, add_e_prompt=add_e_prompt))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        teacher_encoder_hidden_states=None,
        g_prompt=None,
        e_prompt=None,
        **kwargs
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = self.invertible_adapters_forward(hidden_states)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:

                    if self.args.teacher_encoder_hidden_states is not None:
                        layer_outputs = encoder_layer(
                            hidden_states,
                            attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                            teacher_encoder_hidden_state = self.args.teacher_encoder_hidden_states[idx],
                            **kwargs
                        )

                    else:
                        layer_outputs, attention_mask = encoder_layer(
                            hidden_states,
                            attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                            g_prompt=g_prompt,
                            e_prompt=e_prompt,
                            **kwargs
                        )

                hidden_states = layer_outputs[0]
                (attention_mask,) = adjust_tensors_for_parallel(hidden_states, attention_mask)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions, attention_mask] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=attention_mask
        )

class DualPromptBartModel(MyBartModel):

    def __init__(self, config, args, start_g=1, end_g=2, start_e=3, end_e=5):
        super().__init__(config, args)
        self.encoder = DualPromptBartEncoder(config, start_g, end_g, start_e, end_e, self.shared, args)
        self.decoder = MyBartDecoder(config, self.shared, args)

    def forward(
        self,
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
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        g_prompt=None,
        e_prompt=None,
        **kwargs
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                g_prompt=g_prompt,
                e_prompt=e_prompt,
                **kwargs
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        torch.full(
                            (attention_mask.shape[0], encoder_outputs[-1].shape[-1]-attention_mask.shape[-1]), 1).to(
                            attention_mask.device).long(),
                        attention_mask
                    ],
                    dim=1,
                )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # inflate all decoder inputs according to encoder output
        decoder_input_ids, decoder_attention_mask, attention_mask = adjust_tensors_for_parallel(
            encoder_outputs[0], decoder_input_ids, decoder_attention_mask, attention_mask
        )
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class DualPromptBartForConditionalGeneration(MyBartForConditionalGeneration):

    def __init__(self, config, taskcla, args, Lg=5, Le=20, start_g=1, end_g=2, start_e=3, end_e=5):
        super().__init__(config, taskcla, args)
        self.Lg = Lg
        self.Le = Le
        self.start_g = start_g
        self.end_g = end_g
        self.start_e = start_e
        self.end_e = end_e
        self.model = DualPromptBartModel(config, args, start_g, end_g, start_e, end_e)

        init_g_prompt_value = torch.FloatTensor(self.Lg, self.config.d_model).uniform_(-0.5, 0.5)
        self.g_prompt = nn.Embedding(self.Lg, self.config.d_model)
        self.g_prompt.weight = nn.parameter.Parameter(init_g_prompt_value)
        init_e_prompts_value = torch.FloatTensor(self.args.ntasks, self.Le * self.config.d_model).uniform_(-0.5, 0.5)
        self.e_prompts = nn.Embedding(self.args.ntasks, self.Le * self.config.d_model)
        self.e_prompts.weight = nn.parameter.Parameter(init_e_prompts_value)
        self.keys = nn.Embedding(self.args.ntasks, self.config.d_model)

        self.lam = 1.0

    def get_prompt_extended_input_exclude_label(self,input_ids,attention_mask,labels):
        inputs_embeds = self.model.shared(input_ids)

        return inputs_embeds,attention_mask

    def _prepare_prompts(self, input_ids, task_id=None):
        
        inputs_embeds = self.model.shared(input_ids)
        
        q = self.model.encoder(inputs_embeds=inputs_embeds)[0][:, 0, :]
        if task_id is not None:
            sim = utils.sim_matrix(q, self.keys.weight[task_id].reshape(1, -1))
            e_prompt = self.e_prompts.weight[task_id].repeat(inputs_embeds.size(0), 1).reshape(
                inputs_embeds.size(0), -1, self.config.d_model
            )
            matching_loss = sim.mean()
        else:
            sim = utils.sim_matrix(q, self.keys.weight)
            selection = torch.topk(sim, 1, dim=1)
            e_prompt = self.e_prompts.weight[selection.indices].reshape(
                inputs_embeds.size(0), -1, self.config.d_model
            )
            matching_loss = selection.values.mean()
        
        g_prompt = self.g_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        return inputs_embeds, g_prompt, e_prompt, matching_loss
    
    def forward(
        self,
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
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        my_loss = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # for evaluation and training, it's assumed to use the same prompt within the batch.
        inputs_embeds, g_prompt, e_prompt, matching_loss = self._prepare_prompts(input_ids)
        if labels is not None:
            
            use_cache = False

            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        if self.args.is_transfer:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    encoder_outputs=encoder_outputs,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    cross_attn_head_mask=cross_attn_head_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    g_prompt=g_prompt,
                    e_prompt=e_prompt,
                    **kwargs
                )

        else:
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                g_prompt=g_prompt,
                e_prompt=e_prompt,
                **kwargs
            )
        
        lm_logits = self.model.encoder.invertible_adapters_forward(outputs[0], rev=True)
        if self.args.is_transfer:
            lm_logits = self.readouts[self.args.eval_t](lm_logits) + self.final_logits_bias
        else:
            lm_logits = self.lm_head(lm_logits) + self.final_logits_bias
        
        masked_lm_loss = matching_loss * self.lam

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))


        if my_loss is not None:
            masked_lm_loss += my_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        return MyOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            last_hidden_state=outputs.last_hidden_state
        )