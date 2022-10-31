import torch
import torch.nn as nn
from networks.bart import MyBartForConditionalGeneration,MyBartForSequenceClassification,MyBartForTokenClassification
from networks.bart import shift_tokens_right
from networks.roberta import MyRobertaForSequenceClassification, MyRobertaForTokenClassification

class BARTPromptTuningMixinCommon:

    def initialize_soft_prompt(self, n_tokens):
        if 'one' in self.args.baseline:
            self.prompt_embedding = nn.parameter.Parameter(
                self.model.shared.weight[:n_tokens].clone().detach())

        elif 'l2p' in self.args.baseline:
            init_prompt_value = torch.FloatTensor(self.args.M, self.args.Lp * self.config.hidden_size).uniform_(-0.5, 0.5)
            self.prompt_pool = nn.Embedding(self.args.M, self.args.Lp * self.config.hidden_size)
            self.prompt_pool.weight = nn.parameter.Parameter(init_prompt_value)

        else:
            raise NotImplementedError

        # self.prompt_embedding = nn.parameter.Parameter(
        #     torch.FloatTensor(n_tokens, self.model.shared.weight.size(1)).uniform_(-0.5, 0.5))

    def set_soft_prompt_embeds(self, soft_prompt_embeds):
        self.prompt_embedding = nn.parameter.Parameter(soft_prompt_embeds.clone().detach())

    def get_soft_params(self):
        return self.prompt_embedding

    def prepare_inputs_for_generation(self, input_ids, past=None, *args, **kwargs):
        input_ids = input_ids.to(self.device)
        # Drop 'past' to make things easier for us later
        return super().prepare_inputs_for_generation(input_ids, None, *args, **kwargs)

    def _cat_prompt_embedding_to_input(self, input_ids):
        inputs_embeds = self.model.shared(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            ie = inputs_embeds.unsqueeze(0)
        else:
            ie = inputs_embeds

        # prompt_embedding = self.model.drop(self.prompt_embedding)
        prompt_embedding = self.prompt_embedding

        inputs_embeds = torch.cat([prompt_embedding.repeat(ie.size(0), 1, 1),
                                   ie],
                                   dim=1)

        return inputs_embeds

    def _extend_labels(self, labels):
        n_tokens = self.prompt_embedding.shape[-2]

        if len(list(labels.shape)) == 1:
            lb = labels.unsqueeze(0)
        else:
            lb = labels

        # Add '-100's (prevent loss calculation where the learned embed would be)
        n_batches = lb.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), -100).to(self.device), lb], dim=1)

    def _extend_attention_mask(self, attention_mask):
        n_tokens = self.prompt_embedding.shape[-2]

        if len(list(attention_mask.shape)) == 1:
            am = attention_mask.unsqueeze(0)
        else:
            am = attention_mask

        n_batches = am.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), 1).to(self.device), am], dim=1)

    def get_prompt_extended_input_exclude_label(self,input_ids,attention_mask):
        inputs_embeds = self._cat_prompt_embedding_to_input(input_ids)
        attention_mask = self._extend_attention_mask(attention_mask)

        return inputs_embeds,attention_mask


class BARTPromptTuningMixinGeneration(BARTPromptTuningMixinCommon):

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
        **kwargs
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_prompt_embedding_to_input(input_ids)

        if labels is not None:
            labels = self._extend_labels(labels)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)


        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            input_ids=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
            **kwargs
        )


class BARTPromptTuningMixinClassification(BARTPromptTuningMixinCommon):

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
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None
    ):


        if input_ids is not None:
            inputs_embeds = self._cat_prompt_embedding_to_input(input_ids)
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)
            # BART classification is not natrually suppoer iunput_embds


        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)


        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task=task
        )

class RobertaPromptTuningMixinCommon:
    def initialize_soft_prompt(self, n_tokens):
        if 'one' in self.args.baseline:
            self.prompt_embedding = nn.parameter.Parameter(
                self.roberta.embeddings.word_embeddings.weight[:n_tokens].clone().detach())

        elif 'l2p' in self.args.baseline:
            init_prompt_value = torch.FloatTensor(self.args.M, self.args.Lp * self.config.hidden_size).uniform_(-0.5, 0.5)
            self.prompt_pool = nn.Embedding(self.args.M, self.args.Lp * self.config.hidden_size)
            self.prompt_pool.weight = nn.parameter.Parameter(init_prompt_value)

        else:
            raise NotImplementedError

    def set_soft_prompt_embeds(self, soft_prompt_embeds):
        self.prompt_embedding = nn.parameter.Parameter(soft_prompt_embeds.clone().detach())

    def get_soft_params(self):
        return self.prompt_embedding

    def prepare_inputs_for_generation(self, input_ids, past=None, *args, **kwargs):
        input_ids = input_ids.to(self.device)
        # Drop 'past' to make things easier for us later
        return super().prepare_inputs_for_generation(input_ids, None, *args, **kwargs)

    def _cat_prompt_embedding_to_input(self, input_ids):
        inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            ie = inputs_embeds.unsqueeze(0)
        else:
            ie = inputs_embeds

        # prompt_embedding = self.model.drop(self.prompt_embedding)
        prompt_embedding = self.prompt_embedding

        inputs_embeds = torch.cat([prompt_embedding.repeat(ie.size(0), 1, 1),
                                   ie],
                                   dim=1)

        return inputs_embeds

    def _extend_labels(self, labels):
        n_tokens = self.prompt_embedding.shape[-2]

        if len(list(labels.shape)) == 1:
            lb = labels.unsqueeze(0)
        else:
            lb = labels

        # Add '-100's (prevent loss calculation where the learned embed would be)
        n_batches = lb.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), -100).to(self.device), lb], dim=1)

    def _extend_attention_mask(self, attention_mask):
        n_tokens = self.prompt_embedding.shape[-2]

        if len(list(attention_mask.shape)) == 1:
            am = attention_mask.unsqueeze(0)
        else:
            am = attention_mask

        n_batches = am.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), 1).to(self.device), am], dim=1)

    def get_prompt_extended_input_exclude_label(self,input_ids,attention_mask):
        inputs_embeds = self._cat_prompt_embedding_to_input(input_ids)
        attention_mask = self._extend_attention_mask(attention_mask)

        return inputs_embeds,attention_mask

class RobertaPromptTuningMixinClassification(RobertaPromptTuningMixinCommon):
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
    ):


        if input_ids is not None:
            inputs_embeds = self._cat_prompt_embedding_to_input(input_ids)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        # Drop most of the args for now
        return super().forward(
            input_ids=None, # TODO: we don't want to use input_ids , to debug.
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task=task,
            my_loss=my_loss,
            nsp_labels=nsp_labels,
        )


class MyBartForConditionalGenerationSoftPromptTunning(BARTPromptTuningMixinGeneration, MyBartForConditionalGeneration):
    def __init__(self, config, **kwargs):
        BARTPromptTuningMixinGeneration.__init__(self)
        MyBartForConditionalGeneration.__init__(self, config, **kwargs)


class MyBartForSequenceClassificationSoftPromptTunning(BARTPromptTuningMixinClassification, MyBartForSequenceClassification):
    def __init__(self, config, **kwargs):
        BARTPromptTuningMixinClassification.__init__(self)
        MyBartForSequenceClassification.__init__(self, config, **kwargs)
        self.config = config
        self.args = kwargs['args']

class MyBartForTokenClassificationSoftPromptTunning(BARTPromptTuningMixinClassification, MyBartForTokenClassification):
    def __init__(self, config, **kwargs):
        BARTPromptTuningMixinClassification.__init__(self)
        MyBartForTokenClassification.__init__(self, config, **kwargs)
        self.config = config
        self.args = kwargs['args']

class MyBartForSequenceClassificationSoftPromptTunning(BARTPromptTuningMixinClassification, MyBartForSequenceClassification):
    def __init__(self, config, **kwargs):
        BARTPromptTuningMixinClassification.__init__(self)
        MyBartForSequenceClassification.__init__(self, config, **kwargs)
        self.config = config
        self.args = kwargs['args']

class MyRobertaForTokenClassificationSoftPromptTunning(RobertaPromptTuningMixinClassification, MyRobertaForTokenClassification):
    def __init__(self, config, **kwargs):
        RobertaPromptTuningMixinClassification.__init__(self)
        MyRobertaForTokenClassification.__init__(self, config, **kwargs)
        self.config = config
        self.args = kwargs['args']

class MyRobertaForSequenceClassificationSoftPromptTunning(RobertaPromptTuningMixinClassification, MyRobertaForSequenceClassification):
    def __init__(self, config, **kwargs):
        RobertaPromptTuningMixinClassification.__init__(self)
        MyRobertaForSequenceClassification.__init__(self, config, **kwargs)
        self.config = config
        self.args = kwargs['args']