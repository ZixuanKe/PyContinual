
import torch
import torch.nn as nn
from networks.bart import MyBartForConditionalGeneration,MyBartForSequenceClassification,MyBartForTokenClassification
from networks.bart import shift_tokens_right
from networks.roberta import MyRobertaForSequenceClassification, MyRobertaForTokenClassification
from networks.prompt.tuning import BARTPromptTuningMixinCommon, RobertaPromptTuningMixinCommon
from utils import utils


class BARTL2PMixinConditionalGneration(BARTPromptTuningMixinCommon):

    def _cat_selected_prompt_to_input(self, input_ids,labels):
        """
        Selects prompts which minimize the matching function and concatenates them to the inputs.
        x_p = [P_s1; ... ; P_sN; x_e]
        """
        inputs_embeds = self.model.shared(input_ids)

        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

        # Use the frozen pre-trained model to get the query features: q(x) = f(x)[0,:]
        q = self.model(decoder_input_ids=decoder_input_ids,inputs_embeds=inputs_embeds)[0][:, 0, :]
        sim = utils.sim_matrix(q, self.keys.weight)
        selection = torch.topk(sim, self.N, dim=1)
        matching_loss = selection.values.sum(dim=1).mean()
        selected_prompt = self.prompt_pool.weight[selection.indices].reshape(
            -1, self.Lp * self.N, self.config.hidden_size).to(input_ids.device)

        inputs_embeds = torch.cat([selected_prompt, inputs_embeds], dim=1)

        return inputs_embeds, matching_loss

    def _extend_attention_mask(self, attention_mask):
        """
        Extends attention_mask to match the input_ids's shape.
        """
        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [
                torch.full(
                    (n_batches, self.Lp * self.N), 1).to(
                    attention_mask.device).long(),
                attention_mask
            ],
            dim=1,
        )


    def get_prompt_extended_input_exclude_label(self,input_ids,attention_mask,labels):
        inputs_embeds, matching_loss = self._cat_selected_prompt_to_input(input_ids,labels)
        attention_mask = self._extend_attention_mask(attention_mask)

        return inputs_embeds,attention_mask

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
        matching_loss = 0
        if input_ids is not None:
            inputs_embeds, matching_loss = self._cat_selected_prompt_to_input(input_ids,labels)
            matching_loss = self.lam * matching_loss

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
            my_loss = matching_loss,
            **kwargs
        )

class RobertaL2PMixinClassification(RobertaPromptTuningMixinCommon):
    def _cat_selected_prompt_to_input(self, input_ids):
        """
        Selects prompts which minimize the matching function and concatenates them to the inputs.
        x_p = [P_s1; ... ; P_sN; x_e]
        """
        inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        # Use the frozen pre-trained model to get the query features: q(x) = f(x)[0,:]
        q = self.roberta(inputs_embeds=inputs_embeds)[0][:, 0, :]
        sim = utils.sim_matrix(q, self.keys.weight)
        selection = torch.topk(sim, self.N, dim=1)
        matching_loss = selection.values.sum(dim=1).mean()
        selected_prompt = self.prompt_pool.weight[selection.indices].reshape(
            -1, self.Lp * self.N, self.config.hidden_size).to(input_ids.device)

        inputs_embeds = torch.cat([selected_prompt, inputs_embeds], dim=1)

        return inputs_embeds, matching_loss

    def _extend_attention_mask(self, attention_mask):
        """
        Extends attention_mask to match the input_ids's shape.
        """

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [
                torch.full(
                    (n_batches, self.Lp * self.N), 1).to(
                    attention_mask.device).long(),
                attention_mask
            ],
            dim=1,
        )

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
        output_hidden_states=True,
        return_dict=None,
        my_loss=None,
        task=None,
        nsp_labels=None
    ):


        if input_ids is not None:
            inputs_embeds, matching_loss = self._cat_selected_prompt_to_input(input_ids)
            matching_loss = self.lam * matching_loss


        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            input_ids=None,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task=task,
            my_loss=matching_loss,
            nsp_labels=nsp_labels
        )
    
class BARTL2PMixinClassification(BARTPromptTuningMixinCommon):

    def _cat_selected_prompt_to_input(self, input_ids):
        """
        Selects prompts which minimize the matching function and concatenates them to the inputs.
        x_p = [P_s1; ... ; P_sN; x_e]
        """
        inputs_embeds = self.model.shared(input_ids)
        decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)

        if len(list(inputs_embeds)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # Use the frozen pre-trained model to get the query features: q(x) = f(x)[0,:]
        q = self.model(decoder_input_ids=decoder_input_ids,inputs_embeds=inputs_embeds)[0][:, 0, :]
        sim = utils.sim_matrix(q, self.keys.weight)
        selection = torch.topk(sim, self.N, dim=1)
        matching_loss = selection.values.sum(dim=1).mean()
        selected_prompt = self.prompt_pool.weight[selection.indices].reshape(
            -1, self.Lp * self.N, self.config.hidden_size).to(input_ids.device)

        inputs_embeds = torch.cat([selected_prompt, inputs_embeds], dim=1)

        return inputs_embeds, matching_loss

    def _extend_attention_mask(self, attention_mask):
        """
        Extends attention_mask to match the input_ids's shape.
        """

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [
                torch.full(
                    (n_batches, self.Lp * self.N), 1).to(
                    attention_mask.device).long(),
                attention_mask
            ],
            dim=1,
        )

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
            inputs_embeds, matching_loss = self._cat_selected_prompt_to_input(input_ids)
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)
            # BART classification is not natrually suppoer iunput_embds
            matching_loss = self.lam * matching_loss


        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        # print('matching_loss: ',matching_loss)

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
            task=task,
            my_loss=matching_loss
        )



class BARTL2PMixinAll(BARTPromptTuningMixinCommon):

    def _cat_selected_prompt_to_input_generation(self, input_ids,labels):
        """
        Selects prompts which minimize the matching function and concatenates them to the inputs.
        x_p = [P_s1; ... ; P_sN; x_e]
        """


        inputs_embeds = self.model.shared(input_ids).squeeze(0)

        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )


        # Use the frozen pre-trained model to get the query features: q(x) = f(x)[0,:]
        q = self.model(decoder_input_ids=decoder_input_ids,inputs_embeds=inputs_embeds)[0][:, 0, :]
        sim = utils.sim_matrix(q, self.keys.weight)
        selection = torch.topk(sim, self.N, dim=1)
        matching_loss = selection.values.sum(dim=1).mean()
        selected_prompt = self.prompt_pool.weight[selection.indices].reshape(
            -1, self.Lp * self.N, self.config.hidden_size).to(input_ids.device)

        inputs_embeds = torch.cat([selected_prompt, inputs_embeds], dim=1)

        return inputs_embeds, matching_loss




    def _cat_selected_prompt_to_input_classification(self, input_ids):
        """
        Selects prompts which minimize the matching function and concatenates them to the inputs.
        x_p = [P_s1; ... ; P_sN; x_e]
        """
        inputs_embeds = self.model.shared(input_ids)
        decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # Use the frozen pre-trained model to get the query features: q(x) = f(x)[0,:]
        q = self.model(decoder_input_ids=decoder_input_ids,inputs_embeds=inputs_embeds)[0][:, 0, :]
        sim = utils.sim_matrix(q, self.keys.weight)
        selection = torch.topk(sim, self.N, dim=1)
        matching_loss = selection.values.sum(dim=1).mean()
        selected_prompt = self.prompt_pool.weight[selection.indices].reshape(
            -1, self.Lp * self.N, self.config.hidden_size).to(input_ids.device)

        inputs_embeds = torch.cat([selected_prompt, inputs_embeds], dim=1)

        return inputs_embeds, matching_loss

    def _extend_attention_mask(self, attention_mask):
        """
        Extends attention_mask to match the input_ids's shape.
        """

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [
                torch.full(
                    (n_batches, self.Lp * self.N), 1).to(
                    attention_mask.device).long(),
                attention_mask
            ],
            dim=1,
        )


    def get_prompt_extended_input_exclude_label(self,input_ids,attention_mask,labels):
        inputs_embeds, matching_loss = self._cat_selected_prompt_to_input_generation(input_ids,labels) # must be generation
        attention_mask = self._extend_attention_mask(attention_mask)

        return inputs_embeds,attention_mask

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
            task=None,
            **kwargs
    ):



        if self.args.task_name in self.args.generation:

            matching_loss = 0
            if input_ids is not None:
                inputs_embeds, matching_loss = self._cat_selected_prompt_to_input_generation(input_ids,labels)

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
                my_loss = matching_loss,
                **kwargs
            )

        elif self.args.task_name in self.args.classification:

            if input_ids is not None:
                inputs_embeds, matching_loss = self._cat_selected_prompt_to_input_classification(input_ids)
                decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id,
                                                       self.config.decoder_start_token_id)
                # BART classification is not natrually suppoer iunput_embds

            if attention_mask is not None:
                attention_mask = self._extend_attention_mask(attention_mask)

            matching_loss = self.lam * matching_loss
            # print('matching_loss: ',matching_loss)

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
                task=task,
                my_loss=matching_loss
            )



class MyBartForTokenClassificationSoftL2P(BARTL2PMixinClassification, MyBartForTokenClassification):
    def __init__(self, config, taskcla, args, **kwargs):
        BARTL2PMixinClassification.__init__(self)
        MyBartForTokenClassification.__init__(self, config, taskcla, args, **kwargs)

        """
        Prompt pool (P): {P_1, ..., P_M}, P_i [Lp, embed_dim]
        Learnable key: {(k_1, P_1), ..., (k_M, P_M)}, k_i [last_hidden_dim]
        """

        self.taskcla = taskcla
        self.config = config
        self.args = args
        self.tokenizer = None
        self.M = args.M
        self.N = args.N
        self.Lp = args.Lp
        self.keys = nn.Embedding(self.M, self.config.hidden_size)

        # Hyperparameter for the loss function.
        self.lam = 0.5  # Follow the original paper.

class MyBartForSequenceClassificationSoftL2P(BARTL2PMixinClassification, MyBartForSequenceClassification):
    def __init__(self, config, taskcla, args, **kwargs):
        BARTL2PMixinClassification.__init__(self)
        MyBartForSequenceClassification.__init__(self, config, taskcla, args, **kwargs)

        """
        Prompt pool (P): {P_1, ..., P_M}, P_i [Lp, embed_dim]
        Learnable key: {(k_1, P_1), ..., (k_M, P_M)}, k_i [last_hidden_dim]
        """

        self.taskcla = taskcla
        self.config = config
        self.args = args
        self.tokenizer = None
        self.M = args.M
        self.N = args.N
        self.Lp = args.Lp
        self.keys = nn.Embedding(self.M, self.config.hidden_size)

        # Hyperparameter for the loss function.
        self.lam = 0.5  # Follow the original paper.


class MyBartForConditionalGenerationSoftL2P(BARTL2PMixinConditionalGneration, MyBartForConditionalGeneration):
    def __init__(self, config, taskcla, args, **kwargs):
        BARTL2PMixinConditionalGneration.__init__(self)
        MyBartForConditionalGeneration.__init__(self, config, taskcla, args, **kwargs)

        """
        Prompt pool (P): {P_1, ..., P_M}, P_i [Lp, embed_dim]
        Learnable key: {(k_1, P_1), ..., (k_M, P_M)}, k_i [last_hidden_dim]
        """

        self.taskcla = taskcla
        self.config = config
        self.args = args
        self.tokenizer = None
        self.M = args.M
        self.N = args.N
        self.Lp = args.Lp
        self.keys = nn.Embedding(self.M, self.config.hidden_size)

        # Hyperparameter for the loss function.
        self.lam = 0.5  # Follow the original paper.


class MyRobertaForTokenClassificationSoftL2P(RobertaL2PMixinClassification, MyRobertaForTokenClassification):
    def __init__(self, config, taskcla, args, **kwargs):
        RobertaL2PMixinClassification.__init__(self)
        MyRobertaForTokenClassification.__init__(self, config, taskcla, args, **kwargs)

        """
        Prompt pool (P): {P_1, ..., P_M}, P_i [Lp, embed_dim]
        Learnable key: {(k_1, P_1), ..., (k_M, P_M)}, k_i [last_hidden_dim]
        """

        self.taskcla = taskcla
        self.config = config
        self.args = args
        self.tokenizer = None
        self.M = args.M
        self.N = args.N
        self.Lp = args.Lp
        self.keys = nn.Embedding(self.M, self.config.hidden_size)

        # Hyperparameter for the loss function.
        self.lam = 0.5  # Follow the original paper.

class MyRobertaForSequenceClassificationSoftL2P(RobertaL2PMixinClassification, MyRobertaForSequenceClassification):
    def __init__(self, config, taskcla, args, **kwargs):
        RobertaL2PMixinClassification.__init__(self)
        MyRobertaForSequenceClassification.__init__(self, config, taskcla, args, **kwargs)

        """
        Prompt pool (P): {P_1, ..., P_M}, P_i [Lp, embed_dim]
        Learnable key: {(k_1, P_1), ..., (k_M, P_M)}, k_i [last_hidden_dim]
        """

        self.taskcla = taskcla
        self.config = config
        self.args = args
        self.tokenizer = None
        self.M = args.M
        self.N = args.N
        self.Lp = args.Lp
        self.keys = nn.Embedding(self.M, self.config.hidden_size)

        # Hyperparameter for the loss function.
        self.lam = 0.5  # Follow the original paper.