from torch.nn.functional import gelu, elu
import torch.nn.functional as F
import torch.nn as nn
import math
import torch
from collections import OrderedDict
from transformers import RobertaForSequenceClassification
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.alpha_lr.parameters(), self.args.grad_clip_norm)
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
                self.alpha_lr.append(nn.Parameter(alpha_init * torch.ones(p.shape, requires_grad=True)))
        else:
            self.alpha_lr = saved_alpha_lr
            
        self.opt_lr = torch.optim.SGD(list(self.alpha_lr.parameters()), lr=1e-1)

    def prepare(self, accelerator):
        self.alpha_lr, self.opt_lr = accelerator.prepare(self.alpha_lr, self.opt_lr)

    def meta_loss(self, fast_weights, batch, i, is_train=True):

        if fast_weights is None:
            fast_weights = OrderedDict(self.model.named_parameters())

        if self.args.task_name in self.args.ner_datasets:
            raise NotImplementedError
        
        elif self.args.task_name in self.args.classification:
            loss, logits = self.functional_robertaforsequenceclassification(
                fast_weights, self.config, 
                input_ids=batch['input_ids'][i].unsqueeze(0), 
                attention_mask=batch['attention_mask'][i].unsqueeze(0), 
                is_train=is_train, 
                labels=batch['cls_labels'][i].unsqueeze(0), 
                task=batch['task'][i].unsqueeze(0)
            )
        else:
            raise NotImplementedError

        return MyRobertaOutput(
            loss = loss,
            contrast_loss = None,
            sum_loss = None,
            logits = logits,
            ppl = None,
            hidden_states = None
        )
    
    def inner_update(self, fast_weights, batch, i, is_train=True):
        """
        Update the fast weights using the current samples and return the updated fast
        """
        outputs = self.meta_loss(fast_weights, batch, i, is_train=is_train)
        loss = outputs.loss

        if fast_weights is None:
            fast_weights = OrderedDict(self.model.named_parameters())

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        grads = torch.autograd.grad(loss, list(fast_weights.values()), create_graph=True, retain_graph=True, allow_unused=True)

        for i in range(len(grads)):
            if grads[i] is not None:
                torch.clamp(grads[i], min=-self.args.grad_clip_norm, max=self.args.grad_clip_norm)
        
        fast_weights_var = list(
                map(lambda p: p[1][0] - p[0] * nn.functional.relu(p[1][1]) if p[0] is not None else p[1][0], zip(grads, zip(list(fast_weights.values()), self.alpha_lr))))
        
        fast_weights = OrderedDict({k: v for (k,v) in zip(list(fast_weights.keys()), fast_weights_var)})
        
        return fast_weights

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
                x = F.dropout(x[:, 0, :], p=config.hidden_dropout_prob, training=is_train)
                x = F.linear(x, fast_weights[f'classifiers.{t}.dense.weight'], fast_weights[f'classifiers.{t}.dense.bias'])
                x = torch.tanh(x)
                x = F.dropout(x, p=config.hidden_dropout_prob, training=is_train)
                logit = F.linear(x, fast_weights[f'classifiers.{t}.out_proj.weight'], fast_weights[f'classifiers.{t}.out_proj.bias'])
                num_labels = self.taskcla[t][1]
                cur_loss = loss_fct(logit.view(-1, num_labels), labels[t_id].view(-1))
                loss += cur_loss
                logits.append(logit)
            
            loss = loss / len(task)
        else:
            logits = []
            for t in range(self.args.ntasks):
                x = F.linear(sequence_output, fast_weights[f'classifiers.{t}.dense.weight'], fast_weights[f'classifiers.{t}.dense.bias'])
                x = F.dropout(x, p=config.hidden_dropout_prob, training=is_train)
                logit = F.linear(x, fast_weights[f'classifiers.{t}.out_proj.weight'], fast_weights[f'classifiers.{t}.out_proj.bias'])
                logits.append(logit)
            logits = torch.cat(logits, dim=-1)

        return loss, logits

def functional_roberta(fast_weights, config, input_ids=None, attention_mask=None, token_type_ids=None,
                    position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                    encoder_attention_mask=None, is_train = True):

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

    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        if config.is_decoder:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            causal_mask = causal_mask.to(torch.long)
            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape, attention_mask.shape))

    extended_attention_mask = extended_attention_mask.to(dtype=next((p for p in fast_weights.values())).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(encoder_hidden_shape,
                                                                                                                           encoder_attention_mask.shape))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next((p for p in fast_weights.values())).dtype) 
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
    else:
        encoder_extended_attention_mask = None

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.to(dtype=next((p for p in fast_weights.values())).dtype)
    else:
        head_mask = [None] * config.num_hidden_layers
    
    embedding_output = functional_embedding(fast_weights, config, input_ids, position_ids, 
                                            token_type_ids, inputs_embeds, is_train = is_train)
    
    encoder_outputs = functional_encoder(fast_weights, config, embedding_output,
                                   attention_mask=extended_attention_mask,
                                   head_mask=head_mask, encoder_hidden_states=encoder_hidden_states,
                                   encoder_attention_mask=encoder_extended_attention_mask, is_train = is_train)
    
    sequence_output = encoder_outputs
    outputs = (sequence_output,)
    return outputs


def functional_embedding(fast_weights, config, input_ids, position_ids, 
                         token_type_ids, inputs_embeds = None, is_train = True):

    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]

    seq_length = input_shape[1]
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if position_ids is None:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    if inputs_embeds is None:
        inputs_embeds = F.embedding(input_ids, fast_weights['roberta.embeddings.word_embeddings.weight'], padding_idx = 0)
    
    position_embeddings = F.embedding(position_ids, fast_weights['roberta.embeddings.position_embeddings.weight'])

    token_type_embeddings = F.embedding(token_type_ids, fast_weights['roberta.embeddings.token_type_embeddings.weight'])

    embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    
    embeddings = F.layer_norm(embeddings, [config.hidden_size], 
                              weight=fast_weights['roberta.embeddings.LayerNorm.weight'],
                              bias=fast_weights['roberta.embeddings.LayerNorm.bias'],
                              eps=config.layer_norm_eps)

    embeddings = F.dropout(embeddings, p=config.hidden_dropout_prob, training = is_train)
    
    return embeddings

    
def transpose_for_scores(config, x):
    new_x_shape = x.size()[:-1] + (config.num_attention_heads, int(config.hidden_size / config.num_attention_heads))
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

def functional_self_attention(fast_weights, config, layer_idx,
                              hidden_states, attention_mask, head_mask, 
                              encoder_hidden_states, encoder_attention_mask,
                              is_train = True):
    
    attention_head_size = int(config.hidden_size / config.num_attention_heads)
    all_head_size = config.num_attention_heads * attention_head_size
    
    mixed_query_layer = F.linear(hidden_states,
                                fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.query.weight'],
                                fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.query.bias'])
    
    if encoder_hidden_states is not None:
        mixed_key_layer = F.linear(encoder_hidden_states,
                                fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.key.weight'],
                                fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.key.bias'])
        mixed_value_layer = F.linear(encoder_hidden_states,
                                fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.value.weight'],
                                fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.value.bias'])
        attention_mask = encoder_attention_mask
    else:
        mixed_key_layer   = F.linear(hidden_states,
                                fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.key.weight'],
                                fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.key.bias'])
        mixed_value_layer = F.linear(hidden_states,
                                fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.value.weight'],
                                fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.self.value.bias'])

    query_layer = transpose_for_scores(config, mixed_query_layer)
    key_layer   = transpose_for_scores(config, mixed_key_layer)
    value_layer = transpose_for_scores(config, mixed_value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        
    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

    if is_train:
        attention_probs = F.dropout(attention_probs, p= config.attention_probs_dropout_prob)

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
                              is_train = True):
    
    hidden_states = F.linear(hidden_states,
                            fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.output.dense.weight'],
                            fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.output.dense.bias'])

    hidden_states = F.dropout(hidden_states, p=config.hidden_dropout_prob, training = is_train)
    hidden_states = F.layer_norm(hidden_states + input_tensor, [config.hidden_size],
                              weight=fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.output.LayerNorm.weight'],
                              bias=fast_weights['roberta.encoder.layer.'+layer_idx+'.attention.output.LayerNorm.bias'],
                              eps=config.layer_norm_eps)
    
    return hidden_states    


def functional_attention(fast_weights, config, layer_idx,
                         hidden_states, attention_mask=None, head_mask=None,
                         encoder_hidden_states=None, encoder_attention_mask=None,
                         is_train = True):
    
    self_outputs = functional_self_attention(fast_weights, config, layer_idx,
                                             hidden_states, attention_mask, head_mask, 
                                             encoder_hidden_states, encoder_attention_mask, is_train)
    
    attention_output = functional_out_attention(fast_weights, config, layer_idx,
                                                self_outputs, hidden_states, is_train)
    return attention_output

def functional_intermediate(fast_weights, config, layer_idx, hidden_states, is_train = True):
    weight_name = 'roberta.encoder.layer.' + layer_idx + '.intermediate.dense.weight'
    bias_name   = 'roberta.encoder.layer.' + layer_idx + '.intermediate.dense.bias'
    hidden_states = F.linear(hidden_states, fast_weights[weight_name], fast_weights[bias_name])
    hidden_states = gelu(hidden_states)
    
    return hidden_states


def functional_output(fast_weights, config, layer_idx, hidden_states, input_tensor, is_train = True):

    hidden_states = F.linear(hidden_states, 
                             fast_weights['roberta.encoder.layer.'+layer_idx+'.output.dense.weight'], 
                             fast_weights['roberta.encoder.layer.'+layer_idx+'.output.dense.bias'])
    
    hidden_states = F.dropout(hidden_states, p=config.hidden_dropout_prob, training = is_train)
    hidden_states = F.layer_norm(hidden_states + input_tensor, [config.hidden_size],
                              weight=fast_weights['roberta.encoder.layer.'+layer_idx+'.output.LayerNorm.weight'],
                              bias=fast_weights['roberta.encoder.layer.'+layer_idx+'.output.LayerNorm.bias'],
                              eps=config.layer_norm_eps)
    return hidden_states

def functional_layer(fast_weights, config, layer_idx, hidden_states, attention_mask,
                     head_mask, encoder_hidden_states, encoder_attention_mask, is_train = True):
    
    self_attention_outputs = functional_attention(fast_weights, config, layer_idx,
                                                  hidden_states, attention_mask, head_mask,
                                                  encoder_hidden_states, encoder_attention_mask,is_train)
    
    attention_output = self_attention_outputs
    intermediate_output = functional_intermediate(fast_weights, config, layer_idx, attention_output, is_train)
    layer_output = functional_output(fast_weights, config, layer_idx, 
                                     intermediate_output, attention_output, is_train)
    
    return layer_output
    

def functional_encoder(fast_weights, config , hidden_states, attention_mask,
                       head_mask, encoder_hidden_states, encoder_attention_mask, is_train = True):
    
    for i in range(0,config.num_hidden_layers):
        layer_outputs = functional_layer(fast_weights, config, str(i),
                                         hidden_states, attention_mask, head_mask[i], 
                                         encoder_hidden_states, encoder_attention_mask, is_train)
        hidden_states = layer_outputs
        
    outputs = hidden_states
    return outputs

if __name__ == '__main__':
    
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    fast_weights = OrderedDict(model.named_parameters())

    input_ids = torch.Tensor([[  101,  1303,  1110,  1199,  3087,  1106,  4035, 13775,   102],
                              [  101,   178,  1274,  1204,  1176,  1115,  4170,   182,   102]]).to(torch.long)
    token_type_ids = None
    attention_mask = torch.Tensor([[1,  1,  1,  1,  1,  1,  1, 1, 1],
                                   [1,  1,  1,  1,  1,  1,  1, 1, 1]]).to(torch.long)
    
    print(functional_roberta(fast_weights, model.config, input_ids=input_ids, attention_mask=attention_mask, 
                    token_type_ids=token_type_ids,is_train = True))