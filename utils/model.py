import time
from pathlib import Path

import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from copy import deepcopy
from torch import nn
from networks.baselines.supsup import MultitaskMaskLinear

from networks.transformers.bart import MyBartForConditionalGeneration,MyBartForSequenceClassification,MyBartForTokenClassification
from networks.transformers.roberta import MyRobertaForSequenceClassification,MyRobertaForTokenClassification, MyRobertaForMaskedLM
from networks.transformers.bert import MyBertForSequenceClassification,MyBertForTokenClassification

import evaluate
from datasets import load_dataset, load_metric
from networks.finetune.model import MyModel
from networks.baselines.l2p import MyBartForSequenceClassificationSoftL2P,MyBartForConditionalGenerationSoftL2P,MyBartForTokenClassificationSoftL2P
from networks.prompt.inference import MyBartForConditionalGenerationSoftPromptInfer
from networks.prompt.tuning import MyBartForConditionalGenerationSoftPromptTunning,MyRobertaForTokenClassificationSoftPromptTunning, MyRobertaForSequenceClassificationSoftPromptTunning
import utils
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
    Adafactor
)
########################################################################################################################

def print_model_report(model):
    print('-' * 100)
    print(model)
    print('Dimensions =', end=' ')
    count = 0
    for p in model.parameters():
        print(p.size(), end=' ')
        count += np.prod(p.size())
    print()
    print('Num parameters = %s' % (human_format(count)))
    print('-' * 100)

    with open('para', 'a') as clocker_file:
        clocker_file.writelines((human_format(count)).replace('M','') + '\n')


    return count


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim, '=', end=' ')
        opt = optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n + ':', opt[n], end=', ')
        print()
    return


########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


########################################################################################################################

def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean = 0
    std = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean += image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded = mean.view(mean.size(0), mean.size(1), 1, 1).expand_as(image)
    for image, _ in loader:
        std += (image - mean_expanded).pow(2).sum(3).sum(2)

    std = (std / (len(dataset) * image.size(2) * image.size(3) - 1)).sqrt()

    return mean, std


########################################################################################################################

# for ACL
def report_tr(res, e, sbatch, clock0, clock1):
    # Training performance
    print(
        '| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train losses={:.3f} | T: loss={:.3f}, acc={:5.2f}% | D: loss={:.3f}, acc={:5.1f}%, '
        'Diff loss:{:.3f} |'.format(
            e + 1,
            1000 * sbatch * (clock1 - clock0) / res['size'],
            1000 * sbatch * (time.time() - clock1) / res['size'], res['loss_tot'],
            res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')


def report_val(res):
    # Validation performance
    print(
        ' Valid losses={:.3f} | T: loss={:.6f}, acc={:5.2f}%, | D: loss={:.3f}, acc={:5.2f}%, Diff loss={:.3f} |'.format(
            res['loss_tot'], res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')


########################################################################################################################


def cross_entropy(outputs, targets, exp=1, size_average=True, eps=1e-5):
    out = torch.nn.functional.softmax(outputs)
    tar = torch.nn.functional.softmax(targets)
    if exp != 1:
        out = out.pow(exp)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
    out = out + eps / out.size(1)
    out = out / out.sum(1).view(-1, 1).expand_as(out)
    ce = -(tar * out.log()).sum(1)
    if size_average:
        ce = ce.mean()
    return ce


########################################################################################################################

def set_req_grad(layer, req_grad):
    if hasattr(layer, 'weight'):
        layer.weight.requires_grad = req_grad
    if hasattr(layer, 'bias'):
        layer.bias.requires_grad = req_grad
    return


########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


########################################################################################################################

# we need to analysis the results, tensorboard

from torch.utils.tensorboard import SummaryWriter


# default `log_dir` is "runs" - we'll be more specific here
def setup_writer(name):
    writer = SummaryWriter(name)
    return writer


def project_layer(writer, features, class_labels):
    writer.add_embedding(features, metadata=class_labels)


def log_loss(writer, loss_name='training loss', scalar_value=None, global_step=None):
    # ...log the running loss
    writer.add_scalar(loss_name, scalar_value, global_step=global_step)


def log_gate(writer, loss_name='log gate', gate_sum_dict=None, global_step=None):
    # ...log the running loss
    writer.add_scalars(loss_name,
                       gate_sum_dict,
                       global_step=global_step)


########################################################################################################################

# distillation ########################

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss

def prepare_sequence_finetune(args):

    with open('./sequence/' + args.sequence_file, 'r') as f:
        datas = f.readlines()[args.idrandom]
        data = datas.split()

    args.task_name = data[args.ft_task]

    args.all_tasks = data


    if args.classifier_lr is None: args.classifier_lr = (5e-4 + 5e-3) / 2.0  # important for adapter
    if args.prompt_lr is None: args.prompt_lr = 5e-3 #  args.prompt_lr = {5e-2,5e-3}
    if args.adapter_lr is None: args.adapter_lr = 5e-4  # (5e-4 + 5e-3)/2.0

    # if 'DiaperChamp' in args.task_name:
    #     args.num_train_epochs = 50
    # TODO: for some reasons, this one is always bad

    if args.eval_only:
        output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(args.baseline)  + '/' + str(data[args.ft_task]) + "_lm/"
        ckpt = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str( args.baseline)  + '/' + str(data[args.ft_task]) + "_lm/"

        args.model_name_or_path = ckpt
        args.prev_output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(args.baseline)  + '/' + str(data[args.ft_task-1]) + "_lm/"

    else:
        output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(args.baseline)  + '/' + str(data[args.ft_task]) + "_lm/"
        ckpt = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(args.baseline)  + '/' + str(data[args.ft_task - 1]) + "_lm/"

        if args.ft_task > 0:
            args.prev_output = args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(args.baseline)  + '/' + str(data[args.ft_task-1]) + "_lm/"

        else:
            args.prev_output = output

        if args.ft_task == 0 or 'one' in args.baseline or 'mtl' in args.baseline or 'comb' in args.baseline:  # no pre-trained for the first
            args.model_name_or_path = args.base_model_name_or_path

        else:
            args.model_name_or_path = ckpt


    args.output_dir = output

    args.saved_output_dir = [args.base_dir + "/seq" + str(args.idrandom) + "/seed" + str(args.seed) + "/" + str(
        args.baseline) + '/' + str(data[t]) + "_lm/" for t in
                             range(args.ft_task)]


    print('saved_output_dir: ', args.saved_output_dir)
    print('output_dir: ', args.output_dir)
    print('prev_output: ', args.prev_output)
    print('args.model_name_or_path: ', args.model_name_or_path)

    args.summerization_datasets = ['nyt', 'stack', 'emails', 'reddit', 'icsi', 'ami']  # current 4 datasets
    args.asc_datasets = [
        'XuSemEval14_rest',
        'XuSemEval14_laptop',

        'Bing3domains_Speaker',
        'Bing3domains_Router',
        'Bing3domains_Computer',

        'Bing5domains_Nokia6610',
        'Bing5domains_NikonCoolpix4300',
        'Bing5domains_CreativeLabsNomadJukeboxZenXtra40GB',
        'Bing5domains_CanonG3',
        'Bing5domains_ApexAD2600Progressive',

        'Bing9domains_CanonPowerShotSD500',
        'Bing9domains_CanonS100',
        'Bing9domains_DiaperChamp',
        'Bing9domains_HitachiRouter',
        'Bing9domains_ipod',
        'Bing9domains_LinksysRouter',
        'Bing9domains_MicroMP3',
        'Bing9domains_Nokia6600',
        'Bing9domains_Norton']

    args.ccd_datasets = ['yahoo', 'yelp', 'amazon', 'dbpedia', 'agnews']
    args.dialogue_datasets = ['MWOZ_taxi', 'MWOZ_train', 'MWOZ_restaurant', 'MWOZ_hotel', 'MWOZ_attraction', 'sgd_services',
                         'sgd_flights', 'sgd_buses', 'sgd_ridesharing', 'sgd_rentalcars',
                         'sgd_homes', 'sgd_music', 'sgd_events', 'sgd_banks', 'sgd_hotels', 'sgd_calendar',
                         'sgd_media', 'sgd_movies', 'sgd_restaurants', 'sgd_alarm', 'sgd_weather',
                         'sgd_travel', 'sgd_payment', 'sgd_trains',
                         'tma_movie', 'tma_auto', 'tma_restaurant', 'tma_pizza', 'tma_uber', 'tma_coffee',
                         'tmb_hotel', 'tmb_movie', 'tmb_flight', 'tmb_sport',
                         'tmb_restaurant', 'tmb_music', 'tmb_food-ordering'
                         ]
    args.ner_datasets = ['ieer', 'btc', 'gum', 'ritter', 're3d', 'wnut2017', 'wikigold', 'conll2003', 'ontonote']


    args.classification = args.asc_datasets + args.ccd_datasets + args.ner_datasets
    args.generation = args.dialogue_datasets + args.summerization_datasets

    if args.lamb is None:
        args.lamb = 0.75
        if 'ewc' in args.baseline:
            args.lamb = 5000
            args.per_device_train_pool_batch_size = 8


    args.is_reference = False
    args.is_transfer = False


    # if we use default
    #TODO: supsup need to add (see the pool)
    if args.task_name in args.ccd_datasets:
        args.ntasks = 5
        args.max_source_length = 128
        args.num_train_epochs = 100

        if 'adapter_ctr' in args.baseline:
            args.per_device_train_batch_size = 16
        else:
            args.per_device_train_batch_size = 32

        args.per_device_eval_batch_size = 32
        args.sample_num_per_class = 500
        args.sample_cap = None
        args.num_beams = None
        args.val_max_target_length = None
        args.val_min_target_length = None
        args.no_repeat_ngram_size = None
        args.per_device_train_pool_batch_size = 8
        args.pad_to_max_length = True
        args.patient = 10

        if 'prompt' in args.baseline or 'l2p' in args.baseline:
            args.prompt_lr = 5e-3
            args.learning_rate = 5e-5

    if args.task_name in args.dialogue_datasets:
        args.max_source_length = 128

        if 'adapter_ctr' in args.baseline:
            args.num_train_epochs = 30
        else:
            args.num_train_epochs = 50
        args.per_device_train_batch_size = 16
        args.per_device_eval_batch_size = 32
        args.sample_num_per_class = None
        args.sample_cap = None
        args.num_beams = 4
        args.val_max_target_length = 200
        args.val_min_target_length = 30
        args.no_repeat_ngram_size = 3
        args.per_device_train_pool_batch_size = 2
        args.pad_to_max_length = True
        args.patient = 50

        if 'prompt' in args.baseline or 'l2p' in args.baseline:
            args.prompt_lr = 0.05
            args.learning_rate = 0.005


    if args.task_name in args.asc_datasets:
        #TODO: can we get siomilar results as before in RoBERTa?
        args.ntasks = 19
        args.max_source_length = 128
        # 100 is not neccessary good
        args.num_train_epochs = 100
        # if args.num_train_epochs is None:
        # if 'SemEval' in args.task_name:
        #     args.num_train_epochs = 20
        # else:
        #     args.num_train_epochs = 100

        if 'adapter_ctr' in args.baseline:
            args.per_device_train_batch_size = 16
        else:
            args.per_device_train_batch_size = 32
        args.per_device_eval_batch_size = 32
        args.sample_num_per_class = None
        args.sample_cap = None
        args.num_beams = None
        args.val_max_target_length = None
        args.val_min_target_length = None
        args.no_repeat_ngram_size = None
        args.per_device_train_pool_batch_size = 8
        args.pad_to_max_length = True
        args.patient = 5

        if 'prompt' in args.baseline or 'l2p' in args.baseline:
            args.prompt_lr = 5e-3
            args.learning_rate = 5e-5
            args.num_train_epochs = 100

    if args.task_name in args.ner_datasets:
        args.ntasks = 5
        args.max_source_length = 128

        args.num_train_epochs = 100

        if 'adapter_ctr' in args.baseline:
            args.per_device_train_batch_size = 16
        else:
            args.per_device_train_batch_size = 32
        args.per_device_eval_batch_size = 128
        args.sample_num_per_class = 50  #50 need to tune
        if 'sample_cap' in args.baseline:
            args.sample_cap = 200
        else:
            args.sample_cap = None
        args.num_beams = None
        args.val_max_target_length = None
        args.val_min_target_length = None
        args.no_repeat_ngram_size = None
        args.per_device_train_pool_batch_size = 8
        args.pad_to_max_length = True

        args.patient = 50

        if 'prompt' in args.baseline or 'l2p' in args.baseline:
            args.prompt_lr = 5e-3
            args.learning_rate = 5e-5
            args.num_train_epochs = 100

    if args.task_name in args.summerization_datasets:

        if 'adapter_ctr' in args.baseline:
            args.num_train_epochs = 30
            args.per_device_train_batch_size = 2

        else:
            args.num_train_epochs = 50
            args.per_device_train_batch_size = 4
        args.per_device_eval_batch_size = 16
        args.max_source_length = 1024 # important
        args.sample_cap = None
        args.sample_num_per_class = None
        args.num_beams = 4
        args.val_max_target_length = 200
        args.val_min_target_length = 30
        args.no_repeat_ngram_size = 3
        args.per_device_train_pool_batch_size = 2
        args.pad_to_max_length = True
        args.patient = 50

        if 'prompt' in args.baseline or 'l2p' in args.baseline:
            args.prompt_lr = 0.05
            args.learning_rate = 0.005
            if 'prompt' in args.baseline:
                args.max_source_length = 1000
            elif 'l2p' in args.baseline:
                args.max_source_length = 960


        if 'supsup' in args.baseline:
            if 'icsi' in args.task_name:
                args.adapter_lr = 5e-3
            else:
                args.adapter_lr = 5e-2

    args.fp16 = True
    args.warmup_ratio = 0.5

    return args



def load_my_metric(args):
    if args.task_name in args.dialogue_datasets:
        metric = evaluate.load("bleu")
    elif args.task_name in args.ner_datasets:
        metric = evaluate.load("seqeval")
    else:
        metric = load_metric("rouge")
    return metric

def _lookfor_model_prompt(taskcla,args, config):
    if args.task_name in args.ner_datasets:

        if 'l2p' in args.baseline:
            TUNE_MODEL = MyRobertaForTokenClassificationSoftL2P
            INFER_MODEL = MyRobertaForTokenClassificationSoftL2P
        else:
            TUNE_MODEL = MyRobertaForTokenClassificationSoftPromptTunning
            INFER_MODEL = MyRobertaForTokenClassificationSoftPromptTunning
    elif args.task_name in args.classification:

        if 'l2p' in args.baseline:
            TUNE_MODEL = MyRobertaForSequenceClassificationSoftL2P
            INFER_MODEL = MyRobertaForSequenceClassificationSoftL2P
        else:
            TUNE_MODEL = MyRobertaForSequenceClassificationSoftPromptTunning
            INFER_MODEL = MyRobertaForSequenceClassificationSoftPromptTunning
    elif args.task_name in args.generation:
        if 'l2p' in args.baseline:
            TUNE_MODEL =  MyBartForConditionalGenerationSoftL2P
            INFER_MODEL = MyBartForConditionalGenerationSoftPromptInfer
        else:
            TUNE_MODEL = MyBartForConditionalGenerationSoftPromptTunning
            INFER_MODEL = MyBartForConditionalGenerationSoftPromptInfer


    tune_model = TUNE_MODEL.from_pretrained(
        args.model_name_or_path,
        taskcla=taskcla,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        args=args
    )

    for n, p in tune_model.named_parameters():
        if 'classifier' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if 'one' in args.baseline:
        tune_model.initialize_soft_prompt(n_tokens=args.n_tokens)
    elif args.ft_task == 0 and 'l2p' in args.baseline:
        tune_model.initialize_soft_prompt(n_tokens=args.N * args.Lp)
    elif 'l2p' in args.baseline:# load the trianed prompt pool and keys
        print('loadding key and prompt_pool')
        tune_model.keys = torch.load(os.path.join(args.prev_output, 'keys'))
        tune_model.prompt_pool = torch.load(os.path.join(args.prev_output, 'prompt_pool'))

    infer_model = INFER_MODEL.from_pretrained(
        args.model_name_or_path,
        taskcla=taskcla,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        args=args
    )

    infer_model.model = tune_model.model

    for param in infer_model.parameters():
        param.requires_grad = False


    model = MyModel(model=tune_model,teacher=infer_model, args=args)

    return model


def _lookfor_model_ewc(taskcla,args, config):

    if args.task_name in args.ner_datasets:
        MODEL = MyBartForTokenClassification
    elif args.task_name in args.classification:
        MODEL = MyBartForSequenceClassification
    elif args.task_name in args.generation:
        MODEL = MyBartForConditionalGeneration

    model = MODEL.from_pretrained(
        args.model_name_or_path,
        taskcla=taskcla,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        args=args
    )
    teacher = MODEL.from_pretrained(
        args.model_name_or_path,
        taskcla=taskcla,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        args=args
    )

    for param in teacher.parameters():  # nothing is trainable in teacher
        param.requires_grad = False
    model = MyModel(model, teacher, args=args)

    return model


def _lookfor_model_others(taskcla,args, config):
    mlm_model = None


    if 'roberta' in args.model_name_or_path:
        if args.task_name in args.ner_datasets:
            MODEL = MyRobertaForTokenClassification #now everything is in the model

        elif args.task_name in args.classification:
            MODEL = MyRobertaForSequenceClassification

        if 'mlm' in args.baseline:
            MLM_MODEL = MyRobertaForMaskedLM

            mlm_model = MLM_MODEL.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                args=args
            )

    elif 'bart' in args.model_name_or_path:
        if args.task_name in args.ner_datasets:
            MODEL = MyBartForTokenClassification
        elif args.task_name in args.classification:
            MODEL = MyBartForSequenceClassification
        elif args.task_name in args.generation:
            MODEL = MyBartForConditionalGeneration
    elif 'bert' in args.model_name_or_path:
        if args.task_name in args.ner_datasets:
            MODEL = MyRobertaForTokenClassification
        elif args.task_name in args.classification:
            MODEL = MyBertForSequenceClassification


    model = MODEL.from_pretrained(
        args.model_name_or_path,
        taskcla=taskcla,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        args=args
    )


    if 'mlm' in args.baseline:
        mlm_model.roberta = model.roberta

    teacher = MODEL.from_pretrained(
        args.model_name_or_path,
        taskcla=taskcla,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        args=args
    )
    for param in teacher.parameters():  # nothing is trainable in teacher
        param.requires_grad = False

    model = MyModel(model, teacher=teacher, mlm_model=mlm_model, args=args)

    return model



def _lookfor_model_adapter(taskcla,args, config):

    mlm_model = None

    if 'roberta' in args.model_name_or_path:
        if args.task_name in args.ner_datasets:
            MODEL = MyRobertaForTokenClassification #now everything is in the model
        elif args.task_name in args.classification:
            MODEL = MyRobertaForSequenceClassification


        if 'mlm' in args.baseline:
            MLM_MODEL = MyRobertaForMaskedLM
            mlm_model = MLM_MODEL.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                args=args
            )

    elif 'bart' in args.model_name_or_path:
        if args.task_name in args.ner_datasets:
            MODEL = MyBartForTokenClassification
        elif args.task_name in args.classification:
            MODEL = MyBartForSequenceClassification
        elif args.task_name in args.generation:
            MODEL = MyBartForConditionalGeneration

    elif 'bert' in args.model_name_or_path:
        if args.task_name in args.ner_datasets:
            MODEL = MyBertForTokenClassification
        elif args.task_name in args.classification:
            MODEL = MyBertForSequenceClassification




    model = MODEL.from_pretrained(
        args.model_name_or_path,
        taskcla=taskcla,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        args=args
    )

    if (args.ft_task == 0 or 'one' in args.baseline or 'mtl' in args.baseline or 'comb' in args.baseline) and (not args.eval_only):
        # model.add_adapter('adapter',{'mh_adapter':True, 'output_adapter':True,'reduction_factor':args.reduction_factor, 'non_linearity':"relu"}) # no mh_adapter by default
        model.add_adapter('adapter') # no mh_adapter by default
    elif  args.eval_only:
        model.load_adapter(args.output_dir)
    else:
        model.load_adapter(args.prev_output)

    model.train_adapter('adapter') # note this train_adapter will affect even the parent node
    # train adapter reopen the adapter


    if 'supsup' in args.baseline:
        for n, m in model.named_modules():  # other part, including classifier should be trainable
            if isinstance(m, MultitaskMaskLinear):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    if 'mlm' in args.baseline:
        mlm_model.roberta = model.roberta

    if 'adapter_cat' in args.baseline:
        reference_model = MODEL.from_pretrained(
            args.model_name_or_path,
            taskcla=taskcla,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            args=args
        )
        # this teacher is trainable and randomlyh iniialized
        if args.ft_task == 0  and not args.eval_only:
            reference_model.add_adapter('reference_adapter')
        else:
            reference_model.delete_adapter('reference_adapter')  # reset
            reference_model.add_adapter('reference_adapter')

        reference_model.reinitialize_readout()
        model.reinitialize_readout()

        model = MyModel(model, teacher=reference_model, args=args)

    else:
        model = MyModel(model, mlm_model=mlm_model, args=args)

    return model



def lookfor_model_finetune(taskcla,args, config):

    if args.model_name_or_path:  # only do electra
        if 'adapter' in args.baseline:
            model = _lookfor_model_adapter(taskcla,args, config)
            return model

        elif 'ewc' in args.baseline:
            model = _lookfor_model_ewc(taskcla,args, config)
            return model

        elif 'prompt' in args.baseline or 'l2p' in args.baseline:
            model = _lookfor_model_prompt(taskcla,args, config)
            return model

        else:
            model = _lookfor_model_others(taskcla,args, config)
            return model

    else:
        raise ValueError('You must provide the model name or path.')




def get_view_for(n, p, masks, config, args):
    if 'bart' in args.model_name_or_path:
        from utils.bart import get_view_for as get_view_for
    elif 'roberta' in args.model_name_or_path:
        from utils.roberta import get_view_for as get_view_for
    elif 'bert' in args.model_name_or_path:
        from utils.bert import get_view_for as get_view_for

    return get_view_for(n, p, masks, config, args)


def mask(model, accelerator,args):
    if 'bart' in args.model_name_or_path:
        from utils.bart import mask as mask
    elif 'roberta' in args.model_name_or_path:
        from utils.roberta import mask as mask
    elif 'bert' in args.model_name_or_path:
        from utils.bert import mask as mask

    return mask(model, accelerator,args)


def get_view_for_tsv(n, model_ori, args):
    if 'bart' in args.model_name_or_path:
        from utils.bart import get_view_for_tsv as get_view_for_tsv
    elif 'roberta' in args.model_name_or_path:
        from utils.roberta import get_view_for_tsv as get_view_for_tsv
    elif 'bert' in args.model_name_or_path:
        from utils.bert import get_view_for_tsv as get_view_for_tsv

    return get_view_for_tsv(n, model_ori, args)


def lookfor_baseline_variable(self,args):
    if 'bart' in args.model_name_or_path:
        from utils.bart import lookfor_baseline_variable as lookfor_baseline_variable
    elif 'roberta' in args.model_name_or_path:
        from utils.roberta import lookfor_baseline_variable as lookfor_baseline_variable
    elif 'bert' in args.model_name_or_path:
        from utils.bert import lookfor_baseline_variable as lookfor_baseline_variable

    return lookfor_baseline_variable(self,args)



def lookfor_main_metric(results,args):


    if args.task_name in args.asc_datasets:
        eval_main = results['macro_f1']

    elif args.task_name in args.ccd_datasets:
        eval_main = results['macro_f1']

    elif args.task_name in args.ner_datasets:
        eval_main = results['f1']

    elif args.task_name in args.dialogue_datasets:
        eval_main = results['bleu']

    elif args.task_name in args.summerization_datasets:
        eval_main = results['rouge1']  # NER and generation

    return eval_main

def write_result(results,eval_t,args):
    # logger.info( "{} On {}, last epoch rougeLsum = {:.4f}, (seed={})".format(args.model_name_or_path, args.task_name, rougeLsum,args.seed))


    progressive_main_path = os.path.join(args.output_dir + '/../', 'progressive_main_' + str(args.seed))
    if os.path.exists(progressive_main_path):
        eval_main = np.loadtxt(progressive_main_path)
    else:
        eval_main = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)

    eval_main[args.ft_task][eval_t] = lookfor_main_metric(results,args)

    np.savetxt(progressive_main_path, eval_main, '%.4f', delimiter='\t')

    if args.task_name in args.asc_datasets or args.task_name in args.ccd_datasets:
        progressive_micro_f1_path = os.path.join(args.output_dir + '/../',
                                                 'progressive_micro_f1_' + str(args.seed))
        progressive_macro_f1_path = os.path.join(args.output_dir + '/../',
                                                 'progressive_macro_f1_' + str(args.seed))
        progressive_accuracy_path = os.path.join(args.output_dir + '/../',
                                                 'progressive_accuracy_' + str(args.seed))
        progressive_loss_path = os.path.join(args.output_dir + '/../', 'progressive_loss_' + str(args.seed))

        if os.path.exists(progressive_micro_f1_path):
            micro_f1 = np.loadtxt(progressive_micro_f1_path)
            macro_f1 = np.loadtxt(progressive_macro_f1_path)
            accuracy = np.loadtxt(progressive_accuracy_path)
            loss = np.loadtxt(progressive_loss_path)

        else:
            micro_f1 = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
            macro_f1 = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
            accuracy = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
            loss = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)


        micro_f1[args.ft_task][eval_t] = results['micro_f1']
        macro_f1[args.ft_task][eval_t] = results['macro_f1']
        accuracy[args.ft_task][eval_t] = results['accuracy']
        loss[args.ft_task][eval_t] = results['loss']

        np.savetxt(progressive_micro_f1_path, micro_f1, '%.4f', delimiter='\t')
        np.savetxt(progressive_macro_f1_path, macro_f1, '%.4f', delimiter='\t')
        np.savetxt(progressive_accuracy_path, accuracy, '%.4f', delimiter='\t')
        np.savetxt(progressive_loss_path, loss, '%.4f', delimiter='\t')


    elif args.task_name in args.ner_datasets:
        progressive_f1_path = os.path.join(args.output_dir + '/../', 'progressive_f1_' + str(args.seed))
        progressive_precision_path = os.path.join(args.output_dir + '/../',
                                                  'progressive_precision_' + str(args.seed))
        progressive_recall_path = os.path.join(args.output_dir + '/../',
                                               'progressive_recall_' + str(args.seed))
        progressive_accuracy_path = os.path.join(args.output_dir + '/../',
                                                 'progressive_accuracy_' + str(args.seed))

        if os.path.exists(progressive_f1_path):
            f1 = np.loadtxt(progressive_f1_path)
            precision = np.loadtxt(progressive_precision_path)
            recall = np.loadtxt(progressive_recall_path)
            accuracy = np.loadtxt(progressive_accuracy_path)

        else:
            f1 = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
            precision = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
            recall = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
            accuracy = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)

        f1[args.ft_task][eval_t] = results['f1']
        precision[args.ft_task][eval_t] = results['precision']
        recall[args.ft_task][eval_t] = results['recall']
        accuracy[args.ft_task][eval_t] = results['accuracy']

        np.savetxt(progressive_f1_path, f1, '%.4f', delimiter='\t')
        np.savetxt(progressive_precision_path, precision, '%.4f', delimiter='\t')
        np.savetxt(progressive_recall_path, recall, '%.4f', delimiter='\t')
        np.savetxt(progressive_accuracy_path, accuracy, '%.4f', delimiter='\t')


    elif args.task_name in args.dialogue_datasets:
        progressive_bleu_path = os.path.join(args.output_dir + '/../', 'progressive_bleu_' + str(args.seed))

        if os.path.exists(progressive_bleu_path):
            bleu = np.loadtxt(progressive_bleu_path)
        else:
            bleu = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)

        bleu[args.ft_task][eval_t] = results['bleu']

        np.savetxt(progressive_bleu_path, bleu, '%.4f', delimiter='\t')

    else:
        progressive_rouge1_path = os.path.join(args.output_dir + '/../',
                                               'progressive_rouge1_' + str(args.seed))
        progressive_rouge2_path = os.path.join(args.output_dir + '/../',
                                               'progressive_rouge2_' + str(args.seed))
        progressive_rougeL_path = os.path.join(args.output_dir + '/../',
                                               'progressive_rougeL_' + str(args.seed))
        progressive_rougeLsum_path = os.path.join(args.output_dir + '/../',
                                                  'progressive_rougeLsum_' + str(args.seed))

        if os.path.exists(progressive_rouge1_path):
            rouge1 = np.loadtxt(progressive_rouge1_path)
            rouge2 = np.loadtxt(progressive_rouge2_path)
            rougeL = np.loadtxt(progressive_rougeL_path)
            rougeLsum = np.loadtxt(progressive_rougeLsum_path)

        else:
            rouge1 = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
            rouge2 = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
            rougeL = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
            rougeLsum = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)

        rouge1[args.ft_task][eval_t] = results['rouge1']
        rouge2[args.ft_task][eval_t] = results['rouge2']
        rougeL[args.ft_task][eval_t] = results['rougeL']
        rougeLsum[args.ft_task][eval_t] = results['rougeLsum']

        np.savetxt(progressive_rouge1_path, rouge1, '%.4f', delimiter='\t')
        np.savetxt(progressive_rouge2_path, rouge2, '%.4f', delimiter='\t')
        np.savetxt(progressive_rougeL_path, rougeL, '%.4f', delimiter='\t')
        np.savetxt(progressive_rougeLsum_path, rougeLsum, '%.4f', delimiter='\t')

    if args.ft_task == args.ntasks - 1:  # last ft task, we need a final one

        final_main = os.path.join(args.output_dir, 'final_main_' + str(args.seed))
        forward_main = os.path.join(args.output_dir, 'forward_main_' + str(args.seed))

        if  'one' in args.baseline:
            with open(final_main, 'w') as final_main_file:
                for j in range(eval_main.shape[1]):
                    final_main_file.writelines(str(eval_main[j][j]) + '\n')
        else:
            with open(final_main, 'w') as final_main_file,open(forward_main, 'w') as forward_main_file:
                for j in range(eval_main.shape[1]):
                    final_main_file.writelines(str(eval_main[-1][j]) + '\n')
                    forward_main_file.writelines(str(eval_main[j][j]) + '\n')
                    
                    
        if args.task_name in args.asc_datasets or args.task_name in args.ccd_datasets:
            final_micro_f1 = os.path.join(args.output_dir, 'micro_f1_' + str(args.seed))
            forward_micro_f1 = os.path.join(args.output_dir, 'forward_micro_f1_' + str(args.seed))

            final_macro_f1 = os.path.join(args.output_dir, 'macro_f1_' + str(args.seed))
            forward_macro_f1 = os.path.join(args.output_dir, 'forward_macro_f1_' + str(args.seed))

            final_accuracy = os.path.join(args.output_dir, 'accuracy_' + str(args.seed))
            forward_accuracy = os.path.join(args.output_dir, 'forward_accuracy_' + str(args.seed))

            final_loss = os.path.join(args.output_dir, 'loss_' + str(args.seed))
            forward_loss = os.path.join(args.output_dir, 'forward_loss_' + str(args.seed))

            if  'one' in args.baseline:
                with open(final_micro_f1, 'w') as micro_f1_file, open(final_macro_f1, 'w') as macro_f1_file, open(final_accuracy,
                                                                                                          'w') as accuracy_file, open(
                        final_loss, 'w') as loss_file:
                    for j in range(micro_f1.shape[1]):
                        micro_f1_file.writelines(str(micro_f1[j][j]) + '\n')
                        macro_f1_file.writelines(str(macro_f1[j][j]) + '\n')
                        accuracy_file.writelines(str(accuracy[j][j]) + '\n')
                        loss_file.writelines(str(loss[j][j]) + '\n')

            else:
                with open(final_micro_f1, 'w') as final_micro_f1_file, open(final_macro_f1, 'w') as final_macro_f1_file, open(
                        final_accuracy, 'w') as final_accuracy_file, open(final_loss, 'w') as final_loss_file, \
                        open(forward_micro_f1, 'w') as forward_micro_f1_file, open(forward_macro_f1, 'w') as forward_macro_f1_file, open(
                    forward_accuracy, 'w') as forward_accuracy_file, open(forward_loss, 'w') as forward_loss_file:

                    for j in range(micro_f1.shape[1]):
                        final_micro_f1_file.writelines(str(micro_f1[-1][j]) + '\n')
                        final_macro_f1_file.writelines(str(macro_f1[-1][j]) + '\n')
                        final_accuracy_file.writelines(str(accuracy[-1][j]) + '\n')
                        final_loss_file.writelines(str(loss[-1][j]) + '\n')

                        forward_micro_f1_file.writelines(str(micro_f1[j][j]) + '\n')
                        forward_macro_f1_file.writelines(str(macro_f1[j][j]) + '\n')
                        forward_accuracy_file.writelines(str(accuracy[j][j]) + '\n')
                        forward_loss_file.writelines(str(loss[j][j]) + '\n')



        elif args.task_name in args.ner_datasets:
            final_f1 = os.path.join(args.output_dir, 'f1_' + str(args.seed))
            forward_f1 = os.path.join(args.output_dir, 'forward_f1_' + str(args.seed))

            final_precision = os.path.join(args.output_dir, 'precision_' + str(args.seed))
            forward_precision = os.path.join(args.output_dir, 'forward_precision_' + str(args.seed))

            final_recall = os.path.join(args.output_dir, 'recall_' + str(args.seed))
            forward_recall = os.path.join(args.output_dir, 'forward_recall_' + str(args.seed))

            final_accuracy = os.path.join(args.output_dir, 'accuracy_' + str(args.seed))
            forward_accuracy= os.path.join(args.output_dir, 'forward_accuracy_' + str(args.seed))

            if  'one' in args.baseline:
                with open(final_f1, 'w') as f1_file, open(final_precision, 'w') as precision_file, open(final_recall,'w') as recall_file, open(final_accuracy, 'w') as accuracy_file:
                    for j in range(f1.shape[1]):
                        f1_file.writelines(str(f1[j][j]) + '\n')
                        precision_file.writelines(str(precision[j][j]) + '\n')
                        recall_file.writelines(str(recall[j][j]) + '\n')
                        accuracy_file.writelines(str(accuracy[j][j]) + '\n')

            else:
                with open(final_f1, 'w') as final_f1_file, open(final_precision, 'w') as final_precision_file, open(
                        final_recall, 'w') as final_recall_file, open(final_accuracy, 'w') as final_accuracy_file, \
                        open(forward_f1, 'w') as forward_f1_file, open(forward_precision,
                                                                               'w') as forward_precision_file, open(
                    forward_recall, 'w') as forward_recall_file, open(forward_accuracy, 'w') as forward_accuracy_file:
                    for j in range(f1.shape[1]):
                        final_f1_file.writelines(str(f1[-1][j]) + '\n')
                        final_precision_file.writelines(str(precision[-1][j]) + '\n')
                        final_recall_file.writelines(str(recall[-1][j]) + '\n')
                        final_accuracy_file.writelines(str(accuracy[-1][j]) + '\n')

                        forward_f1_file.writelines(str(f1[j][j]) + '\n')
                        forward_precision_file.writelines(str(precision[j][j]) + '\n')
                        forward_recall_file.writelines(str(recall[j][j]) + '\n')
                        forward_accuracy_file.writelines(str(accuracy[j][j]) + '\n')
                        

        elif args.task_name in args.dialogue_datasets:
            final_bleu = os.path.join(args.output_dir, 'bleu_' + str(args.seed))
            forward_bleu = os.path.join(args.output_dir, 'forward_bleu_' + str(args.seed))

            if  'one' in args.baseline:
                with open(final_bleu, 'w') as final_bleu_file:
                    for j in range(bleu.shape[1]):
                        final_bleu_file.writelines(str(bleu[j][j]) + '\n')
            else:
                with open(final_bleu, 'w') as final_bleu_file,open(forward_bleu, 'w') as forward_bleu_file:
                    for j in range(bleu.shape[1]):
                        final_bleu_file.writelines(str(bleu[-1][j]) + '\n')
                        forward_bleu_file.writelines(str(bleu[j][j]) + '\n')
                        
                    
        elif args.task_name in args.summerization_datasets:
            final_rouge1 = os.path.join(args.output_dir, 'rouge1_' + str(args.seed))
            forward_rouge1 = os.path.join(args.output_dir, 'forward_rouge1_' + str(args.seed))

            final_rouge2 = os.path.join(args.output_dir, 'rouge2_' + str(args.seed))
            forward_rouge2 = os.path.join(args.output_dir, 'forward_rouge2_' + str(args.seed))

            final_rougeL = os.path.join(args.output_dir, 'rougeL_' + str(args.seed))
            forward_rougeL = os.path.join(args.output_dir, 'forward_rougeL_' + str(args.seed))

            final_rougeLsum = os.path.join(args.output_dir, 'rougeLsum_' + str(args.seed))
            forward_rougeLsum = os.path.join(args.output_dir, 'forward_rougeLsum_' + str(args.seed))

            if  'one' in args.baseline:  
                with open(final_rouge1, 'w') as rouge1_file, open(final_rouge2, 'w') as rouge2_file, open(final_rougeL, 'w') as rougeL_file, open(final_rougeLsum, 'w') as rougeLsum_file:
                    for j in range(rouge1.shape[1]):
                        rouge1_file.writelines(str(rouge1[j][j]) + '\n')
                        rouge2_file.writelines(str(rouge2[j][j]) + '\n')
                        rougeL_file.writelines(str(rougeL[j][j]) + '\n')
                        rougeLsum_file.writelines(str(rougeLsum[j][j]) + '\n')
    
            else:
                with open(final_rouge1, 'w') as final_rouge1_file, open(final_rouge2, 'w') as final_rouge2_file, open(final_rougeL, 'w') as final_rougeL_file, open(final_rougeLsum, 'w') as final_rougeLsum_file,\
                    open(forward_rouge1, 'w') as forward_rouge1_file, open(forward_rouge2, 'w') as forward_rouge2_file, open(forward_rougeL, 'w') as forward_rougeL_file, open(forward_rougeLsum, 'w') as forward_rougeLsum_file:
                    for j in range(rouge1.shape[1]):
                        final_rouge1_file.writelines(str(rouge1[-1][j]) + '\n')
                        final_rouge2_file.writelines(str(rouge2[-1][j]) + '\n')
                        final_rougeL_file.writelines(str(rougeL[-1][j]) + '\n')
                        final_rougeLsum_file.writelines(str(rougeLsum[-1][j]) + '\n')

                        forward_rouge1_file.writelines(str(rouge1[j][j]) + '\n')
                        forward_rouge2_file.writelines(str(rouge2[j][j]) + '\n')
                        forward_rougeL_file.writelines(str(rougeL[j][j]) + '\n')
                        forward_rougeLsum_file.writelines(str(rougeLsum[j][j]) + '\n')
                        
                        

def gather_imp(head_imp):
    head_imp_list = [torch.zeros_like(head_imp) for _ in range(dist.get_world_size())]
    # Allgather
    dist.all_gather(tensor_list=head_imp_list, tensor=head_imp.contiguous())

    # Since allgather results do not have gradients, we replace the
    # current process's corresponding embeddings with original tensors
    head_imp_list[dist.get_rank()] = head_imp
    # Get full batch embeddings: (bs x N, hidden)
    head_imp = torch.cat(head_imp_list, 0)

    return head_imp


def gather_mean(head_imp):
    head_importance_list = [torch.zeros_like(head_imp) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=head_importance_list, tensor=head_imp.contiguous()) # everyone need to do this
    head_importance_list = torch.stack(head_importance_list)
    head_importance = torch.mean(head_importance_list,dim=0)
    return head_importance

def prompt_eval(self,model,dataloader,metric,eval_t, pred_file, target_file,accelerator):
    tune_model = accelerator.unwrap_model(model).model
    infer_model = accelerator.unwrap_model(model).teacher

    results = self.eval(model=model, dataloader=dataloader, metric=metric, accelerator=accelerator, eval_t=eval_t,
                        tune_model=tune_model, infer_model=infer_model, pred_file=pred_file, target_file=target_file)

    return results



def frequency_norm(frequency,eps=5e-5):
    frequency = (frequency - frequency.mean()) / (frequency.std()+eps)  # 2D, we need to deal with this for each layer
    return frequency


def sim_matrix(a, b, eps=1e-8):
    """Batch version of CosineSimilarity."""
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,attn_size):
        super(Self_Attn,self).__init__()

        self.query_conv = nn.Linear(attn_size,attn_size)
        self.key_conv = nn.Linear(attn_size , attn_size)
        self.value_conv = nn.Linear(attn_size ,attn_size)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B,max_length,hidden_size)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,width,height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,width,height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        # print('energy: ',energy.size())

        attention = self.softmax(energy) # BX (N) X (N)

        # attention =  F.gumbel_softmax(energy,hard=True,dim=-1)
        # print('attention: ',attention)
        proj_value = self.value_conv(x).view(m_batchsize,width,height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,width,height)

        out = self.gamma*out + x


        return out



class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, n_head, d_model, d_inner, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.position_enc = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, enc_q=None,ranking=None):
        #TODO: Positional/ranking embedding

        if enc_q is None:
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
            enc_output = self.pos_ffn(enc_output)

        else:
            enc_output, enc_slf_attn = self.slf_attn(enc_q, enc_input, enc_input)
            enc_output = self.pos_ffn(enc_output)

        enc_output = self.layer_norm(enc_output)

        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5) #sqrt d_k

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v):


        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = torch.squeeze(q,1)
        q = self.layer_norm(q)


        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


        q, attn = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)

        if len_q == 1:
            q = q.transpose(1, 2).contiguous().view(sz_b,-1)
        else:
            q = q.transpose(1, 2).contiguous().view(sz_b, len_q,-1)

        q = self.dropout(self.fc(q))
        q += residual

        return q, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=40):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, enc_input,ranking):
        return enc_input + self.pos_table[:, ranking].clone().detach()



def deep_copy(model,accelerator,args):

    unwrap_model = accelerator.unwrap_model(model)
    unwrap_adaptive_model = deepcopy(unwrap_model)
    optimizer_grouped_parameters = utils.optimize.lookfor_optimize(unwrap_adaptive_model,args)  # everything is based on adative_model
    adaptive_optimizer = AdamW(optimizer_grouped_parameters)
    adaptive_model,adaptive_optimizer = accelerator.prepare(unwrap_adaptive_model,adaptive_optimizer)

    return adaptive_model,adaptive_optimizer
