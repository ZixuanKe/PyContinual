import sys,os,argparse,time
import numpy as np
import torch
from config import set_args
import utils
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import RandomSampler,SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, ConcatDataset
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Arguments
args = set_args()

# Currently, 2 tasks included in TIL-Extraction Category

if args.task == 'ner': #name entity recognition
    if args.experiment=='w2v':
        from dataloaders.ner import w2v as dataloader
    elif args.experiment=='w2v_as':
        from dataloaders.ner import w2v_as as dataloader
    elif  args.experiment=='bert_gen_hat':
        from dataloaders.ner import bert_gen_hat as dataloader
    elif  args.experiment=='bert_gen' or args.experiment=='bert_gen_single':
        from dataloaders.ner import bert_gen as dataloader
    elif args.experiment=='bert_sep':
        from dataloaders.ner import bert_sep as dataloader
    elif 'bert' in args.experiment: #all others
        from dataloaders.ner import bert as dataloader

elif args.task == 'ae': #name entity recognition
    if args.experiment=='w2v':
        from dataloaders.ae import w2v as dataloader
    elif args.experiment=='w2v_as':
        from dataloaders.ae import w2v_as as dataloader
    elif  args.experiment=='bert_gen_hat':
        from dataloaders.ae import bert_gen_hat as dataloader
    elif  args.experiment=='bert_gen' or args.experiment=='bert_gen_single':
        from dataloaders.ae import bert_gen as dataloader
    elif args.experiment=='bert_sep':
        from dataloaders.ae import bert_sep as dataloader
    elif 'bert' in args.experiment: #all others
        from dataloaders.ae import bert as dataloader


# Args -- Approach BERT
if args.approach == 'w2v_kim_owm_ncl':
    from approaches.til_extraction import w2v_cnn_owm_ncl as approach
elif args.approach == 'w2v_kim_ucl_ncl':
    from approaches.til_extraction import w2v_cnn_ucl_ncl as approach
elif args.approach=='w2v_kim_single_ncl' or args.approach=='w2v_kim_double_ncl' or args.approach=='w2v_kim_ncl'\
        or args.approach=='w2v_kim_gcae_ncl' or args.approach=='w2v_kim_pcnn_ncl' or args.approach=='w2v_mlp_sentence_ncl'\
        or args.approach=='w2v_kim_ana_ncl' or args.approach=='w2v_lstm_ncl':
    from approaches.til_extraction import w2v_ncl as approach
elif args.approach=='w2v_kim_single_one' or args.approach=='w2v_kim_double_one' or args.approach=='w2v_kim_one'\
        or args.approach=='w2v_kim_gcae_one' or args.approach=='w2v_kim_pcnn_one' or args.approach=='w2v_mlp_sentence_one'\
        or args.approach=='w2v_kim_ana_one':
    from approaches.til_extraction import w2v_one as approach
elif args.approach=='w2v_kim_hat_ncl':
    from approaches.til_extraction import w2v_cnn_hat_ncl as approach
elif args.approach=='w2v_kim_ewc_ncl' or args.approach=='w2v_gru_ewc_ncl':
    from approaches.til_extraction import w2v_cnn_ewc_ncl as approach
elif args.approach=='w2v_gru_kan_ncl' :
    from approaches.til_extraction import w2v_rnn_kan_ncl as approach
elif args.approach=='w2v_gru_srk_ncl':
    from approaches.til_extraction import w2v_rnn_srk_ncl as approach

# Args -- Approach W2V
if args.approach == 'bert_kim_ucl_ncl':
    from approaches.til_extraction import bert_cnn_ucl_ncl as approach
elif args.approach == 'bert_adapter_ucl_ncl':
    from approaches.til_extraction import bert_adapter_ucl_ncl as approach
elif args.approach == 'bert_kim_owm_ncl':
    from approaches.til_extraction import bert_cnn_owm_ncl as approach
elif args.approach=='bert_one':
    from approaches.til_extraction import bert_one as approach
elif args.approach=='bert_adapter_one':
    from approaches.til_extraction import bert_adapter_one as approach
elif args.approach=='bert_ncl':
    from approaches.til_extraction import bert_ncl as approach
elif args.approach=='bert_adapter_ncl':
    from approaches.til_extraction import bert_adapter_ncl as approach
elif args.approach=='bert_gen_hat_ncl':
    from approaches.til_extraction import bert_gen_hat_ncl as approach
elif args.approach=='bert_gen_ncl':
    from approaches.til_extraction import bert_gen_ncl as approach
elif args.approach=='bert_gen_single_ncl':
    from approaches.til_extraction import bert_gen_single_ncl as approach
elif args.approach=='bert_adapter_mask_ent_ncl':
    from approaches.til_extraction import bert_adapter_mask_ent_ncl as approach
elif args.approach=='bert_adapter_mask_ncl':
    from approaches.til_extraction import bert_adapter_mask_ncl as approach
elif args.approach=='bert_adapter_attention_mask_ncl':
    from approaches.til_extraction import bert_adapter_attention_mask_ncl as approach
elif args.approach=='bert_adapter_two_modules_mask_ncl':
    from approaches.til_extraction import bert_adapter_two_modules_mask_ncl as approach
elif args.approach=='bert_adapter_capsule_mask_cnn_way_ncl':
    from approaches.til_extraction import bert_adapter_capsule_mask_cnn_way_ncl as approach
elif args.approach=='bert_adapter_capsule_mask_ncl' or args.approach=='bert_adapter_capsule_ncl':
    from approaches.til_extraction import bert_adapter_capsule_mask_ncl as approach
elif args.approach=='bert_adapter_capsule_mask_imp_ncl':
    from approaches.til_extraction import bert_adapter_capsule_mask_imp_ncl as approach
elif args.approach=='bert_adapter_mlp_mask_ncl':
    from approaches.til_extraction import bert_adapter_mlp_mask_ncl as approach
elif args.approach=='bert_adapter_capsule_mask_fp16_ncl':
    from approaches.til_extraction import bert_adapter_capsule_mask_fp16_ncl as approach
elif args.approach=='bert_mtl':
    from approaches.til_extraction import bert_mtl as approach
elif args.approach=='bert_sep_sentence_ncl':
    from approaches.til_extraction import bert_sep_ncl as approach
elif args.approach=='bert_sep_sentence_one':
    from approaches.til_extraction import bert_sep_one as approach
elif args.approach=='bert_sep_sentence_mtl':
    from approaches.til_extraction import bert_sep_mtl as approach
elif args.approach=='bert_mlp_one' or args.approach=='bert_kim_one':
    from approaches.til_extraction import bert_cnn_one as approach
elif args.approach=='bert_mlp_ncl' or args.approach=='bert_kim_ncl' or args.approach=='bert_lstm_ncl':
    from approaches.til_extraction import bert_cnn_ncl as approach
elif args.approach=='bert_gru_srk_ncl':
    from approaches.til_extraction import bert_rnn_srk_ncl as approach
elif args.approach=='bert_gru_kan_ncl':
    from approaches.til_extraction import bert_rnn_kan_ncl as approach
elif args.approach=='bert_mlp_hat_ncl' or args.approach=='bert_kim_hat_ncl' or args.approach=='bert_gru_hat_ncl':
    from approaches.til_extraction import bert_cnn_hat_ncl as approach
elif args.approach=='bert_kim_ewc_ncl' or args.approach=='bert_gru_ewc_ncl' or args.approach=='bert_lstm_ewc_ncl':
    from approaches.til_extraction import bert_cnn_ewc_ncl as approach
elif args.approach=='bert_adapter_ewc_ncl':
    from approaches.til_extraction import bert_adapter_ewc_ncl as approach
elif args.approach=='bert_mlp_mtl' or args.approach=='bert_kim_mtl':
    from approaches.til_extraction import bert_cnn_mtl as approach
elif args.approach=='bert_sep_mlp_sentence_mtl' or args.approach=='bert_sep_kim_sentence_mtl':
    from approaches.til_extraction import bert_sep_cnn_mtl as approach
elif args.approach=='bert_sep_pcnn_one' or args.approach=='bert_sep_gcae_one' or args.approach=='bert_sep_kim_sentence_one'\
        or args.approach=='bert_sep_mlp_sentence_one':
    from approaches.til_extraction import bert_sep_cnn_one as approach
elif args.approach=='bert_sep_pcnn_ncl' or args.approach=='bert_sep_gcae_ncl' or args.approach=='bert_sep_kim_sentence_ncl'\
        or args.approach=='bert_sep_mlp_sentence_ncl':
    from approaches.til_extraction import bert_sep_cnn_ncl as approach

# # Args -- Network
if args.approach == 'bert_kim_ucl_ncl':
    from networks.til_extraction import bert_kim_ucl as network
elif args.approach == 'bert_adapter_ucl_ncl':
    from networks.til_extraction import bert_adapter_ucl as network
elif args.approach == 'bert_kim_owm_ncl':
    from networks.til_extraction import bert_kim_owm as network
elif 'bert_sep_pcnn' in args.approach:
    from networks.til_extraction import bert_sep_pcnn as network
elif 'bert_sep_kim_sentence' in args.approach:
    from networks.til_extraction import bert_sep_kim_sentence as network
elif 'bert_sep_mlp_sentence' in args.approach:
    from networks.til_extraction import bert_sep_mlp_sentence as network
elif 'bert_sep_gcae' in args.approach:
    from networks.til_extraction import bert_sep_gcae as network
elif 'bert_sep_sentence' in args.approach:
    from networks.til_extraction import bert_sep_sentence as network
elif 'bert_gru_srk' in args.approach:
    from networks.til_extraction import bert_gru_srk as network
elif 'bert_kim_hat' in args.approach:
    from networks.til_extraction import bert_kim_hat as network
elif 'bert_gru_hat' in args.approach:
    from networks.til_extraction import bert_gru_hat as network
elif 'bert_mlp_hat' in args.approach:
    from networks.til_extraction import bert_mlp_hat as network
elif 'bert_mlp' in args.approach:
    from networks.til_extraction import bert_mlp as network
elif 'bert_kim' in args.approach or 'bert_kim_ewc' in args.approach:
    from networks.til_extraction import bert_kim as network
elif 'bert_gen_hat' in args.approach:
    from networks.til_extraction import bert_gen_hat as network
elif 'bert_gen_single' in args.approach:
    from networks.til_extraction import bert_gen_single as network
elif 'bert_gen' in args.approach:
    from networks.til_extraction import bert_gen as network
elif 'bert_adapter_mask' in args.approach:
    from networks.til_extraction import bert_adapter_mask as network
elif 'bert_adapter_attention_mask' in args.approach:
    from networks.til_extraction import bert_adapter_attention_mask as network
elif 'bert_adapter_two_modules_mask' in args.approach:
    from networks.til_extraction import bert_adapter_two_modules_mask as network
elif 'bert_adapter_capsule_mask_imp' in args.approach:
    from networks.til_extraction import bert_adapter_capsule_mask_imp as network
elif 'bert_adapter_capsule_mask' in args.approach:
    from networks.til_extraction import bert_adapter_capsule_mask as network
elif 'bert_adapter_capsule' in args.approach:
    from networks.til_extraction import bert_adapter_capsule as network
elif 'bert_adapter_mlp_mask' in args.approach:
    from networks.til_extraction import bert_adapter_mlp_mask as network
elif 'bert_adapter' in args.approach:
    from networks.til_extraction import bert_adapter as network
elif 'bert_lstm' in args.approach:
    from networks.til_extraction import bert_lstm as network
elif 'bert_gru_kan' in args.approach:
    from networks.til_extraction import bert_gru_kan as network
elif 'bert_gru' in args.approach:
    from networks.til_extraction import bert_gru as network
elif 'bert' in args.approach:
    from networks.til_extraction import bert as network

# Network
if args.approach == 'w2v_kim_owm_ncl':
    from networks.til_extraction import w2v_kim_owm as network
elif args.approach == 'w2v_kim_ucl_ncl':
    from networks.til_extraction import w2v_kim_ucl as network
elif 'w2v_kim_hat' in args.approach:
    from networks.til_extraction import w2v_kim_hat as network
elif 'w2v_kim_single' in args.approach:
    from networks.til_extraction import w2v_single as network
elif 'w2v_kim_double' in args.approach:
    from networks.til_extraction import w2v_double as network
elif 'w2v_kim_gcae' in args.approach:
    from networks.til_extraction import w2v_gcae as network
elif 'w2v_kim_pcnn' in args.approach:
    from networks.til_extraction import w2v_pcnn as network
elif 'w2v_kim_ana' in args.approach:
    from networks.til_extraction import w2v_ana as network
elif 'w2v_mlp_sentence' in args.approach:
    from networks.til_extraction import w2v_mlp_sentence as network
elif 'w2v_kim_ewc' in args.approach:
    from networks.til_extraction import w2v_kim as network
elif 'w2v_gru_ewc' in args.approach:
    from networks.til_extraction import w2v_gru as network
elif 'w2v_gru_srk' in args.approach :
    from networks.til_extraction import w2v_gru_srk as network
elif 'w2v_gru_kan' in args.approach:
    from networks.til_extraction import w2v_gru_kan as network
elif 'w2v_lstm' in args.approach:
    from networks.til_extraction import w2v_lstm as network
elif 'w2v_kim' in args.approach:
    from networks.til_extraction import w2v_kim as network

