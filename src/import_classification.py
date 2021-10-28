from config import set_args

# Arguments
args = set_args()

# ----------------------------------------------------------------------
# Language Datasets.
# ----------------------------------------------------------------------
if args.task == 'asc': #aspect sentiment classication
    if args.experiment=='w2v':
        from dataloaders.asc import w2v as dataloader
    elif args.experiment=='w2v_as':
        from dataloaders.asc import w2v_as as dataloader
    elif 'bert' in args.experiment: #all others
        from dataloaders.asc import bert as dataloader

elif args.task == 'dsc': #document sentiment classication
    if args.experiment=='w2v':
        from dataloaders.dsc import w2v as dataloader
    elif 'bert' in args.experiment: #all others
        from dataloaders.dsc import bert as dataloader

elif args.task == 'ssc': #sentence sentiment classication
    if args.experiment=='w2v':
        from dataloaders.ssc import w2v as dataloader
    elif 'bert' in args.experiment: #all others
        from dataloaders.ssc import bert as dataloader

elif args.task == 'nli': #aspect sentiment classication
    if args.experiment=='w2v':
        from dataloaders.nli import w2v as dataloader
    elif 'bert' in args.experiment: #all others
        from dataloaders.nli import bert as dataloader

elif args.task == 'newsgroup': #aspect sentiment classication
    if args.experiment=='w2v':
        from dataloaders.newsgroup import w2v as dataloader
    elif 'bert' in args.experiment: #all others
        from dataloaders.newsgroup import bert as dataloader

# ----------------------------------------------------------------------
# Image Datasets.
# ----------------------------------------------------------------------

elif args.task == 'celeba':
    from dataloaders.celeba import celeba as dataloader
elif args.task == 'femnist':
    from dataloaders.femnist import femnist as dataloader
elif args.task == 'vlcs':
    from dataloaders.vlcs import vlcs as dataloader
elif args.task == 'cifar10' and 'dil' in args.scenario:
    from dataloaders.cifar10 import dil_cifar10 as dataloader
elif args.task == 'mnist' and 'dil' in args.scenario:
    from dataloaders.mnist import dil_mnist as dataloader
elif args.task == 'fashionmnist' and 'dil' in args.scenario:
    from dataloaders.fashionmnist import dil_fashionmnist as dataloader
elif args.task == 'cifar100' and 'dil' in args.scenario:
    from dataloaders.cifar100 import dil_cifar100 as dataloader



# ----------------------------------------------------------------------
# Image Approaches.
# ----------------------------------------------------------------------

if args.approach == 'cnn_ncl' or args.approach == 'mlp_ncl':
    from approaches.classification import cnn_ncl as approach
elif args.approach == 'cnn_hat_ncl' or args.approach == 'mlp_hat_ncl':
    from approaches.classification import cnn_hat_ncl as approach
elif args.approach == 'cnn_aem_ncl' or args.approach == 'mlp_aem_ncl':
    from approaches.classification import cnn_aem_ncl as approach
elif args.approach=='cnn_one':
    from approaches.classification import cnn_one as approach
elif args.approach=='cnn_mtl':
    from approaches.classification import cnn_mtl as approach
elif args.approach=='cnn_hat_aux_ncl' or args.approach == 'mlp_hat_aux_ncl':
    from approaches.classification import cnn_hat_aux_ncl as approach
elif args.approach=='cnn_hat_merge_ncl' or args.approach=='mlp_hat_merge_ncl':
    from approaches.classification import cnn_hat_merge_ncl as approach
elif args.approach=='cnn_ewc_ncl' or args.approach=='mlp_ewc_ncl':
    from approaches.classification import cnn_ewc_ncl as approach
elif args.approach=='cnn_ucl_ncl' or args.approach=='mlp_ucl_ncl':
    from approaches.classification import cnn_ucl_ncl as approach
elif args.approach=='cnn_owm_ncl' or args.approach=='mlp_owm_ncl':
    from approaches.classification import cnn_owm_ncl as approach
elif args.approach=='cnn_derpp_ncl' or args.approach=='mlp_derpp_ncl':
    from approaches.classification import cnn_derpp_ncl as approach
elif args.approach=='cnn_acl_ncl' or args.approach=='mlp_acl_ncl':
    from approaches.classification import cnn_acl_ncl as approach
elif args.approach=='cnn_hal_ncl' or args.approach=='mlp_hal_ncl':
    from approaches.classification import cnn_hal_ncl as approach
elif args.approach=='cnn_cat_ncl' or args.approach=='mlp_cat_ncl':
    from approaches.classification import cnn_cat_ncl as approach


# ----------------------------------------------------------------------
# Lanaguage Approaches.
# ----------------------------------------------------------------------

if args.approach == 'w2v_kim_owm_ncl':
    from approaches.classification import w2v_cnn_owm_ncl as approach
elif args.approach == 'w2v_kim_ucl_ncl':
    from approaches.classification import w2v_cnn_ucl_ncl as approach
elif args.approach=='w2v_kim_single_ncl' or args.approach=='w2v_kim_double_ncl' or args.approach=='w2v_kim_ncl'\
        or args.approach=='w2v_kim_gcae_ncl' or args.approach=='w2v_kim_pcnn_ncl' or args.approach=='w2v_mlp_sentence_ncl'\
        or args.approach=='w2v_kim_ana_ncl' or args.approach=='w2v_lstm_ncl':
    from approaches.classification import w2v_ncl as approach
elif args.approach=='w2v_kim_single_one' or args.approach=='w2v_kim_double_one' or args.approach=='w2v_kim_one'\
        or args.approach=='w2v_kim_gcae_one' or args.approach=='w2v_kim_pcnn_one' or args.approach=='w2v_mlp_sentence_one'\
        or args.approach=='w2v_kim_ana_one':
    from approaches.classification import w2v_one as approach
elif args.approach=='w2v_kim_hat_ncl':
    from approaches.classification import w2v_cnn_hat_ncl as approach
elif args.approach=='w2v_kim_ewc_ncl':
    from approaches.classification import w2v_cnn_ewc_ncl as approach
elif args.approach=='w2v_gru_kan_ncl' :
    from approaches.classification import w2v_rnn_kan_ncl as approach
elif args.approach=='w2v_gru_srk_ncl':
    from approaches.classification import w2v_rnn_srk_ncl as approach
if args.approach == 'w2v_kim_derpp_ncl':
    from approaches.classification import w2v_cnn_derpp_ncl as approach
elif args.approach == 'w2v_kim_a-gem_ncl':
    from approaches.classification import w2v_cnn_agem_ncl as approach
elif args.approach == 'w2v_kim_l2_ncl':
    from approaches.classification import w2v_cnn_l2_ncl as approach

# Args -- Approach
if args.approach == 'bert_kim_ucl_ncl':
    from approaches.classification import bert_cnn_ucl_ncl as approach
if args.approach == 'bert_kim_derpp_ncl':
    from approaches.classification import bert_cnn_derpp_ncl as approach
elif args.approach == 'bert_kim_gem_ncl':
    from approaches.classification import bert_cnn_gem_ncl as approach
elif args.approach == 'bert_kim_a-gem_ncl':
    from approaches.classification import bert_cnn_agem_ncl as approac
elif args.approach == 'bert_kim_owm_ncl':
    from approaches.classification import bert_cnn_owm_ncl as approach
elif args.approach=='bert_one':
    from approaches.classification import bert_one as approach
if args.approach == 'bert_adapter_derpp_ncl':
    from approaches.classification import bert_adapter_derpp_ncl as approach
elif args.approach=='bert_adapter_one':
    from approaches.classification import bert_adapter_one as approach
elif args.approach=='bert_ncl':
    from approaches.classification import bert_ncl as approach
elif args.approach=='bert_adapter_mask_aux_ncl':
    from approaches.classification import bert_adapter_mask_aux_ncl as approach
elif args.approach=='bert_aux_trans_sup_ncl':
    from approaches.classification import bert_aux_trans_sup_ncl as approach
elif args.approach=='bert_aux_forget_sup_ncl':
    from approaches.classification import bert_aux_forget_sup_ncl as approach
elif args.approach=='bert_adapter_mask_sup_ncl_2stage':
    from approaches.classification import bert_adapter_mask_sup_ncl_2stage as approach
elif args.approach=='bert_adapter_mask_sup_ncl':
    from approaches.classification import bert_adapter_mask_sup_ncl as approach
elif args.approach=='bert_adapter_mask_trans_sup_ncl':
    from approaches.classification import bert_adapter_mask_trans_sup_ncl as approach
elif args.approach=='bert_adapter_mask_forget_sup_ncl':
    from approaches.classification import bert_adapter_mask_forget_sup_ncl as approach
elif args.approach=='bert_adapter_mask_trans_forget_sup_ncl':
    from approaches.classification import bert_adapter_mask_trans_forget_sup_ncl as approach
elif args.approach=='bert_sup_ncl':
    from approaches.classification import bert_sup_ncl as approach
elif args.approach=='bert_adapter_sup_ncl':
    from approaches.classification import bert_adapter_sup_ncl as approach
elif args.approach=='bert_adapter_grow_ncl':
    from approaches.classification import bert_adapter_grow_ncl as approach
elif args.approach=='bert_adapter_grow_trans_sup_ncl':
    from approaches.classification import bert_adapter_grow_trans_sup_ncl as approach
elif args.approach=='bert_adapter_grow_forget_sup_ncl':
    from approaches.classification import bert_adapter_grow_forget_sup_ncl as approach
elif args.approach=='bert_adapter_grow_trans_forget_sup_ncl':
    from approaches.classification import bert_adapter_grow_trans_forget_sup_ncl as approach
elif args.approach=='bert_adapter_ncl':
    from approaches.classification import bert_adapter_ncl as approach
elif args.approach == 'bert_adapter_ucl_ncl':
    from approaches.classification import bert_adapter_ucl_ncl as approach
elif args.approach=='bert_gen_hat_ncl':
    from approaches.classification import bert_gen_hat_ncl as approach
elif args.approach=='bert_gen_ncl':
    from approaches.classification import bert_gen_ncl as approach
elif args.approach=='bert_gen_single_ncl':
    from approaches.classification import bert_gen_single_ncl as approach
elif args.approach=='bert_capsule_mask_ncl':
    from approaches.classification import bert_capsule_mask_ncl as approach
elif args.approach=='bert_adapter_mask_ncl':
    from approaches.classification import bert_adapter_mask_ncl as approach
elif args.approach=='bert_adapter_mask_ent_ncl':
    from approaches.classification import bert_adapter_mask_ent_ncl as approach
elif args.approach=='bert_adapter_capsule_mask_ent_ncl':
    from approaches.classification import bert_adapter_capsule_mask_ent_ncl as approach
elif args.approach=='bert_adapter_ewc_ncl':
    from approaches.classification import bert_adapter_ewc_ncl as approach
elif args.approach == 'bert_adapter_a-gem_ncl':
    from approaches.classification import bert_adapter_agem_ncl as approach
elif args.approach == 'bert_adapter_gem_ncl':
    from approaches.classification import bert_adapter_gem_ncl as approach
elif args.approach=='bert_adapter_l2_ncl':
    from approaches.classification import bert_adapter_l2_ncl as approach
elif args.approach=='bert_adapter_owm_ncl':
    from approaches.classification import bert_adapter_owm_ncl as approach
elif args.approach=='bert_adapter_capsule_mask_cnn_way_ncl':
    from approaches.classification import bert_adapter_capsule_mask_cnn_way_ncl as approach
elif args.approach=='bert_adapter_capsule_mask_imp_ncl':
    from approaches.classification import bert_adapter_capsule_mask_ncl as approach
elif args.approach=='bert_adapter_capsule_mask_ncl' or args.approach=='bert_adapter_capsule_ncl':
    from approaches.classification import bert_adapter_capsule_mask_ncl as approach
elif args.approach=='bert_adapter_mlp_mask_ncl':
    from approaches.classification import bert_adapter_mlp_mask_ncl as approach
elif args.approach=='bert_mtl' or args.approach=='bert_adapter_mtl':
    from approaches.classification import bert_mtl as approach
elif args.approach=='bert_mlp_one' or args.approach=='bert_kim_one':
    from approaches.classification import bert_cnn_one as approach
elif args.approach=='bert_mlp_ncl' or args.approach=='bert_kim_ncl' or args.approach=='bert_lstm_ncl':
    from approaches.classification import bert_cnn_ncl as approach
elif args.approach=='bert_gru_srk_ncl':
    from approaches.classification import bert_rnn_srk_ncl as approach
elif args.approach=='bert_gru_kan_ncl':
    from approaches.classification import bert_rnn_kan_ncl as approach
elif args.approach=='bert_mlp_hat_ncl' or args.approach=='bert_kim_hat_ncl':
    from approaches.classification import bert_cnn_hat_ncl as approach
elif args.approach=='bert_kim_hat_ent_ncl':
    from approaches.classification import bert_cnn_hat_ent_ncl as approach
elif args.approach=='bert_kim_ewc_ncl':
    from approaches.classification import bert_cnn_ewc_ncl as approach
elif args.approach=='bert_mlp_mtl' or args.approach=='bert_kim_mtl':
    from approaches.classification import bert_cnn_mtl as approach
elif args.approach=='bert_sep_pcnn_ncl' or args.approach=='bert_sep_gcae_ncl' or args.approach=='bert_sep_kim_sentence_ncl'\
        or args.approach=='bert_sep_mlp_sentence_ncl':
    from approaches.classification import bert_sep_cnn_ncl as approach



# ----------------------------------------------------------------------
# Image Networks.
# ----------------------------------------------------------------------


# # Args -- Network
if 'cnn_hat_aux' in args.approach or 'cnn_hat_merge' in args.approach:
    from networks.classification import cnn as network
    from networks.classification import cnn_hat as aux_network
elif 'mlp_hat_aux' in args.approach or 'mlp_hat_merge' in args.approach:
    from networks.classification import mlp as network
    from networks.classification import mlp_hat as aux_network
elif 'cnn_aem' in args.approach:
    from networks.classification import cnn_aem as network
elif 'cnn_one' in args.approach:
    from networks.classification import cnn as network
elif 'cnn_hat' in args.approach:
    from networks.classification import cnn_hat as network
elif 'cnn_ucl' in args.approach:
    from networks.classification import cnn_ucl as network
elif 'cnn_owm' in args.approach:
    from networks.classification import cnn_owm as network
elif 'cnn_acl' in args.approach:
    from networks.classification import cnn_acl as network
elif 'cnn_cat' in args.approach:
    from networks.classification import cnn_cat as network
elif 'cnn' in args.approach:
    from networks.classification import cnn as network
elif 'mlp_ucl' in args.approach:
    from networks.classification import mlp_ucl as network
elif 'mlp_owm' in args.approach:
    from networks.classification import mlp_owm as network
elif 'mlp_hat' in args.approach:
    from networks.classification import mlp_hat as network
elif 'mlp_acl' in args.approach:
    from networks.classification import mlp_acl as network
elif 'mlp_cat' in args.approach:
    from networks.classification import mlp_cat as network
elif 'mlp' in args.approach:
    from networks.classification import mlp as network


# ----------------------------------------------------------------------
# Lanaguage Networks.
# ----------------------------------------------------------------------

if args.approach == 'bert_kim_ucl_ncl':
    from networks.classification import bert_kim_ucl as network
elif args.approach == 'bert_adapter_ucl_ncl':
    from networks.classification import bert_adapter_ucl as network
elif args.approach == 'bert_adapter_owm_ncl':
    from networks.classification import bert_adapter_owm as network
elif args.approach == 'bert_kim_owm_ncl':
    from networks.classification import bert_kim_owm as network
elif 'bert_aux_trans_sup' in args.approach or 'bert_aux_forget_sup' in args.approach \
        or 'bert_aux_trans_forget_sup' in args.approach:
    from networks.classification import bert as network
    from networks.classification import bert_adapter_mask as aux_network
elif 'bert_adapter_mask_aux' in args.approach:
    from networks.classification import bert_adapter as network
    # from networks.classification import bert_adapter_mask as network
    from networks.classification import bert_adapter_mask as aux_network

elif 'bert_sep_pcnn' in args.approach:
    from networks.classification import bert_sep_pcnn as network
elif 'bert_sep_kim_sentence' in args.approach:
    from networks.classification import bert_sep_kim_sentence as network
elif 'bert_sep_mlp_sentence' in args.approach:
    from networks.classification import bert_sep_mlp_sentence as network
elif 'bert_sep_gcae' in args.approach:
    from networks.classification import bert_sep_gcae as network
elif 'bert_sep_sentence' in args.approach:
    from networks.classification import bert_sep_sentence as network
elif 'bert_gru_srk' in args.approach:
    from networks.classification import bert_gru_srk as network
elif 'bert_kim_hat' in args.approach:
    from networks.classification import bert_kim_hat as network
elif 'bert_mlp_hat' in args.approach:
    from networks.classification import bert_mlp_hat as network
elif 'bert_mlp' in args.approach:
    from networks.classification import bert_mlp as network
elif 'bert_kim' in args.approach or 'bert_kim_ewc' in args.approach:
    from networks.classification import bert_kim as network
elif 'bert_gen_hat' in args.approach:
    from networks.classification import bert_gen_hat as network
elif 'bert_gen_single' in args.approach:
    from networks.classification import bert_gen_single as network
elif 'bert_gen' in args.approach:
    from networks.classification import bert_gen as network
elif 'bert_capsule_mask' in args.approach:
    from networks.classification import bert_capsule_mask as network
elif 'bert_adapter_grow' in args.approach:
    from networks.classification import bert_adapter_grow as network
elif 'bert_adapter_mask' in args.approach:
    from networks.classification import bert_adapter_mask as network
elif 'bert_adapter_attention_mask' in args.approach:
    from networks.classification import bert_adapter_attention_mask as network
elif 'bert_adapter_two_modules_mask' in args.approach:
    from networks.classification import bert_adapter_two_modules_mask as network
elif 'bert_adapter_capsule_grow' in args.approach:
    from networks.classification import bert_adapter_capsule_grow as network
elif 'bert_adapter_capsule_mask' in args.approach:
    from networks.classification import bert_adapter_capsule_mask as network
elif 'bert_adapter_capsule' in args.approach:
    from networks.classification import bert_adapter_capsule as network
elif 'bert_adapter_mlp_mask' in args.approach:
    from networks.classification import bert_adapter_mlp_mask as network
elif 'bert_adapter' in args.approach:
    from networks.classification import bert_adapter as network
elif 'bert_lstm' in args.approach:
    from networks.classification import bert_lstm as network
elif 'bert_gru_kan' in args.approach:
    from networks.classification import bert_gru_kan as network
elif 'bert' in args.approach:
    from networks.classification import bert as network


if args.approach == 'w2v_kim_owm_ncl':
    from networks.classification import w2v_kim_owm as network
elif args.approach == 'w2v_kim_ucl_ncl':
    from networks.classification import w2v_kim_ucl as network
elif 'w2v_kim_hat' in args.approach:
    from networks.classification import w2v_kim_hat as network
elif 'w2v_kim_single' in args.approach:
    from networks.classification import w2v_single as network
elif 'w2v_kim_double' in args.approach:
    from networks.classification import w2v_double as network
elif 'w2v_kim_gcae' in args.approach:
    from networks.classification import w2v_gcae as network
elif 'w2v_kim_pcnn' in args.approach:
    from networks.classification import w2v_pcnn as network
elif 'w2v_kim_ana' in args.approach:
    from networks.classification import w2v_ana as network
elif 'w2v_mlp_sentence' in args.approach:
    from networks.classification import w2v_mlp_sentence as network
elif 'w2v_kim_ewc' in args.approach:
    from networks.classification import w2v_kim as network
elif 'w2v_lstm' in args.approach:
    from networks.classification import w2v_lstm as network
elif 'w2v_gru_srk' in args.approach :
    from networks.classification import w2v_gru_srk as network
elif 'w2v_gru_kan' in args.approach:
    from networks.classification import w2v_gru_kan as network
elif 'w2v_kim' in args.approach:
    from networks.classification import w2v_kim as network
