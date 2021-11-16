from config import set_args

# Arguments
args = set_args()

# ----------------------------------------------------------------------
# Language Datasets.
# ----------------------------------------------------------------------


bert_backbone = ['bert','bert_adapter','bert_frozen']
w2v_backbone = ['w2v','w2v_as']

language_dataset = ['asc','dsc','ssc','nli','newsgroup']
image_dataset = ['celeba','femnist','vlcs','cifar10','mnist','fashionmnist','cifar100']


if args.task == 'asc': #aspect sentiment classication
    if args.backbone=='w2v':
        from dataloaders.asc import w2v as dataloader
    elif args.backbone=='w2v_as':
        from dataloaders.asc import w2v_as as dataloader
    elif args.backbone in bert_backbone: #all others
        from dataloaders.asc import bert as dataloader

elif args.task == 'dsc': #document sentiment classication
    if args.backbone=='w2v':
        from dataloaders.dsc import w2v as dataloader
    elif args.backbone in bert_backbone:  # all others
        from dataloaders.dsc import bert as dataloader

elif args.task == 'ssc': #sentence sentiment classication
    if args.backbone=='w2v':
        from dataloaders.ssc import w2v as dataloader
    elif args.backbone in bert_backbone:  # all others
        from dataloaders.ssc import bert as dataloader

elif args.task == 'nli': #natural language inference
    if args.backbone=='w2v':
        from dataloaders.nli import w2v as dataloader
    elif args.backbone in bert_backbone:  # all others
        from dataloaders.nli import bert as dataloader

elif args.task == 'newsgroup': #20newsgroup
    if args.backbone=='w2v':
        from dataloaders.newsgroup import w2v as dataloader
    elif args.backbone in bert_backbone:  # all others
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
# Image approaches.
# ----------------------------------------------------------------------

if args.task in image_dataset:
    if args.baselines == 'hat':
        from approaches.classification import cnn_hat as approach
        if args.backbone == 'mlp': from networks.classification import mlp_hat as network
        if args.backbone == 'cnn': from networks.classification import cnn_hat as network
    elif args.baseline=='ucl':
        from approaches.classification import cnn_ucl as approach
        if args.backbone == 'mlp': from networks.classification import mlp_ucl as network
        if args.backbone == 'cnn': from networks.classification import cnn_ucl as network
    elif args.baseline=='owm':
        from approaches.classification import cnn_owm as approach
        if args.backbone == 'mlp': from networks.classification import mlp_owm as network
        if args.backbone == 'cnn': from networks.classification import cnn_owm as network
    elif args.baseline=='acl':
        from approaches.classification import cnn_acl as approach
        if args.backbone == 'mlp': from networks.classification import mlp_acl as network
        if args.backbone == 'cnn': from networks.classification import cnn_acl as network
    elif args.baseline=='cat':
        from approaches.classification import cnn_cat as approach
        if args.backbone == 'mlp': from networks.classification import mlp_cat as network
        if args.backbone == 'cnn': from networks.classification import cnn_cat as network
    elif args.baseline=='derpp':
        from approaches.classification import cnn_derpp as approach
        if args.backbone == 'mlp': from networks.classification import mlp as network
        if args.backbone == 'cnn': from networks.classification import cnn as network
    elif args.baselines=='one':
        from approaches.classification import cnn_one as approach
        if args.backbone == 'mlp': from networks.classification import mlp as network
        if args.backbone == 'cnn': from networks.classification import cnn as network
    if args.baselines == 'ncl':
        from approches.classification import cnn_ncl as approach
        if args.backbone == 'mlp': from networks.classification import mlp as network
        if args.backbone == 'cnn': from networks.classification import cnn as network
    elif args.baseline=='mtl':
        from approaches.classification import cnn_mtl as approach
        if args.backbone == 'mlp': from networks.classification import mlp as network
        if args.backbone == 'cnn': from networks.classification import cnn as network
    elif args.baseline=='ewc':
        from approaches.classification import cnn_ewc as approach
        if args.backbone == 'mlp': from networks.classification import mlp as network
        if args.backbone == 'cnn': from networks.classification import cnn as network
    elif args.baseline=='hal':
        from approaches.classification import cnn_hal as approach
        if args.backbone == 'mlp': from networks.classification import mlp as network
        if args.backbone == 'cnn': from networks.classification import cnn as network

# ----------------------------------------------------------------------
# Lanaguage approaches.
# ----------------------------------------------------------------------
if args.task in language_dataset:
    
    if args.backbone in w2v_backbone:
        if args.baseline == 'owm':
            from approaches.classification import w2v_cnn_owm as approach
            from networks.classification import w2v_kim_owm as network
        elif args.baseline == 'ucl':
            from approaches.classification import w2v_cnn_ucl as approach
            from networks.classification import w2v_kim_ucl as network
        elif args.baseline=='ncl':
            from approaches.classification import w2v_ncl as approach
            from networks.classification import w2v_kim as network
        elif args.baseline=='one':
            from approaches.classification import w2v_one as approach
            from networks.classification import w2v_kim as network
        elif args.baseline=='hat':
            from approaches.classification import w2v_cnn_hat as approach
            from networks.classification import w2v_kim_hat as network
        elif args.baseline=='ewc':
            from approaches.classification import w2v_cnn_ewc as approach
            from networks.classification import w2v_kim as network
        elif args.baseline=='kan' :
            from approaches.classification import w2v_rnn_kan as approach
            from networks.classification import w2v_gru_kan as network
        elif args.baseline=='srk':
            from approaches.classification import w2v_rnn_srk as approach
            from networks.classification import w2v_gru_srk as network
        if args.baseline == 'derpp':
            from approaches.classification import w2v_cnn_derpp as approach
            from networks.classification import w2v_kim as network
        elif args.baseline == 'a-gem':
            from approaches.classification import w2v_cnn_agem as approach
            from networks.classification import w2v_kim as network
        elif args.baseline == 'l2':
            from approaches.classification import w2v_cnn_l2 as approach
            from networks.classification import w2v_kim as network


    # Args -- baseline
    if args.backbone == 'bert_frozen':
        if args.baseline == 'ucl':
            from approaches.classification import bert_cnn_ucl as approach
            from networks.classification import bert_kim_ucl as network
        if args.baseline == 'derpp':
            from approaches.classification import bert_cnn_derpp as approach
            from networks.classification import bert_kim as network
        elif args.baseline == 'gem':
            from approaches.classification import bert_cnn_gem as approach
            from networks.classification import bert_kim as network
        elif args.baseline == 'a-gem':
            from approaches.classification import bert_cnn_agem as approach
            from networks.classification import bert_kim as network
        elif args.baseline == 'owm':
            from approaches.classification import bert_cnn_owm as approach
            from networks.classification import bert_kim_owm as network
        elif args.baseline=='one':
            from approaches.classification import bert_cnn_one as approach
        elif args.baseline == 'ncl':
            from approaches.classification import bert_cnn_ncl as approach
            from networks.classification import bert_kim as network
        elif args.baseline == 'srk':
            from approaches.classification import bert_rnn_srk as approach
            from networks.classification import bert_gru_srk as network
        elif args.baseline=='kan':
            from approaches.classification import bert_rnn_kan as approach
            from networks.classification import bert_gru_kan as network
        elif args.baseline=='hat':
            from approaches.classification import bert_cnn_hat as approach
            from networks.classification import bert_kim_hat as network
        elif args.baseline=='ewc':
            from approaches.classification import bert_cnn_ewc as approach
            from networks.classification import bert_kim as network
        elif args.baseline=='mtl':
            from approaches.classification import bert_cnn_mtl as approach
            from networks.classification import bert_kim as network





    if args.backbone == 'bert':
        if args.baseline=='one':
            from approaches.classification import bert_one as approach
            from networks.classification import bert as network
        if args.baseline=='ncl':
            from approaches.classification import bert_ncl as approach
            from networks.classification import bert as network
        if args.baseline == 'mtl':
            from approaches.classification import bert_mtl as approach
            from networks.classification import bert as network
    
    
    if args.backbone == 'bert_adapter':
        if args.baseline == 'derpp':
            from approaches.classification import bert_adapter_derpp as approach
            from networks.classification import bert_adapter as network
        elif args.baseline=='one':
            from approaches.classification import bert_adapter_one as approach
            from networks.classification import bert_adapter as network
        elif args.baseline=='ncl':
            from approaches.classification import bert_adapter_ncl as approach
            from networks.classification import bert_adapter as network
        elif args.baseline == 'ucl':
            from approaches.classification import bert_adapter_ucl as approach
            from networks.classification import bert_adapter_ucl as network
        elif args.baseline=='ewc':
            from approaches.classification import bert_adapter_ewc as approach
            from networks.classification import bert_adapter as network
        elif args.baseline == 'a-gem':
            from approaches.classification import bert_adapter_agem as approach
            from networks.classification import bert_adapter as network
        elif args.baseline == 'gem':
            from approaches.classification import bert_adapter_gem as approach
            from networks.classification import bert_adapter as network
        elif args.baseline=='l2':
            from approaches.classification import bert_adapter_l2 as approach
            from networks.classification import bert_adapter as network
        elif args.baseline=='owm':
            from approaches.classification import bert_adapter_owm as approach
            from networks.classification import bert_adapter_owm as network
        elif args.baseline=='ctr' or  args.baseline=='b-cl':
            from approaches.classification import bert_adapter_capsule_mask as approach
            from networks.classification import bert_adapter_capsule_mask as network
        elif args.baseline=='classic':
            from approaches.classification import bert_adapter_mask as approach
            from networks.classification import bert_adapter_mask as network
        elif args.baseline == 'mtl':
            from approaches.classification import bert_mtl as approach
            from networks.classification import bert_adapter as network
        elif args.baseline == 'hat':
            from approaches.classification import bert_adapter_mask as approach
            from networks.classification import bert_adapter_mask as network
