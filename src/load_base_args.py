from config import set_args

# Arguments
args = set_args()

# Currently, 4 tasks included in TIL-Classification Category

#TODO: some base arguments can be set here

def load():
    # ============= dataset base ==================
    #TODO: need to double check nlp datasets, in particular the adapter
    if args.task == 'asc': #aspect sentiment classication
        args.ntasks = 19
        args.num_train_epochs = 10
        args.xusemeval_num_train_epochs = 10
        args.bingdomains_num_train_epochs = 30
        args.bingdomains_num_train_epochs_multiplier = 3
        args.nepochs = 100

    if args.task == 'dsc': #document sentiment classication
        args.ntasks = 10
        args.num_train_epochs = 20
        args.nepochs = 100


    if args.task == 'newsgroup': #aspect sentiment classication
        args.ntasks = 10
        args.class_per_task = 2
        args.num_train_epochs = 10
        args.nepochs = 100



    if args.task == 'celeba':
        if args.ntasks != 20: args.ntasks = 10
        args.nclasses = 2
        args.train_data_size = 100
        args.data_size = 'full'
        args.image_size = 32
        args.image_channel = 3
        args.nepochs = 1000

    if args.task == 'cifar10':
        if args.ntasks != 20: args.ntasks = 10
        args.nclasses = 10
        args.image_size = 32
        args.image_channel = 3
        args.nepochs = 1000


    if args.task == 'cifar100':
        if args.ntasks != 20: args.ntasks = 10
        args.nclasses = 100
        args.image_size = 32
        args.image_channel = 3
        args.nepochs = 1000


    if args.task == 'mnist':
        if args.ntasks != 20: args.ntasks = 10
        args.nclasses = 10
        args.image_size = 28
        args.image_channel = 1
        args.nepochs = 1000

    if args.task == 'femnist':
        if args.ntasks != 20: args.ntasks = 10
        args.nclasses = 62
        args.data_size = 'full'
        args.image_size = 28
        args.image_channel = 1
        args.train_data_size = 6200
        args.nepochs = 1000


    # ============= adapter building base ==================
    if 'adapter_ucl' in args.approach:
        args.build_adapter_ucl = True
    if 'adapter_owm' in args.approach:
        args.build_adapter_owm = True
    # ============= approach base ==================


    if 'acl' in args.approach:
        args.buffer_size = 128
        args.buffer_percent = 0.02
        args.alpha = 0.5
        args.beta = 0.5

    if 'merge' in args.approach:
        args.temp=1
        args.base_temp=1
        args.aux_net = True
        args.amix = True
        args.amix_head = True
        args.amix_head_norm = True
        args.attn_type = 'self'
        args.task_based = True
        args.mix_type = 'Attn-HCHP-Outside'
        args.naug = 1

    if 'cat' in args.approach:
        args.n_head = 5
        args.similarity_detection = 'auto'
        args.loss_type = 'multi-loss-joint-Tsim'

    if 'hal' in args.approach:
        args.buffer_size = 128
        args.buffer_percent = 0.02
        args.gamma = 0.1
        args.beta = 0.5
        args.hal_lambda = 0.1

    if 'derpp' in args.approach:
        args.buffer_size = 128
        args.buffer_percent = 0.02
        args.alpha = 0.5
        args.beta = 0.5

    if 'ucl' in args.approach:
        args.ratio=0.125
        args.beta=0.002
        args.lr_rho=0.01
        args.alpha=5
        args.optimizer='SGD'
        args.clipgrad=100
        args.lr_min=2e-6
        args.lr_factor=3
        args.lr_patience=5

    # ============= experiemnt base ==================
    if args.ent_id:
        args.resume_model = True
        args.eval_only = True
        args.eval_batch_size = 1

    if args.eval_only:
        args.resume_model = True
        if args.ntasks == 10 and not args.eval_each_step:
            args.resume_from_task = 9
        elif args.ntasks == 20 and not args.eval_each_step:
            args.resume_from_task = 19

    if args.unseen:
        args.ntasks_unseen = 10


    #================ analysis base ================

    if args.exit_after_first_task:
        args.nepochs = 1
        args.num_train_epochs = 1
        args.xusemeval_num_train_epochs = 1
        args.bingdomains_num_train_epochs = 1
        args.bingdomains_num_train_epochs_multiplier = 1
        args.eval_only = False

    return args