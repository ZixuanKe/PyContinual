import torch
import os
import numpy as np
import logging
import sys
import random
from config import set_args
args = set_args()

# if you want to use the reset parameters --------------------------
if args.use_predefine_args:
    import load_base_args
    args = load_base_args.load()


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#
if not args.multi_gpu: torch.autograd.set_detect_anomaly(True)

# ----------------------------------------------------------------------
# Create needed folder, results name, reult matrix, if any.
# ----------------------------------------------------------------------

if args.output=='':
    args.output='./res/'+args.scenario+'/'+args.task+'/'+args.backbone+'_'+args.baseline+'_'+str(args.note)+'.txt'

model_path = './models/'+args.scenario+'/'+args.task+'/'
res_path = './res/'+args.scenario+'/'+args.task+'/'

if not os.path.isdir(res_path): os.makedirs(res_path)
if not os.path.isdir(model_path): os.makedirs(model_path)


performance_output=args.output+'_performance'
performance_output_forward=args.output+'_forward_performance'
f1_macro_output=args.output+'_f1_macro'
f1_macro_output_forward=args.output+'_forward_f1_macro'

precision_avg_output=args.output+'_precision_avg'
precision_avg_output_forward=args.output+'_forward_precision_avg'
recall_avg_output=args.output+'_recall_avg'
recall_avg_output_forward=args.output+'_forward_recall_avg'
f1_avg_output=args.output+'_f1_avg'
f1_avg_output_forward=args.output+'_forward_f1_avg'

performance_d_prec=args.output+'_performance_d_prec'
performance_d_recall=args.output+'_performance_d_reacll'
performance_d_f1=args.output+'_performance_d_f1'


acc=np.zeros((args.ntasks,args.ntasks),dtype=np.float32)
lss=np.zeros((args.ntasks,args.ntasks),dtype=np.float32)
f1_macro=np.zeros((args.ntasks,args.ntasks),dtype=np.float32)

precision_avg=np.zeros((args.ntasks,args.ntasks),dtype=np.float32)
recall_avg=np.zeros((args.ntasks,args.ntasks),dtype=np.float32)
f1_avg=np.zeros((args.ntasks,args.ntasks),dtype=np.float32)

base_model_path = args.model_path
base_resume_from_file = args.resume_from_file

#
# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)  # for random sample
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]'); sys.exit()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

classification_tasks = ['asc', 'dsc', 'nli', 'ssc', 'newsgroup',
                        'celeba', 'femnist', 'cifar10', 'mnist', 'fashionmnist', 'cifar100','vlcs']
extraction_tasks = ['ner', 'ae']



def resume_checkpoint(appr,net):
    if args.resume_model:
        checkpoint = torch.load(args.resume_from_file)
        net.load_state_dict(checkpoint['model_state_dict'])
        logger.info('resume_from_file: '+str(args.resume_from_file))


    if args.resume_model:
        if hasattr(appr, 'mask_pre'): appr.mask_pre = torch.load(args.resume_from_file+'_mask_pre') # not in state_dict
        if hasattr(appr, 'mask_back'): appr.mask_back = torch.load(args.resume_from_file+'_mask_back')

        #for GEM
        if hasattr(appr, 'buffer'): appr.buffer = torch.load(args.resume_from_file+'_buffer') # not in state_dict
        if hasattr(appr, 'grad_dims'): appr.grad_dims = torch.load(args.resume_from_file+'_grad_dims') # not in state_dict
        if hasattr(appr, 'grads_cs'): appr.grads_cs = torch.load(args.resume_from_file+'_grads_cs') # not in state_dict
        if hasattr(appr, 'grads_da'): appr.grads_da = torch.load(args.resume_from_file+'_grads_da') # not in state_dict
        if hasattr(appr, 'history_mask_pre'): appr.history_mask_pre = torch.load(args.resume_from_file+'_history_mask_pre') # not in state_dict
        if hasattr(appr, 'similarities'): appr.similarities = torch.load(args.resume_from_file+'_similarities') # not in state_dict
        if hasattr(appr, 'check_federated'): appr.check_federated = torch.load(args.resume_from_file+'_check_federated') # not in state_dict
